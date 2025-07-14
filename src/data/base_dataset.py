from typing import *
from torch import Tensor
from src.utils import StepTracker

import os
from copy import deepcopy
import numpy as np
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils import rotation_diff, homogenize_points, inverse_c2w
from src.data.easy_dataset import EasyDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class BaseDataset(EasyDataset):
    def __init__(self,
        opt: Options,
        name: Optional[str] = None,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
    ):
        super().__init__()

        self.opt = opt
        self.training = training
        self.step_tracker = step_tracker

        name = name if name is not None else opt.dataset_name
        self.name = name

        self.root = opt.all_file_dirs_train[name] if training else opt.all_file_dirs_test[name]
        if self.root.endswith("/"):
            self.root = self.root[:-1]
        self.sample_dirs = os.listdir(self.root)

        # For dataset-specific settings
        self.min_bounded_gap, self.max_bounded_gap, self.test_max_bounded_gap = None, None, None
        self.depth_quantile, self.is_static = None, None
        self.dataset_args = deepcopy(opt.default_dataset_args)
        if name in opt.dataset_args:
            self.dataset_args.update(opt.dataset_args[name])
        for k, v in self.dataset_args.items():
            setattr(self, k, v)
        if not self.training and self.test_max_bounded_gap is not None:
            self.max_bounded_gap = self.test_max_bounded_gap  # reset to test max bounded gap

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return_dict = self._try_getitem(index)

        # 1. Valid points
        if "depth_mask" in return_dict and self.name not in ["dycheck", "nvidia", "tapvid3d"]:
            depth_masks = return_dict["depth_mask"]  # (F, H, W)
            H, W = depth_masks.shape[-2:]
            valid_ratio = depth_masks.sum(dim=(-2, -1)) / (H * W)  # (F,)
        else:
            valid_ratio = torch.ones(len(return_dict["image"]), dtype=torch.float32)  # (F,)
        # 2. Camera rotation
        R = return_dict["C2W"][:self.opt.num_input_frames, :3, :3].unsqueeze(0)  # (1, F, 3, 3)

        # Filter some outliers
        if (
            ## 1. Too many invalid points
            valid_ratio.min() < self.opt.min_valid_ratio or
            ## 2. Camera rotates too much
            rotation_diff(R[:, 1:], R[:, :-1]).max() > self.opt.max_rotation_diff
        ):
            return self.__getitem__(np.random.randint(len(self)))  # re-sample
        else:
            return return_dict

    # ONLY implement this function for each dataset
    def _try_getitem(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples into a batch of tensors.
        """
        return_batch = {}
        for key in batch[0].keys():
            if key in ["name", "uid", "track_world", "track_xy", "visibility"]:
                return_batch[key] = [sample[key] for sample in batch]  # a list of str or Tensor
            else:
                return_batch[key] = torch.stack([sample[key] for sample in batch])  # a Tensor
        return return_batch

    ################################ Helper Functions ################################

    def _frame_sample(self, num_frames: int, fixed_start_idx: Optional[int] = None) -> Tuple[List[int], List[int]]:
        frame_idxs = np.arange(num_frames, dtype=int)
        F_all, F_in, F_out = len(frame_idxs), self.opt.num_input_frames, self.opt.num_output_frames

        if not self.training:
            min_gap = max_gap = self.max_bounded_gap
        else:
            min_gap, max_gap = self.min_bounded_gap, self.max_bounded_gap

        # Pick a video clip
        if F_all >= max_gap:
            gap = np.random.randint(min_gap, max_gap+1)
            if fixed_start_idx is not None:
                start_idx = fixed_start_idx
            else:
                if self.training:
                    start_idx = np.random.randint(0, F_all - gap + 1)
                else:
                    start_idx = 0
            clip_frame_idxs = frame_idxs[start_idx:start_idx+gap]
        else:  # some samples may be too short for bounded sampling
            gap = F_all
            clip_frame_idxs = frame_idxs

        # Sample input and output frames
            ## 1. Uniform input and output
        if self.opt.view_sampling_type == "uniform":
            input_frame_idxs = clip_frame_idxs[np.linspace(0, gap-1, F_in, dtype=int)].tolist()
            output_frame_idxs = clip_frame_idxs[np.linspace(0, gap-1, F_out, dtype=int)].tolist()
            ## 2. Uniform input and arbitrary output, including input in output if possible
        elif self.opt.view_sampling_type == "arbitrary_output_include_input":
            input_frame_idxs = clip_frame_idxs[np.linspace(0, gap-1, F_in, dtype=int)].tolist()
            if self.training:
                if np.random.rand() < self.opt.include_input_prob:  # include input in output
                    if F_out > F_in:  # input + arbitrary
                        output_frame_idxs = input_frame_idxs + \
                            np.random.choice(clip_frame_idxs, size=F_out-F_in, replace=F_out-F_in > gap).tolist()
                    else:  # sample from input
                        output_frame_idxs = np.random.choice(input_frame_idxs, size=F_out, replace=F_out > F_in).tolist()
                else:  # arbitrary output
                    output_frame_idxs = np.random.choice(clip_frame_idxs, size=F_out, replace=F_out > gap).tolist()
            else:
                output_frame_idxs = clip_frame_idxs[np.linspace(0, gap-1, F_out, dtype=int)].tolist()
        else:
            raise ValueError(f"Invalid view sampling type: [{self.opt.view_sampling_type}]")

        return sorted(input_frame_idxs), sorted(output_frame_idxs)

    def _data_augment(self,
        images: Tensor,
        depths: Optional[Tensor],
        depth_masks: Optional[Tensor],
        C2W: Tensor,
        fxfycxcy: Tensor,
        tracks_world: Optional[Tensor],
        tracks_xy: Optional[Tensor],
        visibilities: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        images, C2W, fxfycxcy = images.clone(), C2W.clone(), fxfycxcy.clone()  # not inplace
        if depths is not None:
            depths, depth_masks = depths.clone(), depth_masks.clone()  # not inplace
        if tracks_world is not None:
            tracks_world, tracks_xy, visibilities = tracks_world.clone(), tracks_xy.clone(), visibilities.clone()  # not inplace

        assert images.ndim == 4  # (F, C, H, W)
        H, W = images.shape[-2:]
        new_H, new_W = self.opt.input_res

        # Resize and CenterCrop images
        scale_factor = max(new_H / H, new_W / W)
        if self.training and scale_factor <= 1.:
            scale_factor = np.random.uniform(scale_factor, 1.)
        scaled_H, scaled_W = round(H * scale_factor), round(W * scale_factor)
        # Assume we don't have to worry about changing the intrinsics based on how the images are rounded
        images = tvT.Resize((scaled_H, scaled_W), tvT.InterpolationMode.BICUBIC)(images)  # intrinsic not changed
        # Adjust the intrinsics to account for the cropping
        images = tvT.CenterCrop((new_H, new_W))(images)  # intrinsic changed
        fxfycxcy[:, 0] *= (scaled_W / new_W)
        fxfycxcy[:, 1] *= (scaled_H / new_H)

        # (Optional) Resize and CenterCrop depths
        if depths is not None:
            depths = tvT.Resize((scaled_H, scaled_W), tvT.InterpolationMode.NEAREST_EXACT)(depths)
            depths = tvT.CenterCrop((new_H, new_W))(depths)
            depth_masks = tvT.Resize((scaled_H, scaled_W), tvT.InterpolationMode.NEAREST_EXACT)(depth_masks)
            depth_masks = tvT.CenterCrop((new_H, new_W))(depth_masks)

        # (Optional) Resize and CenterCrop XY coordinates of tracks
        if tracks_world is not None:
            tracks_xy[..., 0] = tracks_xy[..., 0] * (scaled_W / W) - (scaled_W - new_W) / 2.
            tracks_xy[..., 1] = tracks_xy[..., 1] * (scaled_H / H) - (scaled_H - new_H) / 2.
            tracks_xy = tracks_xy.round().long()
            visibilities = visibilities & (tracks_xy[..., 0] >= 0) & (tracks_xy[..., 0] < new_W) & \
                (tracks_xy[..., 1] >= 0) & (tracks_xy[..., 1] < new_H)

        return images.clamp(0., 1.), depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities

    def _camera_normalize(self, C2W: Tensor, depths: Optional[Tensor], tracks_world: Optional[Tensor]) -> Union[Tensor, Optional[Tensor], Optional[Tensor]]:
        C2W = C2W.clone()  # not inplace

        if self.opt.camera_norm_type == "none":
            pass
        elif self.opt.camera_norm_type == "canonical":
            transform = inverse_c2w(C2W[0, ...])  # (4, 4)
            C2W = transform.unsqueeze(0) @ C2W  # (F, 4, 4); first camera is canonical

            if tracks_world is not None:
                tracks_world_homo = homogenize_points(tracks_world)  # (F, N, 4)
                tracks_world_homo = tracks_world_homo @ transform.T.unsqueeze(0)  # (F, N, 4)
                tracks_world = tracks_world_homo[..., :3] / tracks_world_homo[..., 3:]  # (F, N, 3)
        else:
            raise ValueError(f"Invalid camera normalization type: [{self.opt.camera_norm_type}]")

        return C2W, depths, tracks_world
