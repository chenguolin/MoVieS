from typing import *
from src.utils import StepTracker

import os
import h5py
import numpy as np
import imageio.v2 as iio
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils import unproject_depth
from src.data.base_dataset import BaseDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class SpringDataset(BaseDataset):
    def __init__(self,
        opt: Options,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
    ):
        super().__init__(opt, "spring", training, step_tracker)

    def _try_getitem(self, index: int) -> Dict[str, Any]:
        sample_dir = self.sample_dirs[index]
        sample_dir = os.path.join(self.root, sample_dir)
        uid = os.path.basename(sample_dir)

        # Image & camera are preprocessed in the preprocess script
        file_paths = os.listdir(sample_dir)
        image_paths = sorted(set([os.path.join(sample_dir, file_path) for file_path in file_paths if file_path.endswith(".jpg")]))
        num_frames = len(image_paths)

        # Sample frames
        input_frame_idxs, output_frame_idxs = self._frame_sample(num_frames)
        timesteps = torch.tensor([idx for idx in input_frame_idxs + output_frame_idxs]).float()  # (F,)
        timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())  # (F,)
            ## Randomly fixed timesteps for output frames
        if self.is_static and np.random.rand() < 0.5 and self.training:
            F_in = self.opt.num_input_frames
            timesteps[F_in:] = torch.ones_like(timesteps[F_in:]) * torch.rand(1)
        selected_frame_idxs = set(input_frame_idxs + output_frame_idxs)  # to avoid duplicate file access

        # Load images (.jpg)
        images = {
            idx: tvT.ToTensor()(iio.imread(image_paths[idx]))
            for idx in selected_frame_idxs
        }
        images = torch.stack([images[idx] for idx in (input_frame_idxs + output_frame_idxs)]).float()  # (F, 3, H, W)
        F, (H, W) = images.shape[0], images.shape[-2:]

        # Load cameras (.npz): ground-truth
        camera_paths = sorted([os.path.join(sample_dir, file_path) for file_path in file_paths if file_path.endswith(".npz")])
        assert len(image_paths) == len(camera_paths)
        cameras = {
            idx: np.load(camera_paths[idx])
            for idx in selected_frame_idxs
        }
        C2W = torch.from_numpy(np.stack([cameras[idx]["camera_pose"] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, 4, 4)
        fxfycxcy = torch.from_numpy(np.stack([cameras[idx]["fxfycxcy"] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, 4)

        # Load depths (.h5): ground-truth
        with h5py.File(os.path.join(self.root + "_gtdepth", f"{uid}.h5"), "r") as f:
            assert len(image_paths) == len(f["depth"])
            disparities, depth_masks = {}, {}
            for idx in selected_frame_idxs:
                depth = f["depth"][idx].astype(np.float32)
                depth_mask = (depth != 0.)
                # depth = cv2.inpaint(depth, ~depth_mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)  # fill depth holes for visualization
                depth[~depth_mask] = 1e4  # not fill depth holes
                disparities[idx] = 1. / depth.clip(1e-4, 1e4)
                depth_masks[idx] = depth_mask
        disparities = torch.from_numpy(np.stack([disparities[idx] for idx in (input_frame_idxs + output_frame_idxs)])).float()  # (F, H, W)
        depth_masks = torch.from_numpy(np.stack([depth_masks[idx] for idx in (input_frame_idxs + output_frame_idxs)])).bool()  # (F, H, W)
            ## Depth masks
        depth_masks = depth_masks & (1./self.opt.zfar <= disparities) & (disparities <= 1./self.opt.znear)  # (F, H, W)
        disparities = disparities.clamp(1./self.opt.zfar, 1./self.opt.znear)
        depths = 1. / disparities

        # Dummy 3D point tracking
        tracks_world = torch.zeros(F, 1, 3).float()  # (F, N=1, 3)
        tracks_xy = torch.zeros(F, 1, 2).long()  # (F, N=1, 2)
        visibilities = torch.zeros(F, 1).bool()  # (F, N=1)

        # Data augmentation
        images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities = \
            self._data_augment(images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities)
            ## (Optional) Mask by quantile after downsampling for efficiency
        if self.depth_quantile is not None:
            depth_masks = depth_masks & (depths > np.quantile(depths.numpy(), self.depth_quantile))
            depth_masks = depth_masks & (depths < np.quantile(depths.numpy(), 1.-self.depth_quantile))

        # Camera normalization
            ## 1. Transform 3D tracks if needed
            ## 2. Scale 3D tracks and depths if needed
        C2W, depths, tracks_world = self._camera_normalize(C2W, depths, tracks_world)

        # (Optional) Scaling depth and camera pose and 3D points according to the XYZ normalization; cf. VGGT
        if self.opt.norm_xyz:
            F_in = self.opt.num_input_frames
            _xyz = unproject_depth(depths[None, :F_in, ...], C2W[None, :F_in, ...], fxfycxcy[None, :F_in, ...]).squeeze(0)
            _xyz_norm = (_xyz.norm(dim=1) * depth_masks[:F_in, ...]).sum() / (depth_masks[:F_in, ...].sum() + 1e-6)
            scaling_factor = 1. / (_xyz_norm + 1e-6) * self.opt.camera_norm_unit
            depths = depths * scaling_factor
            C2W[:, :3, 3] = C2W[:, :3, 3] * scaling_factor
            tracks_world = tracks_world * scaling_factor

        return {
            "name": self.name,                         # str
            "uid": uid,                                # str
            "timestep": timesteps,                     # Tensor: (F,)
            "image": images,                           # Tensor: (F, 3, H, W)
            "C2W": C2W,                                # Tensor: (F, 4, 4)
            "fxfycxcy": fxfycxcy,                      # Tensor: (F, 4)
            "depth": depths,                           # Tensor: (F, H, W)
            "depth_mask": depth_masks.bool(),          # BoolTensor: (F, H, W)
            "track_world": tracks_world,               # Tensor: (F, N, 3)
            "track_xy": tracks_xy.long(),              # LongTensor: (F, N, 2)
            "visibility": visibilities.bool(),         # BoolTensor: (F, N)
            "depth_weight": torch.tensor(1.),          # Tensor: (1,)
            "motion_weight": torch.tensor(0.),         # Tensor: (1,); dynamic without motion supervision
        }
