from typing import *
from src.utils import StepTracker

import os
import h5py
import numpy as np
import imageio.v2 as iio
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils import unproject_depth, homogenize_points
from src.data.base_dataset import BaseDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class DynamicreplicaDataset(BaseDataset):
    def __init__(self,
        opt: Options,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
    ):
        super().__init__(opt, "dynamicreplica", training, step_tracker)

        # 3D point tracking is only available for left views
        self.sample_dirs = [sample_dir for sample_dir in self.sample_dirs if sample_dir.endswith("_left")]

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

        # Load 3D tracks (.h5)
        with h5py.File(os.path.join(self.root + "_gttraj", f"{uid}.h5"), "r") as f:
            assert len(image_paths) == len(f["track_xy"])
            tracks_xy = {
                idx: torch.from_numpy(f["track_xy"][idx]).long()
                for idx in selected_frame_idxs
            }
            visibilities = {
                idx: (
                    torch.from_numpy(f["visibility"][idx]).bool() &
                    (tracks_xy[idx][:, 0] >= 0) & (tracks_xy[idx][:, 0] < W) &
                    (tracks_xy[idx][:, 1] >= 0) & (tracks_xy[idx][:, 1] < H)
                )
                for idx in selected_frame_idxs
            }
        tracks_xy = torch.stack([tracks_xy[idx] for idx in (input_frame_idxs + output_frame_idxs)])  # (F, N, 2)
        visibilities = torch.stack([visibilities[idx] for idx in (input_frame_idxs + output_frame_idxs)])  # (F, N)
            ## Unproject XY to world XYZ
        tracks_world = []
        for f in range(F):
            x_img = tracks_xy[f, :, 0]  # (N,)
            y_img = tracks_xy[f, :, 1]  # (N,)
            fx, fy, cx, cy = fxfycxcy[f, 0], fxfycxcy[f, 1], fxfycxcy[f, 2], fxfycxcy[f, 3]
            fx, fy, cx, cy = fx * W, fy * H, cx * W, cy * H
            z = depths[f, y_img.clip(0, H-1), x_img.clip(0, W-1)]  # (N,)
            visibilities[f] = visibilities[f] & depth_masks[f, y_img.clip(0, H-1), x_img.clip(0, W-1)]  # also mask out invalid depths
            x_cam = (x_img - cx) * z / fx
            y_cam = (y_img - cy) * z / fy
            z_cam = z
            track_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, 3)
            track_cam_homo = homogenize_points(track_cam)  # (N, 4)
            track_world = track_cam_homo @ C2W[f].T  # (N, 4)
            track_world = track_world[:, :3] / track_world[:, 3:]  # (N, 3)
            track_world[~visibilities[f]] = float("nan")
            tracks_world.append(track_world)
        tracks_world = torch.stack(tracks_world).float()  # (F, N, 3)

        # Data augmentation
        images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities = \
            self._data_augment(images, depths, depth_masks, C2W, fxfycxcy, tracks_world, tracks_xy, visibilities)
            ## (Optional) Mask by quantile after downsampling for efficiency
        if self.min_depth_quantile is not None:
            depth_masks = depth_masks & (depths > np.quantile(depths.numpy(), self.min_depth_quantile))
        if self.max_depth_quantile is not None:
            depth_masks = depth_masks & (depths < np.quantile(depths.numpy(), self.max_depth_quantile))

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
            "depth_weight": torch.tensor(0.5),         # Tensor: (1,); noisy depth
            "motion_weight": torch.tensor(0.5),        # Tensor: (1,); noisy depth
        }
