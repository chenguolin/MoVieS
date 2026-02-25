from typing import *
from src.utils import StepTracker

import os
import math
import numpy as np
from decord import VideoReader, cpu
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils import unproject_depth
from src.data.base_dataset import BaseDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Stereo4dDataset(BaseDataset):
    def __init__(self,
        opt: Options,
        training: bool = True,
        step_tracker: Optional[StepTracker] = None,
    ):
        super().__init__(opt, "stereo4d", training, step_tracker)

        # # Filter out invalid samples
        # _new_sample_dirs = []
        # for sample_dir in self.sample_dirs:
        #     sample_dir = os.path.join(self.root, sample_dir)
        #     if len(os.listdir(sample_dir)) == 4:
        #         _new_sample_dirs.append(sample_dir)
        # self.sample_dirs = _new_sample_dirs

    def _try_getitem(self, index: int) -> Dict[str, Any]:
        sample_dir = self.sample_dirs[index]
        sample_dir = os.path.join(self.root, sample_dir)
        uid = os.path.basename(sample_dir)

        # Video & camera are preprocessed in Stereo4D
        video_path = os.path.join(sample_dir, f"{uid}-left_rectified.mp4")
        npz_path = os.path.join(self.root.replace("processed", "npz"), f"{uid}.npz")
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)

        # Sample frames
        input_frame_idxs, output_frame_idxs = self._frame_sample(num_frames)
        timesteps = torch.tensor([idx for idx in input_frame_idxs + output_frame_idxs]).float()  # (F,)
        timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())  # (F,)
            ## Randomly fixed timesteps for output frames
        if self.is_static and np.random.rand() < 0.5 and self.training:
            F_in = self.opt.num_input_frames
            timesteps[F_in:] = torch.ones_like(timesteps[F_in:]) * torch.rand(1)
        selected_frame_idxs = set(input_frame_idxs + output_frame_idxs)  # to avoid duplicate file access

        # Load images (.mp4)
        images = {
            idx: tvT.ToTensor()(vr[idx].asnumpy())
            for idx in selected_frame_idxs
        }
        images = torch.stack([images[idx] for idx in (input_frame_idxs + output_frame_idxs)]).float()  # (F, 3, H, W)
        F, (H, W) = images.shape[0], images.shape[-2:]

        # Load cameras (.npz)
        npz = load_dataset_npz(npz_path)
        extrs_rectified = npz["extrs_rectified"][(input_frame_idxs + output_frame_idxs), ...]  # (F, 3, 4)
        hfov, cameras, C2W, fxfycxcy = 60., [], [], []  # `60.` hard-coded for Stereo4D
        for idx in range(len(extrs_rectified)):
            intr_normalized = {
                "fx": (1 / 2.) / math.tan(math.radians(hfov / 2.)),
                "fy": (1 / 2.) / math.tan(math.radians(hfov / 2.)) * W / H,
                "cx": 0.5, "cy": 0.5, "k1": 0., "k2": 0.,
            }
            camera = CameraAZ({
                "extr": extrs_rectified[idx][:3, :],  # (3, 4)
                "intr_normalized": intr_normalized,  # Dict[str, float]: fx fy cx cy k1 k2
            })
            C2W.append(camera.get_c2w())  # (4, 4)
            fxfycxcy.append(np.array([intr_normalized["fx"], intr_normalized["fy"], intr_normalized["cx"], intr_normalized["cy"]]))  # (4,)
            cameras.append(camera)  # a list of `CameraAZ`
        C2W = torch.from_numpy(np.stack(C2W, axis=0)).float()  # (F, 4, 4)
        fxfycxcy = torch.from_numpy(np.stack(fxfycxcy, axis=0)).float()  # (F, 4)
        cameras: List[CameraAZ]

        # Load 3D tracks and depths (.npz)
        tracks_world_np = npz["track3d"].transpose(1, 0, 2)[(input_frame_idxs + output_frame_idxs), ...]  # (F, N, 3)
        tracks_world = torch.from_numpy(tracks_world_np).float()  # (F, N, 3)
        visibilities = ~torch.isnan(tracks_world).any(dim=-1)  # (F, N)
        depths = torch.full((F, H*W), float("inf"))  # (F, H*W)
        tracks_xy = []  # (F, N, 2)
        for f in range(F):
            ## 3D points projected to the image plane to get depths
            xy, valid_mask, depth = cameras[f].world_2_pix_np(tracks_world_np[f], H, W, self.opt.znear)  # (N, 2), (N,), (N,)
            xy, valid_mask, depth = torch.from_numpy(xy).long(), torch.from_numpy(valid_mask).bool(), torch.from_numpy(depth).float()
            valid_mask = visibilities[f] = visibilities[f] & valid_mask  # (N,)
            tracks_xy.append(xy)  # (N, 2)
            linear_idxs = xy[valid_mask, 1] * W + xy[valid_mask, 0]  # (M,)
            depths[f].scatter_reduce_(
                dim=-1, index=linear_idxs, src=depth[valid_mask], reduce="amin", include_self=True)
        depths = depths.reshape(F, H, W)  # (F, H, W)
        depth_masks = (self.opt.znear <= depths) & (depths <= self.opt.zfar)  # (F, H, W)
        depths = depths.clamp(self.opt.znear, self.opt.zfar)
        tracks_xy = torch.stack(tracks_xy, dim=0)  # (F, N, 2)

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
            "depth_weight": torch.tensor(0.2),         # Tensor: (1,); depth predicted from SEA-RAFT
            "motion_weight": torch.tensor(0.2),        # Tensor: (1,); motion predicted from BootsTAP
        }


################################ Stereo4D Functions ################################


def load_dataset_npz(path):
    """
    Load released npz format
    """
    with open(path, "rb") as f:
        data_zip = np.load(f)
        data = {}
        for k in data_zip.keys():
            data[k] = data_zip[k]
    # --------------
    # Camera intrinsics
    # --------------
    data["meta_fov"] = {
        "start_yaw_in_degrees": data["fov_bounds"][0],
        "end_yaw_in_degrees": data["fov_bounds"][1],
        "start_tilt_in_degrees": data["fov_bounds"][2],
        "end_tilt_in_degrees": data["fov_bounds"][3],
    }
    data.pop("fov_bounds")
    # --------------
    # Camera poses
    # --------------
    c2w = data["camera2world"]  # (T, 3, 4)
    R = c2w[:, :, :3]
    t = c2w[:, :, 3:]

    # Compute inverse: R^T and new translation
    R_inv = np.transpose(R, (0, 2, 1))  # Transpose R
    t_inv = -np.matmul(R_inv, t)
    data["extrs_rectified"] = np.concatenate([R_inv, t_inv], axis=-1)
    data.pop("camera2world")
    # --------------
    # 3D tracks
    # --------------
    lengths = data["track_lengths"]
    shape = (len(lengths), len(data["timestamps"]), 3)
    tracks = np.full(shape, np.nan)
    tracks[
        np.repeat(np.arange(lengths.shape[0]), lengths),
        data["track_indices"], :
    ] = data["track_coordinates"]
    data["track3d"] = tracks
    data.pop("track_lengths")
    data.pop("track_indices")
    data.pop("track_coordinates")
    return data


class CameraAZ:
    def __init__(self, from_json: Dict[str, np.ndarray]):
        """
        Initialize the object with JSON data.
        """
        self.extr = from_json["extr"]  # (3, 4); world-to-camera
        self.intr_normalized = from_json["intr_normalized"]  # Dict[str, float]: fx fy cx cy k1 k2

    def __str__(self):
        return f"extr: \n{self.extr}\n intr_normalized: \n{self.intr_normalized}"

    def to_json_format(self):
        return {
            "extr": self.extr,  # (3, 4)
            "intr_normalized": self.intr_normalized,  # Dict[str, float] (6,)
        }

    def get_c2w(self) -> np.ndarray:
        """
        Get the camera-to-world transformation matrix.

        Outputs:
            numpy.ndarray: A 4x4 camera-to-world transformation matrix
        """
        w2c = np.concatenate((self.extr, np.array([[0., 0., 0., 1.]])), axis=0)
        c2w = np.linalg.inv(w2c)  # (4, 4)
        return c2w

    def get_hfov_deg(self) -> float:
        """
        Get the horizontal field of view (HFOV) in degrees.
        """
        return math.degrees(2 * np.arctan(0.5 / self.intr_normalized["fx"]))

    def get_intri_matrix(self, imh: int, imw: int) -> np.ndarray:
        """
        Get the intrinsic matrix.

        Inputs:
            - `imh`: The height of the image
            - `imw`: The width of the image

        Outputs:
            numpy.ndarray: A 3x3 intrinsic matrix
        """
        return np.array([
            [self.intr_normalized["fx"] * imw, 0, self.intr_normalized["cx"] * imw],
            [0., self.intr_normalized["fy"] * imh, self.intr_normalized["cy"] * imh],
            [0., 0., 1.],
        ])

    def pix_2_world_np(
        self,
        xy: np.ndarray,
        depth: np.ndarray,
        valid_depth_min: float = 0.001,
        valid_depth_max: float = 100.,
    ):
        """Unproject points from ndc from world frame.

        Inputs:
            - `xy`: (N, 2)
            - `depth`: (H, W)

        Outputs:
            - `xyz_world`: (N, 3)
            - `valid_mask`: (N,)
        """

        _, dim = xy.shape
        assert dim == 2
        imh, imw = depth.shape

        valid_mask = (
            (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < imw) & (xy[:, 1] < imh)
        )

        x_cam = (
            xy[..., 0] / imw - self.intr_normalized["cx"]
        ) / self.intr_normalized["fx"]
        y_cam = (
            xy[..., 1] / imh - self.intr_normalized["cy"]
        ) / self.intr_normalized["fy"]
        z_cam = np.ones_like(xy[..., 0])
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        x_query = np.clip(np.round(xy[:, 0]).astype(int), 0, imw - 1)
        y_query = np.clip(np.round(xy[:, 1]).astype(int), 0, imh - 1)
        depth_values = depth[y_query, x_query]

        valid_mask = (
            valid_mask
            & (depth_values > valid_depth_min)
            & (depth_values < valid_depth_max)
        )

        xyz_cam = depth_values[:, None] * xyz_cam

        xyz_world = (self.extr[:3, :3].T @ (xyz_cam - self.extr[:3, 3]).T).T
        return xyz_world, valid_mask

    def world_2_pix_np(
        self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.001
    ):
        """Project points from world frame to screen space.

        Inputs:
            - `xyz_world`: (N, 3)
            - `imh`: The height of the image
            - `imw`: The width of the image
            - `min_depth`: The minimum depth value

        Outputs:
            - `xy`: (N, 2)
            - `valid_mask`: (N,)
            - `depth`: (N,)
        """
        xyz_world_homo = np.concatenate(
            (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
        )
        xyz_homo = (
            self.get_intri_matrix(imh, imw) @ self.extr @ xyz_world_homo.T
        ).T  # (N, 3)
        depth = xyz_homo[:, 2]
        xy = (xyz_homo[:, :2] / xyz_homo[:, 2:]).round()
        valid_mask = (
            (xy[:, 0] >= 0)
            & (xy[:, 1] >= 0)
            & (xy[:, 0] <= imw-1)
            & (xy[:, 1] <= imh-1)
            & (depth > min_depth)
        )
        return xy, valid_mask, depth
