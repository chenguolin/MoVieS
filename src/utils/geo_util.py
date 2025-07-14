from typing import *
from numpy import ndarray
from torch import Tensor, LongTensor, BoolTensor

import numpy as np
import torch


def rotation_diff(R1: Tensor, R2: Tensor) -> Tensor:
    """Compute the angle between two rotation matrices by Rodrigues formula.

    Inputs:
        - `R1`: (B, V, 3, 3)
        - `R2`: (B, V, 3, 3)

    Outputs:
        - `angle`: (B, V) in degrees
    """
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)  # (B, V, 3, 3); R_rel = R1^T @ R2
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B, V); trace of `R_rel`

    # Compute rotation angle by Rodrigues formula
    cos_theta = (trace - 1.) / 2.
    cos_theta = torch.clamp(cos_theta, -1., 1.)  # for numerical stability

    return torch.rad2deg(torch.acos(cos_theta))  # (B, V)


def project_points(
    xyz_world: Tensor,
    C2W: Tensor,
    fxfycxcy: Tensor,
    H: int,
    W: int,
    znear: float = 0.001,
    zfar: float = 100.,
    margin: int = 0,
) -> Tuple[LongTensor, BoolTensor]:
    """Project points in 3D world coordinate to image plane and valid_mask.

    Inputs:
        - `xyz_world`: (N, 3)
        - `C2W`: (4, 4)
        - `fxfycxcy`: (4,)
        - `H`: image height
        - `W`: image width
        - `znear`: near plane
        - `zfar`: far plane
        - `margin`: margin for valid mask

    Outputs:
        - `xy`: (N, 2); LongTensor
        - `valid_mask`: (N,); BoolTensor
    """
    W2C = inverse_c2w(C2W)
    homo_xyz_world = homogenize_points(xyz_world)  # (N, 4)
    xyz_camera = homo_xyz_world @ W2C.T  # (N, 4)
    xyz_camera = xyz_camera[..., :3] / xyz_camera[..., 3:]  # (N, 3)

    x, y, z = xyz_camera.unbind(dim=-1)  # (N,)
    valid_z = (z > znear) & (z < zfar)

    fx, fy, cx, cy = fxfycxcy.unbind(dim=-1)  # (1,)
    u = (fx * x + cx * z) / z
    v = (fy * y + cy * z) / z
    u_round = (u * W).round().long()  # (N,)
    v_round = (v * H).round().long()  # (N,)

    valid_mask = (u_round >= margin) & (u_round <= W-1-margin) & (v_round >= margin) & (v_round <= H-1-margin) & valid_z

    # Considering occlusion, only keep the closest points
    if valid_mask.any():
        u_valid = u_round[valid_mask]  # (M,)
        v_valid = v_round[valid_mask]  # (M,)
        z_valid = z[valid_mask]  # (M,)
        idx_valid = torch.arange(len(z))[valid_mask]  # (M,)

        linear_idx = v_valid * W + u_valid  # (M,)
        depth_buffer = torch.full((H * W,), float("inf"), device=z.device)
        depth_buffer.scatter_reduce_(dim=0, index=linear_idx, src=z_valid, reduce="amin", include_self=True)

        visible = z_valid == depth_buffer[linear_idx]  # not occluded

        is_visible = torch.zeros_like(valid_mask)
        is_visible[idx_valid[visible]] = True
        valid_mask = valid_mask & is_visible

    return torch.stack([u_round, v_round], dim=-1).long(), valid_mask.bool()


def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (B, V, H, W)
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `xyz_world`: (B, V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    B, V, H, W = depth_map.shape

    depth_map = depth_map.reshape(B*V, H, W).float()
    C2W = C2W.reshape(B*V, 4, 4).float()
    K = fxfycxcy_to_intrinsics(fxfycxcy).reshape(B*V, 3, 3).float()

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
    xyz = xyz_map.view(B*V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.bmm(xyz, inverse_k(K).transpose(1, 2))
    xyz_map = xyz.view(B*V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = homogenize_points(xyz_map)
    xyz_world = torch.bmm(C2W, xyz_homo.reshape(B*V, -1, 4).transpose(1, 2)).to(dtype)  # (B*V, 4, H*W)
    xyz_world = xyz_world[:, :3, ...] / xyz_world[:, 3:, ...]  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(B, V, 3, H, W)
    return xyz_world


def plucker_ray(h: int, w: int, C2W: Tensor, fxfycxcy: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `h`: image height
        - `w`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    y = (y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / h
    x = (x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / w
    x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, h*w, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, h*w, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, h*w, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, h*w, 3)

    ray_o = ray_o.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)


def inverse_c2w(C2W: Tensor) -> Tensor:
    """Compute the inverse of a batch of camera-to-world matrices in a closed form.

    Inputs:
        - `C2W`: (..., 4, 4)

    Outputs:
        - `W2C`: (..., 4, 4)
    """
    # try:
    #     return C2W.inverse()
    # except:  # some numerical issues
    R = C2W[..., :3, :3]  # (..., 3, 3)
    t = C2W[..., :3, 3]   # (..., 3)

    R_inv = R.transpose(-2, -1)  # (..., 3, 3)
    t_inv = -torch.einsum("...ij,...j->...i", R_inv, t)  # (..., 3)

    W2C = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    row3 = torch.tensor([0., 0., 0., 1.], device=R.device, dtype=R.dtype).expand_as(W2C[..., 0, :])  # (..., 4)
    W2C = torch.cat([W2C, row3.unsqueeze(-2)], dim=-2)  # (..., 4, 4)
    return W2C


def inverse_k(K: Tensor) -> Tensor:
    """Compute the inverse of a batch of intrinsics matrices in a closed form.

    Inputs:
        - `K`: (..., 3, 3)

    Outputs:
        - `K_inv`: (..., 3, 3)
    """
    # try:
    #     return K.inverse()
    # except:  # some numerical issues
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    row0 = torch.stack([1. / fx, torch.zeros_like(fx), -cx / fx], dim=-1)  # (..., 3)
    row1 = torch.stack([torch.zeros_like(fy), 1. / fy, -cy / fy], dim=-1)  # (..., 3)
    row2 = torch.tensor([0., 0., 1.], device=fx.device, dtype=fx.dtype).expand_as(row0)  # (..., 3)
    K_inv = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)
    return K_inv


def fxfycxcy_to_intrinsics(fxfycxcy: Tensor) -> Tensor:
    """Convert a batch of fxfycxcy to intrinsics.

    Inputs:
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `intrinsics`: (B, V, 3, 3)
    """
    fx, fy, cx, cy = fxfycxcy.unbind(dim=-1)  # each is (B, V)

    row0 = torch.stack([fx, torch.zeros_like(fx), cx], dim=-1)  # (B, V, 3)
    row1 = torch.stack([torch.zeros_like(fy), fy, cy], dim=-1)  # (B, V, 3)
    row2 = torch.tensor([0., 0., 1.], device=fx.device, dtype=fx.dtype).expand(fx.shape[0], fx.shape[1], 3)  # (B, V, 3)
    intrinsics = torch.stack([row0, row1, row2], dim=-2)  # (B, V, 3, 3)
    return intrinsics


def intrinsics_to_fxfycxcy(intrinsics: Tensor) -> Tensor:
    """Convert a batch of intrinsics to fxfycxcy.

    Inputs:
        - `intrinsics`: (B, V, 3, 3)

    Outputs:
        - `fxfycxcy`: (B, V, 4)
    """
    fx = intrinsics[:, :, 0, 0]
    fy = intrinsics[:, :, 1, 1]
    cx = intrinsics[:, :, 0, 2]
    cy = intrinsics[:, :, 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1)  # (B, V, 4)


def homogenize_points(points: Tensor) -> Tensor:
    """Homogenize a batch of 3D points: (xyz) -> (xyz1).

    Inputs:
        - `points`: (..., 2) or (..., 3)

    Outputs:
        - `points_homo`: (..., 3) or (..., 4)
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: Tensor) -> Tensor:
    """Homogenize a batch of 3D vectors: (xyz) -> (xyz0).

    Inputs:
        - `vectors`: (..., 2) or (..., 3)

    Outputs:
        - `vectors_homo`: (..., 3) or (..., 4)
    """
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


################################ Solving `y = Ax + b` by Least Squares ################################
# Copied from https://github.com/DepthAnything/Video-Depth-Anything/blob/main/utils/util.py


def compute_scale_and_shift(prediction: ndarray, target: ndarray, mask: Optional[ndarray] = None, scale_only: bool = False):
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)


def compute_scale(prediction: ndarray, target: ndarray, mask: Optional[ndarray] = None):
    if mask is None:
        mask = np.ones_like(target)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)

    x_0 = b_0 / (a_00 + 1e-6)

    return x_0


def compute_scale_and_shift_full(prediction: ndarray, target: ndarray, mask: Optional[ndarray] = None):
    if mask is None:
        mask = np.ones_like(target)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1.
    x_1 = 0.

    det = a_00 * a_11 - a_01 * a_01

    if det != 0.:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1
