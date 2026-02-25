from typing import *
from torch import Tensor

import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from einops import rearrange
from kiui.op import inverse_sigmoid

from gsplat import rasterization


class Camera:
    def __init__(self,
        C2W: Tensor, fxfycxcy: Tensor, h: int, w: int,
        znear: float = 0.01, zfar: float = 1000.,
    ):
        self.fxfycxcy = fxfycxcy.clone().float()
        self.C2W = C2W.clone().float()
        try:
            self.W2C = self.C2W.inverse()
        except:  # some numerical issues
            self.W2C = torch.zeros_like(self.C2W)
            self.W2C[:3, :3] = self.C2W[:3, :3].T
            self.W2C[:3, 3] = -self.C2W[:3, 3] @ self.C2W[:3, :3]
            self.W2C[3, 3] = 1.

        self.znear = znear
        self.zfar = zfar
        self.h = h
        self.w = w

        fx, fy, cx, cy = self.fxfycxcy[0], self.fxfycxcy[1], self.fxfycxcy[2], self.fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)  # `tanHalfFovX` actually
        self.tanfovY = 1 / (2 * fy)  # `tanHalfFovY` actually
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        self.K = torch.zeros(3, 3, dtype=torch.float32, device=self.C2W.device)
        self.K[0, 0] = fx * self.w
        self.K[1, 1] = fy * self.h
        self.K[0, 2] = cx * self.w
        self.K[1, 2] = cy * self.h
        self.K[2, 2] = 1.

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, device=fovX.device)

            z_sign = 1

            P[0, 0] = 2 * znear / (right - left)
            P[1, 1] = 2 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovX, self.fovY, self.shiftX, self.shiftY).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.C2W[:3, 3]


class GaussianModel:
    def __init__(self):
        self.xyz = None
        self.color = None
        self.scale = None
        self.rotation = None
        self.opacity = None

        self.sh_degree = 0

    @property
    def d_sh(self):
        return (self.sh_degree + 1) ** 2

    def set_data(self, xyz: Tensor, color: Tensor, scale: Tensor, rotation: Tensor, opacity: Tensor, sh_degree: int = 0) -> "GaussianModel":
        self.xyz = xyz
        self.color = color
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.sh_degree = sh_degree
        assert self.color.shape[1] == self.d_sh * 3
        return self

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GaussianModel":
        self.xyz = self.xyz.to(device, dtype)
        self.color = self.color.to(device, dtype)
        self.scale = self.scale.to(device, dtype)
        self.rotation = self.rotation.to(device, dtype)
        self.opacity = self.opacity.to(device, dtype)
        return self

    def save_ply(self, path: str, opacity_threshold: float = 0., compatible: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        sh = rearrange(self.color.detach().cpu().numpy(), "n (k rgb) -> n k rgb", rgb=3)
        rgb = (sh[:, 0] * 255.).clip(0., 255.).astype(np.uint8)
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # Filter out points with low opacity
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        sh = sh[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        rgb = rgb[mask]

        # Invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(torch.from_numpy(opacity)).numpy()
            scale = torch.log(torch.from_numpy(scale) + 1e-8).numpy()
            sh = (torch.from_numpy(sh) - 0.5).numpy() / 0.28209479177387814

        dtype_full = [(attribute, "f4") for attribute in self._construct_list_of_attributes()]
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, sh, opacity, scale, rotation, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str, compatible: bool = True):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        sh = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ), axis=1)
        for i in range(self.d_sh - 3):
            sh = np.concatenate((sh, np.asarray(plydata.elements[0][f"f_rest_{i}"])[:, None]), axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scale = np.stack((
            np.asarray(plydata.elements[0]["scale_0"]),
            np.asarray(plydata.elements[0]["scale_1"]),
            np.asarray(plydata.elements[0]["scale_2"]),
        ), axis=1)
        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"]),
            np.asarray(plydata.elements[0]["rot_1"]),
            np.asarray(plydata.elements[0]["rot_2"]),
            np.asarray(plydata.elements[0]["rot_3"]),
        ), axis=1)

        self.xyz = torch.from_numpy(xyz).float()
        self.color = torch.from_numpy(sh).float()
        self.opacity = torch.from_numpy(opacity).float()
        self.scale = torch.from_numpy(scale).float()
        self.rotation = torch.from_numpy(rotation).float()

        if compatible:
            self.opacity = torch.sigmoid(self.opacity)
            self.scale = torch.exp(self.scale)
            self.color = 0.28209479177387814 * self.color + 0.5

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(3):
            l.append(f"f_dc_{i}")
        for i in range(self.d_sh - 3):
            l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self.rotation.shape[1]):
            l.append(f"rot_{i}")
        return l


def render(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: Tensor,
    fxfycxcy: Tensor,
    znear: float = 0.001,
    zfar: float = 100.,
    radius_clip: float = 0.,
    bg_color: Union[Tensor, Tuple[float, float, float]] = (0., 0., 0.),
    scaling_modifier: float = 1.,
    render_mode: Literal["RGB", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    distributed: bool = False,
):
    if C2W.ndim == 2:  # batch size 1
        C2W = C2W.unsqueeze(0)
        fxfycxcy = fxfycxcy.unsqueeze(0)

    viewpoint_cameras = [
        Camera(C2W[i], fxfycxcy[i], height, width, znear, zfar)
        for i in range(C2W.shape[0])
    ]  # [(B, 4, 4), (B, 4)]

    if not isinstance(bg_color, Tensor):
        bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)
        bg_color = bg_color.unsqueeze(0).repeat(C2W.shape[0], 1)  # (B, 3)
    else:
        bg_color = bg_color.to(C2W.device, dtype=torch.float32)

    pc = pc.to(dtype=torch.float32)

    render_colors, render_alphas, _ = rasterization(
        means=pc.xyz,  # (N, 3)
        quats=pc.rotation,  # (N, 4)
        scales=pc.scale * scaling_modifier,  # (N, 3)
        opacities=pc.opacity.squeeze(-1),  # (N,)
        colors=rearrange(pc.color, "n (k rgb) -> n k rgb", rgb=3) if pc.sh_degree > 0 else pc.color,  # (N(, K=`sh_degree`), 3)
        viewmats=torch.stack([cam.world_view_transform.transpose(0, 1) for cam in viewpoint_cameras]),  # (B, 4, 4)
        Ks=torch.stack([cam.K for cam in viewpoint_cameras]),  # (B, 3, 3)
        width=int(width),
        height=int(height),
        near_plane=znear,
        far_plane=zfar,
        radius_clip=radius_clip,
        sh_degree=pc.sh_degree if pc.sh_degree > 0 else None,
        backgrounds=bg_color,  # (B, 3)
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
        distributed=distributed,
    )

    render_colors = rearrange(render_colors, "b h w c -> b c h w")
    render_alphas = rearrange(render_alphas, "b h w c -> b c h w")

    return_dict = {
        "image": render_colors[:, :3, ...],  # (B, 3, H, W)
        "alpha": render_alphas,  # (B, 1, H, W)
    }
    if render_mode in ["RGB+D", "RGB+ED"]:
        return_dict["depth"] = render_colors[:, 3:, ...]  # (B, 1, H, W)
    return return_dict
