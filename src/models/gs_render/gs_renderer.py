from typing import *
from torch import Tensor

import torch
import torch.nn.functional as tF
from einops import rearrange
from torch_scatter import scatter_add, scatter_max

from src.models.gs_render.gs_util import GaussianModel, render
from src.options import Options
from src.utils import unproject_depth


class GaussianRenderer:
    def __init__(self, opt: Options):
        self.opt = opt

        # Create a mask for the spherical harmonics coefficients
        # Having a large DC component and small view-dependent components at initialization
        self.d_sh = (self.opt.sh_degree + 1) ** 2
        self.sh_mask = torch.ones((self.d_sh,), dtype=torch.float32, requires_grad=False)
        for degree in range(1, opt.sh_degree + 1):
            self.sh_mask[degree**2 : (degree+1)**2] = 0.1 * (0.25**degree)

    @torch.autocast("cuda", dtype=torch.float32)
    def render(self,
        model_outputs: Dict[str, Tensor],
        input_C2W: Optional[Tensor],
        input_fxfycxcy: Optional[Tensor],
        C2W: Tensor,
        fxfycxcy: Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        bg_color: Union[Tensor, Tuple[float, float, float]] = (0., 0., 0.),
        scaling_modifier: float = 1.,
        in_image_format: bool = True,
        after_activation: bool = False,
        return_pc: bool = False,
        training: bool = False,
    ):
        if not in_image_format:
            assert height is not None and width is not None
            assert "xyz" in model_outputs  # depth must be in image format

        color, scale, rotation, opacity = model_outputs["color"], model_outputs["scale"], model_outputs["rotation"], model_outputs["opacity"]
        depth, xyz = model_outputs.get("depth", None), model_outputs.get("xyz", None)
        assert depth is not None or xyz is not None  # at least one of depth and XYZ must be provided

        # (Optional) Dynamic 3DGS
        motion_color, motion_opacity = model_outputs.get("motion_color", None), model_outputs.get("motion_opacity", None)
        motion_scale, motion_rotation = model_outputs.get("motion_scale", None), model_outputs.get("motion_rotation", None)

        # (Optional) Confidence
        conf = model_outputs.get("conf", torch.ones_like(opacity))
        motion_conf = model_outputs.get("motion_conf", None)

        # Rendering resolution could be different from input resolution
        H = height if height is not None else color.shape[-2]
        W = width if width is not None else color.shape[-1]

        # Reshape for rendering
        if in_image_format:
            color = rearrange(color, "b v c h w -> b (v h w) c")
            scale = rearrange(scale, "b v c h w -> b (v h w) c")
            rotation = rearrange(rotation, "b v c h w -> b (v h w) c")
            opacity = rearrange(opacity, "b v c h w -> b (v h w) c")
            conf = rearrange(conf, "b v c h w -> b (v h w) c")
            # (Optional) Dynamic 3DGS
            if motion_color is not None:
                motion_color = rearrange(motion_color, "b v c h w -> b (v h w) c")
            if motion_scale is not None:
                motion_scale = rearrange(motion_scale, "b v c h w -> b (v h w) c")
            if motion_rotation is not None:
                motion_rotation = rearrange(motion_rotation, "b v c h w -> b (v h w) c")
            if motion_opacity is not None:
                motion_opacity = rearrange(motion_opacity, "b v c h w -> b (v h w) c")
            if motion_conf is not None:
                motion_conf = rearrange(motion_conf, "b v c h w -> b (v h w) c")

        # Prepare XYZ for rendering
        if xyz is None:
            assert depth is not None and input_C2W is not None and input_fxfycxcy is not None
            depth = self.depth_activation(depth) if not after_activation else depth
            xyz = unproject_depth(depth.squeeze(2), input_C2W, input_fxfycxcy)
        xyz = self.xyz_activation(xyz) if not after_activation else xyz
        offset = model_outputs.get("offset", torch.zeros_like(xyz))
        xyz = xyz + (self.offset_activation(offset) if not after_activation else offset)
        if in_image_format:
            if depth is not None:
                depth = rearrange(depth, "b v c h w -> b (v h w) c")
            xyz = rearrange(xyz, "b v c h w -> b (v h w) c")

        # Prepare other attributes
        if not after_activation:
            color = self.color_activation(color) if motion_color is None else self.color_activation(motion_color)
            opacity = self.opacity_activation(opacity) if motion_opacity is None else self.opacity_activation(motion_opacity)
            scale = self.scale_activation(scale) if motion_scale is None else self.scale_activation(motion_scale)
            rotation = self.rotation_activation(rotation) if motion_rotation is None else self.rotation_activation(motion_rotation)
        else:
            color = color if motion_color is None else motion_color
            opacity = opacity if motion_opacity is None else motion_opacity
            scale = scale if motion_scale is None else motion_scale
            rotation = rotation if motion_rotation is None else motion_rotation
        conf = conf if motion_conf is None else motion_conf

        # Spherical harmonics
        color = rearrange(color, "b vhw (k rgb) -> b vhw k rgb", rgb=3)
        assert color.shape[-2] == self.d_sh
        color = color * self.sh_mask[None, None, :, None].to(color.device)  # (B, (V*H*W), K, 3)
        color = rearrange(color, "b vhw k rgb -> b vhw (k rgb)")

        # Gaussian usage
        gaussian_usage = (opacity > self.opt.opacity_threshold).float().mean(1).squeeze(-1)  # (B,)

        pcs = []
        for i in range(C2W.shape[0]):
            pcs.append(GaussianModel().set_data(xyz[i], color[i], scale[i], rotation[i], opacity[i], self.opt.sh_degree))

        (B, V), device = C2W.shape[:2], C2W.device
        images = torch.zeros(B, V, 3, H, W, dtype=torch.float32, device=device)
        alphas = torch.zeros(B, V, 1, H, W, dtype=torch.float32, device=device)
        depths = torch.zeros(B, V, 1, H, W, dtype=torch.float32, device=device)

        voxel_ratios = []
        for i in range(C2W.shape[0]):
            # (Optional) GS pruning based on opacity
            if self.opt.prune_ratio > 0.:
                _xyz, _color, _scale, _rotation, _opacity = pcs[i].xyz, pcs[i].color, pcs[i].scale, pcs[i].rotation, pcs[i].opacity  # (N, 3), (N, 3), (N, 3), (N, 4), (N, 1)

                if training:  # prune a fix ratio of gaussians
                    num_gaussians = _xyz.shape[0]
                    keep_ratio = 1. - self.opt.prune_ratio
                    random_ratio = keep_ratio * self.opt.random_ratio
                    keep_ratio = keep_ratio - random_ratio
                    num_keep = int(keep_ratio * num_gaussians)
                    num_keep_random = int(random_ratio * num_gaussians)
                    # Rank by opacity
                    idxs = _opacity.argsort(dim=0, descending=True)
                    keep_idxs = idxs[:num_keep]
                    if num_keep_random > 0:
                        rest_idxs = idxs[num_keep:]
                        random_idxs = rest_idxs[torch.randperm(rest_idxs.shape[0])[:num_keep_random]]
                        keep_idxs = torch.cat([keep_idxs, random_idxs], dim=0)
                else:  # filter via opacity threshold
                    keep_idxs = (_opacity > self.opt.opacity_threshold).squeeze(-1)

                # Pruning
                _xyz = _xyz[keep_idxs]
                _color = _color[keep_idxs]
                _scale = _scale[keep_idxs]
                _rotation = _rotation[keep_idxs]
                _opacity = _opacity[keep_idxs]
                pcs[i].set_data(_xyz, _color, _scale, _rotation, _opacity)

            # (Optional) Voxelization
            if self.opt.voxel_size > 0.:
                _xyz, _color, _scale, _rotation, _opacity = pcs[i].xyz, pcs[i].color, pcs[i].scale, pcs[i].rotation, pcs[i].opacity  # (N, 3), (N, 3), (N, 3), (N, 4), (N, 1)
                _conf = conf[i].squeeze(-1)  # (N,)
                if self.opt.prune_ratio > 0.:
                    _conf = _conf[keep_idxs]

                voxel_indices = (_xyz / self.opt.voxel_size).round().long()  # (N, 3)
                unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

                ## Compute softmax weights per voxel
                conf_voxel_max, _ = scatter_max(_conf, inverse_indices, dim=0)
                conf_exp = torch.exp(_conf - conf_voxel_max[inverse_indices])
                voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # (N_unique_voxels,)
                weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6))[:, None]  # (N, 1)
                voxel_ratios.append(voxel_weights.shape[0] / _xyz.shape[0])

                ## Aggregate per voxel
                _xyz = scatter_add(_xyz * weights, inverse_indices, dim=0)
                _color = scatter_add(_color * weights, inverse_indices, dim=0)
                _scale = scatter_add(_scale * weights, inverse_indices, dim=0)
                _rotation = tF.normalize(scatter_add(_rotation * weights, inverse_indices, dim=0), p=2, dim=-1)
                _opacity = scatter_add(_opacity * weights, inverse_indices, dim=0)
                pcs[i].set_data(_xyz, _color, _scale, _rotation, _opacity)

            render_results = render(
                pcs[i],
                H,
                W,
                C2W[i],
                fxfycxcy[i],
                self.opt.znear,
                self.opt.zfar,
                self.opt.gsplat_radius_clip,
                bg_color,
                scaling_modifier,
                self.opt.gsplat_render_mode,
                self.opt.gsplat_rasterize_mode,
                self.opt.gsplat_distributed,
            )
            images[i, ...] = render_results["image"]
            alphas[i, ...] = render_results["alpha"]
            if "depth" in render_results:
                depths[i, ...] = render_results["depth"]

        output_dict = {
            "image": images,  # (B, V, 3, H, W)
            "alpha": alphas,  # (B, V, 1, H, W)
        }
        if "depth" in render_results:
            output_dict["depth"] = depths  # (B, V, 1, H, W)

        if return_pc:
            output_dict["pc"] = pcs
        if self.opt.voxel_size > 0.:
            output_dict["voxel_ratio"] = torch.tensor(voxel_ratios, dtype=torch.float32, device=device)  # (B,)
        output_dict["gaussian_usage"] = gaussian_usage  # (B,)

        return output_dict

    ################################ 3DGS Attribute Activation Functions ################################

    def color_activation(self, color: Tensor):
        if self.opt.color_act_type == "identity":
            return color.clamp(0., 1.)  # [0, 1]
        elif self.opt.color_act_type == "tanh":
            return torch.tanh(color) * 0.5 + 0.5  # [0, 1]
        else:
            raise ValueError(f"Invalid color activation type: [{self.opt.color_act_type}]")

    def scale_activation(self, scale: Tensor):
        if self.opt.scale_act_type == "identity":
            return scale
        elif self.opt.scale_act_type == "range":
            return self.opt.scale_act_min + \
                (self.opt.scale_act_max - self.opt.scale_act_min) * (torch.tanh(scale) * 0.5 + 0.5)  # [scale_act_min, scale_act_max]
        elif self.opt.scale_act_type == "exp":
            return torch.exp(scale + self.opt.scale_act_bias).clamp(max=self.opt.scale_act_max)  # (0, scale_act_max]
        elif self.opt.scale_act_type == "softplus":
            return (self.opt.scale_act_scale * tF.softplus(scale)).clamp(max=self.opt.scale_act_max)  # (0, scale_act_max]
        else:
            raise ValueError(f"Invalid scale activation type: [{self.opt.scale_act_type}]")

    def rotation_activation(self, rotation: Tensor, dim: int = -1):
        return tF.normalize(rotation, p=2, dim=dim)  # [-1, 1]

    def opacity_activation(self, opacity: Tensor):
        return torch.tanh(opacity + self.opt.opacity_act_bias) * 0.5 + 0.5  # [0, 1]

    def depth_activation(self, depth: Tensor):
        if self.opt.depth_act_type == "identity":
            return depth.clamp(min=self.opt.znear, max=self.opt.zfar)
        elif self.opt.depth_act_type == "range":
            return self.opt.znear + \
                (self.opt.zfar - self.opt.znear) * (torch.tanh(depth) * 0.5 + 0.5)  # [znear, zfar]
        elif self.opt.depth_act_type == "norm_exp":
            d = depth.norm(dim=-1, keepdim=True)
            depth = depth / d.clamp(min=1e-8) * torch.expm1(d)
            return depth.clamp(min=self.opt.znear, max=self.opt.zfar)  # [znear, zfar]
        elif self.opt.depth_act_type == "exp":
            return torch.exp(depth).clamp(min=self.opt.znear, max=self.opt.zfar)  # [znear, zfar]
        else:
            raise ValueError(f"Invalid depth activation type: [{self.opt.depth_act_type}]")

    def xyz_activation(self, xyz: Tensor):
        if self.opt.xyz_act_type == "identity":
            return xyz
        elif self.opt.xyz_act_type == "norm_exp":
            d = xyz.norm(dim=-1, keepdim=True)
            xyz = xyz / d.clamp(min=1e-8) * torch.expm1(d)
            return xyz
        elif self.opt.xyz_act_type == "inv_log":
            return torch.sign(xyz) * torch.expm1(torch.abs(xyz))
        else:
            raise ValueError(f"Invalid xyz activation type: [{self.opt.xyz_act_type}]")

    def offset_activation(self, offset: Tensor):
        return offset
