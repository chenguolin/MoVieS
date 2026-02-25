from typing import *
from torch import Tensor

import torch
from torch import nn
from einops import rearrange

from .conv import FeatureEmbed
from .gs_aggregator import GSAggregator
from .gs_dpt_head import GSDPTHead
from .linear_head import LinearHead

from src.options import Options
from src.utils import convert_to_buffer, zero_init_module, plucker_ray, inverse_c2w, fxfycxcy_to_intrinsics

import sys; sys.path.append("extensions/vggt")
from extensions.vggt.vggt.models.vggt import VGGT
from extensions.vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding


class VGGSplaT(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        assert opt.depth_act_type == "identity"  # activated in VGGT

        self.opt = opt
        self.is_dynamic = is_dynamic = opt.output_motion or opt.dynamic_depth

        vggt = VGGT.from_pretrained("facebook/VGGT-1B")

        # (Optional) Plucker embedding
        if opt.input_plucker:
            self.plucker_embed = FeatureEmbed("3d",
                6,  # hard-coded for plucker embedding
                vggt.aggregator.patch_embed.embed_dim,
                t_ratio=1,
                s_ratio=vggt.aggregator.patch_size,
            )
            self.plucker_ln = nn.GroupNorm(
                num_groups=1,
                num_channels=vggt.aggregator.patch_embed.embed_dim,
            )
            zero_init_module(self.plucker_embed)
            zero_init_module(self.plucker_ln)  # zero init norm weights

        # Encoder
        self.encoder = vggt.aggregator.patch_embed

        # Backbone
        self.backbone = GSAggregator(
            patch_embed="dummy",  # seperate DINO from `GSAggregator`
            extra_dim=9,  # hard-coded for pose token (3: R, 4: t, 2: fxfy)
            time_dim=opt.time_dim if is_dynamic else 0,
            memory_efficient_attention=opt.memory_efficient_attention,
        )
        if opt.vggt_init:
            missing_keys, unexpected_keys = self.backbone.load_state_dict(vggt.aggregator.state_dict(), strict=False)
            for name in missing_keys:
                assert "extra_embed" in name or "time_embed" in name
            for name in unexpected_keys:
                assert "patch_embed" in name

        # Depth head
        # self.depth_head = vggt.depth_head
        self.depth_head = GSDPTHead(  # use custom GSDPTHead for more flexibility
            dim_in=2 * 1024,  # hard-coded from VGGT-1B
            output_dim=2,  # depth (1) + conf (1)
            activation="exp",
            conf_activation="expp1",
            shortcut_dim=-1,  # no shortcut
            time_dim=opt.time_dim if opt.dynamic_depth else 0,
        )
        if opt.vggt_init:
            missing_keys, unexpected_keys = self.depth_head.load_state_dict(vggt.depth_head.state_dict(), strict=False)
            if opt.dynamic_depth:
                for name in missing_keys:
                    assert "time_embed" in name
                assert len(unexpected_keys) == 0
            else:
                assert len(missing_keys) == len(unexpected_keys) == 0

        # 3DGS head
        if opt.splat:
            if opt.use_dpt_splat_head:
                self.splat_head = GSDPTHead(
                    dim_in=2 * 1024,  # hard-coded from VGGT-1B
                    output_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4 + 1,  # SH (rgb: 3) + opacity (1) + scale (3) + rot (4) + conf (1)
                    activation="linear",  # i.e., no activation
                    conf_activation="expp1",
                    shortcut_dim=3,  # image shortcut
                    time_dim=opt.time_dim if opt.dynamic_splat else 0,
                )
            else:
                self.splat_head = LinearHead(
                    dim_in=2 * 1024,  # hard-coded from VGGT-1B
                    output_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4 + 1,  # SH (rgb: 3) + opacity (1) + scale (3) + rot (4) + conf (1)
                    activation="linear",  # i.e., no activation
                    conf_activation="expp1",
                    shortcut_dim=3,  # image shortcut
                    time_dim=opt.time_dim if opt.dynamic_splat else 0,
                )
            if not opt.rendering:
                self.splat_head.requires_grad_(False)

        # Motion head
        if opt.output_motion:
            # self.motion_head = vggt.point_head
            self.motion_head = GSDPTHead(  # use custom GSDPTHead for more flexibility
                dim_in=2 * 1024,  # hard-coded from VGGT-1B
                output_dim=4,  # motion (3) + conf (1)
                activation="linear",  # i.e., no activation; not use the original `inv_log` for stability
                conf_activation="expp1",
                shortcut_dim=-1,  # no shortcut
                time_dim=opt.time_dim,
            )
            if opt.vggt_init:
                missing_keys, unexpected_keys = self.motion_head.load_state_dict(vggt.point_head.state_dict(), strict=False)
                for name in missing_keys:
                    assert "time_embed" in name
                assert len(unexpected_keys) == 0

            if opt.splat and opt.motion_splat and not opt.dynamic_splat:
                if opt.use_dpt_motion_splat_head:
                    self.motion_splat_head = GSDPTHead(
                        dim_in=2 * 1024,  # hard-coded from VGGT-1B
                        output_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4 + 1,  # SH (rgb: 3) + opacity (1) + scale (3) + rot (4) + conf (1)
                        activation="linear",  # i.e., no activation
                        conf_activation="expp1",
                        shortcut_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4,  # static GS as shortcut
                        time_dim=opt.time_dim,
                    )
                else:
                    self.motion_splat_head = LinearHead(
                        dim_in=2 * 1024,  # hard-coded from VGGT-1B
                        output_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4 + 1,  # SH (rgb: 3) + opacity (1) + scale (3) + rot (4) + conf (1)
                        activation="linear",  # i.e., no activation
                        conf_activation="expp1",
                        shortcut_dim=(opt.sh_degree+1)**2 * 3 + 1 + 3 + 4,  # static GS as shortcut
                        time_dim=opt.time_dim,
                    )
                if not opt.rendering:
                    self.motion_splat_head.requires_grad_(False)

        del vggt

        # Handle not used parameters: no gradient & not save to checkpoint
        if opt.freeze_dino_encoder:
            convert_to_buffer(self.encoder, persistent=True)  # pretrained DINOv2
        if opt.freeze_attn_backbone:
            convert_to_buffer(self.backbone, persistent=True)  # pretrained attention backbone

    def forward(self,
        images: Tensor,
        C2W: Tensor,
        fxfycxcy: Tensor,
        input_timesteps: Optional[Tensor] = None,
        output_timesteps: Optional[Tensor] = None,
        frames_chunk_size: int = 8,
        only_frame0: bool = False,
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor], Optional[List[Dict[str, Tensor]]]]:
        return_dict = {}
        image_size = images.shape[-2:]

        if self.opt.freeze_dino_encoder:
            self.encoder.eval()
        if self.opt.freeze_attn_backbone:
            self.backbone.eval()

        # (Optional) Plucker embedding
        if self.opt.input_plucker:
            plucker, _ = plucker_ray(image_size[-2], image_size[-1], C2W, fxfycxcy)  # (B, F_in, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w")
            plucker_embeds = self.plucker_ln(self.plucker_embed(plucker))  # (B, D, F_in, h, w)
            plucker_embeds = rearrange(plucker_embeds, "b d f h w -> (b f) (h w) d")  # (B*F_in, h*w, D)
        else:
            plucker_embeds = 0.

        # (Optional) Prepare pose information
        W2C = inverse_c2w(C2W)  # (B, F_in, 4, 4)
        intrinsics = fxfycxcy_to_intrinsics(fxfycxcy)  # (B, F_in, 3, 3)
        extra_info = extri_intri_to_pose_encoding(W2C, intrinsics, (1, 1)).to(C2W.dtype)  # (B, F_in, 9)

        # (Optional) Prepare timestep information
            ## Input timesteps
        if self.opt.input_timestep:
            assert input_timesteps is not None
            assert self.opt.time_dim % 2 == 0
            freqs = 2 ** torch.arange(self.opt.time_dim // 2, dtype=input_timesteps.dtype, device=input_timesteps.device)
            input_timestep_embeds = torch.cat([
                torch.sin(input_timesteps[..., None] * freqs[None, None, :]),  # (B, F_in, 1) x (1, 1, D)
                torch.cos(input_timesteps[..., None] * freqs[None, None, :]),  # (B, F_in, 1) x (1, 1, D)
            ], dim=-1)  # (B, F_in, D)
        else:
            input_timestep_embeds = None
            ## Output timesteps
        if self.is_dynamic:
            assert output_timesteps is not None
            assert self.opt.time_dim % 2 == 0
            freqs = 2 ** torch.arange(self.opt.time_dim // 2, dtype=output_timesteps.dtype, device=output_timesteps.device)
            output_timestep_embeds = torch.cat([
                torch.sin(output_timesteps[..., None] * freqs[None, None, :]),  # (B, F_out, 1) x (1, 1, D)
                torch.cos(output_timesteps[..., None] * freqs[None, None, :]),  # (B, F_out, 1) x (1, 1, D)
            ], dim=-1)  # (B, F_out, D)
        else:
            output_timestep_embeds = None

        # Encode images
        patch_tokens = self.encoder(
            rearrange(
                (images - self.backbone._resnet_mean) / self.backbone._resnet_std,
                "b f c h w -> (b f) c h w"
            )
        )
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]  # (B*F_in, h*w, D)
        patch_tokens = patch_tokens + plucker_embeds

        # Backbone: `patch_tokens` are provided, so `images` are not really used
        aggregated_tokens_list, patch_start_idx = self.backbone(images, extra_info, input_timestep_embeds, patch_tokens)

        # Depth head
        if not self.opt.dynamic_depth:
            pred_depths, depth_confs = self.depth_head(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )
            pred_depths = rearrange(pred_depths, "b f h w c -> b f c h w")
            return_dict["depth"] = pred_depths  # (B, F_in, 1, H, W)
            return_dict["depth_conf"] = depth_confs  # (B, F_in, H, W)
        else:
            pred_dynamic_depth_list = []
            for i in range(output_timestep_embeds.shape[1]):
                pred_depths, depth_confs = self.depth_head(
                    aggregated_tokens_list,
                    images,
                    patch_start_idx,
                    output_timestep_embeds[:, i],  # (B, D)
                    frames_chunk_size=frames_chunk_size,
                    only_frame0=only_frame0,
                )
                pred_depths = rearrange(pred_depths, "b f h w c -> b f c h w")
                pred_depths = torch.cat([pred_depths, depth_confs.unsqueeze(2)], dim=2)  # (B, F_in, 1+1, H, W)
                pred_dynamic_depth_list.append(pred_depths)
            return_dict["dynamic_depth"] = torch.stack(pred_dynamic_depth_list, dim=1)  # (B, F_out, F_in, 1+1, H, W)

        # 3DGS head
        if self.opt.rendering:
            if self.opt.splat:
                if not self.opt.dynamic_splat:
                    gaussians, gs_confs = self.splat_head(
                        aggregated_tokens_list,
                        images,
                        patch_start_idx,
                        frames_chunk_size=frames_chunk_size,
                    )
                    gaussians = rearrange(gaussians, "b f h w c -> b f c h w")
                    colors, scales, rotations, opacities = torch.split(gaussians, [(self.opt.sh_degree+1)**2 * 3, 3, 4, 1], dim=2)
                    return_dict.update({
                        "color": colors,  # (B, F_in, 3, H, W)
                        "scale": scales,  # (B, F_in, 3, H, W)
                        "rotation": rotations,  # (B, F_in, 4, H, W)
                        "opacity": opacities,  # (B, F_in, 1, H, W)
                        "conf": gs_confs.unsqueeze(2),  # (B, F_in, 1, H, W)
                    })
                else:
                    pred_dynamic_splat_list = []
                    for i in range(output_timestep_embeds.shape[1]):
                        dynamic_gaussians, gs_confs = self.splat_head(
                            aggregated_tokens_list,
                            images,
                            patch_start_idx,
                            output_timestep_embeds[:, i],  # (B, D)
                            frames_chunk_size=frames_chunk_size,
                            only_frame0=only_frame0,
                        )
                        dynamic_gaussians = rearrange(dynamic_gaussians, "b f h w c -> b f c h w")
                        colors, scales, rotations, opacities = torch.split(dynamic_gaussians, [(self.opt.sh_degree+1)**2 * 3, 3, 4, 1], dim=2)
                        pred_dynamic_splat_list.append({
                            "color": colors,  # (B, F_in, 3, H, W)
                            "scale": scales,  # (B, F_in, 3, H, W)
                            "rotation": rotations,  # (B, F_in, 4, H, W)
                            "opacity": opacities,  # (B, F_in, 1, H, W)
                            "conf": gs_confs.unsqueeze(2),  # (B, F_in, 1, H, W)
                        })
                    return_dict["dynamic_splat"] = pred_dynamic_splat_list  # a list of `F_out` dict of (B, F_in, C, H, W)
            else:  # dummy 3DGS features for visualization
                dummy_rot = torch.randn_like(torch.cat([images, images[:, :, 0:1,...]], dim=2))
                dummy_rot = dummy_rot / dummy_rot.norm(dim=2, keepdim=True)
                return_dict.update({
                    "color": images,  # (B, F_in, 3, H, W)
                    "scale": torch.ones_like(images) * 0.0001,  # (B, F_in, 3, H, W)
                    "rotation": dummy_rot,  # (B, F_in, 4, H, W)
                    "opacity": torch.ones_like(images[:, :, 0:1, ...]) * 0.95,  # (B, F_in, 1, H, W)
                })

        # Motion & dynamic GS head
        if self.opt.output_motion:
            pred_motion_list, pred_motion_splat_list = [], []
            for i in range(output_timestep_embeds.shape[1]):
                ## Motion
                pred_motions, motion_confs = self.motion_head(
                    aggregated_tokens_list,
                    images,
                    patch_start_idx,
                    output_timestep_embeds[:, i],  # (B, D)
                    frames_chunk_size=frames_chunk_size,
                    only_frame0=only_frame0,
                )
                pred_motions = rearrange(pred_motions, "b f h w c -> b f c h w")
                pred_motions = torch.cat([pred_motions, motion_confs.unsqueeze(2)], dim=2)  # (B, F_in, 3+1, H, W)
                pred_motion_list.append(pred_motions)
                ## Dynamic GS
                if self.opt.splat and self.opt.rendering and self.opt.motion_splat and not self.opt.dynamic_splat:
                    motion_gaussians, motion_gs_confs = self.motion_splat_head(
                        aggregated_tokens_list,
                        gaussians,
                        patch_start_idx,
                        output_timestep_embeds[:, i],  # (B, D)
                        frames_chunk_size=frames_chunk_size,
                        only_frame0=only_frame0,
                    )
                    motion_gaussians = rearrange(motion_gaussians, "b f h w c -> b f c h w")
                    motion_colors, motion_scales, motion_rotations, motion_opacities = torch.split(motion_gaussians, [(self.opt.sh_degree+1)**2 * 3, 3, 4, 1], dim=2)
                    pred_motion_splat_list.append({
                        "motion_color": motion_colors,  # (B, F_in, 3, H, W)
                        "motion_scale": motion_scales,  # (B, F_in, 3, H, W)
                        "motion_rotation": motion_rotations,  # (B, F_in, 4, H, W)
                        "motion_opacity": motion_opacities,  # (B, F_in, 1, H, W)
                        "motion_conf": motion_gs_confs.unsqueeze(2),  # (B, F_in, 1, H, W)
                    })
            pred_motions = torch.stack(pred_motion_list, dim=1)  # (B, F_out, F_in, 3+1, H, W)
            pred_motion_splat = None if len(pred_motion_splat_list) == 0 \
                else pred_motion_splat_list  # a list of `F_out` dict of (B, F_in, C, H, W)
        else:
            pred_motions, pred_motion_splat = None, None

        # (Optional) Post-processing motion 3DGS attributes
        if pred_motions is not None and pred_motion_splat is not None:
            for i in range(output_timestep_embeds.shape[1]):
                pred_motion_norms = torch.norm(pred_motions[:, i, :, :3, ...], dim=2, keepdim=True)  # (B, F_in, 1, H, W)
                mask = (pred_motion_norms >= self.opt.mask_by_motion).float()  # (B, F_in, 1, H, W)
                mask_opacity = (pred_motion_norms >= self.opt.mask_by_motion_opacity).float()  # (B, F_in, 1, H, W)
                for k in ["color", "opacity", "rotation", "scale", "conf"]:
                    if k == "opacity":
                        pred_motion_splat[i][f"motion_{k}"] = mask_opacity * pred_motion_splat[i][f"motion_{k}"] + (1. - mask_opacity) * return_dict[k]
                    else:
                        pred_motion_splat[i][f"motion_{k}"] = mask * pred_motion_splat[i][f"motion_{k}"] + (1. - mask) * return_dict[k]

        return return_dict, pred_motions, pred_motion_splat
