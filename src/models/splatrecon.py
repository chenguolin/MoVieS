from typing import *
from torch import Tensor
from src.utils import StepTracker

import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange
from pytorch_msssim import ssim as SSIM
from lpips import LPIPS

from .losses import DepthLoss, MotionLoss

from src.options import Options
from src.models.networks import VGGSplaT
from src.models.gs_render import GaussianRenderer
from src.utils import convert_to_buffer, colorize_depth, normalize_among_last_dims


class SplatRecon(nn.Module):
    """A wrapper model for 3DGS feed-forward reconstruction."""
    def __init__(self, opt: Options, step_tracker: Optional[StepTracker] = None):
        super().__init__()

        self.opt = opt
        self.step_tracker = step_tracker

        # Reconstruction model
        self.backbone = VGGSplaT(opt)

        # Rendering
        self.gs_renderer = GaussianRenderer(self.opt)

        # Pretrained models as module buffers
            ## LPIPS loss
        self.lpips_loss = LPIPS(net="vgg")
        convert_to_buffer(self.lpips_loss, persistent=False)  # no gradient & not save to checkpoint
            ## Depth and motion loss
        self.depth_loss, self.motion_loss = DepthLoss(self.opt), MotionLoss(self.opt)
        convert_to_buffer(self.depth_loss, persistent=False)
        convert_to_buffer(self.motion_loss, persistent=False)

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Tensor], dtype: torch.dtype = torch.float32):
        outputs = {}

        images = data["image"].to(dtype)  # (B, F, 3, H, W)
        C2W = data["C2W"].to(dtype)  # (B, F, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)
        timesteps = data["timestep"].to(dtype)  # (B, F)
        depths = data["depth"].to(dtype)  # (B, F, H, W)
        depth_masks = data["depth_mask"].to(dtype)  # (B, F, H, W)

        H, W = images.shape[-2:]

        # Network inputs
        F_in = images.shape[1] - self.opt.num_output_frames  # self.opt.num_input_frames  # `F_in` may be random if `opt.random_video_size` is True
        input_images = images[:, :F_in, ...]  # (B, F_in, 3, H, W)
        input_C2W = C2W[:, :F_in, ...]  # (B, F_in, 4, 4)
        input_fxfycxcy = fxfycxcy[:, :F_in, ...]  # (B, F_in, 4)
        input_timesteps = timesteps[:, :F_in]  # (B, F_in)
        input_depths = depths[:, :F_in,...]  # (B, F_in, H, W)
        input_depth_masks = depth_masks[:, :F_in,...]  # (B, F_in, H, W)

        # Ground-truth supervision
        F_out = self.opt.num_output_frames
        if F_out > 0:
            output_images = images[:, F_in:,...]  # (B, F_out, 3, H, W)
            output_C2W = C2W[:, F_in:,...]  # (B, F_out, 4, 4)
            output_fxfycxcy = fxfycxcy[:, F_in:,...]  # (B, F_out, 4)
            output_timesteps = timesteps[:, F_in:]  # (B, F_out)
            output_depths = depths[:, F_in:,...]  # (B, F_out, H, W)
            output_depth_masks = depth_masks[:, F_in:,...]  # (B, F_out, H, W)
        else:  # the same as inputs
            F_out = F_in
            output_images = input_images  # (B, F_out=F_in, 3, H, W)
            output_C2W = input_C2W  # (B, F_out=F_in, 4, 4)
            output_fxfycxcy = input_fxfycxcy  # (B, F_out=F_in, 4)
            output_timesteps = input_timesteps  # (B, F_out=F_in)
            output_depths = input_depths  # (B, F_out=F_in, H, W)
            output_depth_masks = input_depth_masks  # (B, F_out=F_in, H, W)

        # Transform images to per-pixel 3DGS features
        model_outputs, pred_motions, pred_motion_splat = \
            self.backbone(
                input_images,
                input_C2W,
                input_fxfycxcy,
                input_timesteps,
                output_timesteps,
                self.opt.frames_chunk_size,
            )

        # Rendering
        if self.opt.rendering:
            with torch.enable_grad() if self.opt.splat else torch.no_grad():
                ## Dynamic rendering:
                    ### 1. Dynamic motion
                        #### 1.1. `pred_motions`: (B, F_out, F_in, 3 (xyz) + 1 (conf), H, W)
                        #### 1.2. `pred_motion_splat`: a list of `F_out` dict of (B, F_in, C, H, W); only scale & motion
                    ### 2. Dynamic depth
                        #### 2.1. `dynamic_depth`: (B, F_out, F_in, 1 (depth) + 1 (conf), H, W)
                        #### 2.2. `dynamic_splat`: a list of `F_out` dict of (B, F_in, C, H, W); all 3DGS attributes
                if pred_motions is not None or self.opt.dynamic_depth:
                    render_outputs: Dict[str, Tensor] = {}
                    render_outputs_list: List[Dict[str, Tensor]] = []
                    for i in range(F_out):  # for different output timesteps
                        if pred_motions is not None:
                            model_outputs["offset"] = pred_motions[:, i, :, :3, ...]  # (B, F_in, 3, H, W)
                        if pred_motion_splat is not None:
                            model_outputs.update(pred_motion_splat[i])  # a dict of (B, F_in, C, H, W)
                        if self.opt.dynamic_splat:
                            assert "dynamic_splat" in model_outputs
                            for k in model_outputs["dynamic_splat"][i].keys():
                                model_outputs[k] = model_outputs["dynamic_splat"][i][k]  # reset all gs attributes each out timestep
                        if self.opt.dynamic_depth:
                            assert "dynamic_depth" in model_outputs and model_outputs["dynamic_depth"].ndim == 6
                            model_outputs["depth"] = model_outputs["dynamic_depth"][:, i, :, :1, ...]  # (B, F_in, 1, H, W)
                        _render_outputs = self.gs_renderer.render(
                            model_outputs,  # a dict of (B, F_in, C, H, W)
                            input_C2W,  # (B, F_in, 4, 4)
                            input_fxfycxcy,  # (B, F_in, 4)
                            output_C2W[:, i:i+1, ...],  # (B, 1, 4, 4): one view corresponding to one timestep
                            output_fxfycxcy[:, i:i+1, ...],  # (B, 1, 4)
                        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
                        render_outputs_list.append(_render_outputs)
                    for k in ["gaussian_usage", "voxel_ratio"]:
                        if k in render_outputs_list[0]:
                            render_outputs[k] = render_outputs_list[0][k]  # (B,)
                    for k in render_outputs_list[0].keys():
                        if k not in ["gaussian_usage", "voxel_ratio"]:
                            render_outputs[k] = torch.cat([render_outputs_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
                ## Static rendering
                else:
                    render_outputs = self.gs_renderer.render(
                        model_outputs,
                        input_C2W,
                        input_fxfycxcy,
                        output_C2W,
                        output_fxfycxcy,
                    )
                render_images = render_outputs["image"].to(dtype)  # (B, F_out, 3, H, W)
                # render_masks = render_outputs["alpha"]  # (B, F_out, 1, H, W)
                if "depth" in render_outputs:
                    render_depths = render_outputs["depth"].squeeze(2)  # (B, F_out, H, W)
                else:
                    render_depths = None

            # (Optional) Visualize dynamic rendering results
            if not self.training and (pred_motions is not None or self.opt.dynamic_depth):
                ## Rendering at time0
                with torch.no_grad():
                    render_outputs_time0: Dict[str, Tensor] = {}
                    render_outputs_time0_list: List[Dict[str, Tensor]] = []
                    for i in range(F_out):  # for different output timesteps
                        if pred_motions is not None:
                            model_outputs["offset"] = pred_motions[:, 0, :, :3, ...]  # (B, F_in, 3, H, W)
                        if pred_motion_splat is not None:
                            model_outputs.update(pred_motion_splat[0])  # a dict of (B, F_in, C, H, W)
                        if self.opt.dynamic_splat:
                            assert "dynamic_splat" in model_outputs
                            for k in model_outputs["dynamic_splat"][0].keys():
                                model_outputs[k] = model_outputs["dynamic_splat"][0][k]  # reset all gs attributes each out timestep
                        if self.opt.dynamic_depth:
                            assert "dynamic_depth" in model_outputs and model_outputs["dynamic_depth"].ndim == 6
                            model_outputs["depth"] = model_outputs["dynamic_depth"][:, 0, :, :1, ...]  # (B, F_in, 1, H, W)
                        _render_outputs = self.gs_renderer.render(
                            model_outputs,  # a dict of (B, F_in, C, H, W)
                            input_C2W,  # (B, F_in, 4, 4)
                            input_fxfycxcy,  # (B, F_in, 4)
                            output_C2W[:, i:i+1, ...],  # (B, 1, 4, 4): one view corresponding to one timestep
                            output_fxfycxcy[:, i:i+1, ...],  # (B, 1, 4)
                        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
                        render_outputs_time0_list.append(_render_outputs)
                    for k in render_outputs_time0_list[0].keys():
                        if k not in ["gaussian_usage", "voxel_ratio"]:
                            render_outputs_time0[k] = torch.cat([render_outputs_time0_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
                    outputs["images_render_time0"] = render_outputs_time0["image"]  # (B, F_out, 3, H, W)
                ## Rendering at camera0
                with torch.no_grad():
                    render_outputs_camera0: Dict[str, Tensor] = {}
                    render_outputs_camera0_list: List[Dict[str, Tensor]] = []
                    for i in range(F_out):  # for different output timesteps
                        if pred_motions is not None:
                            model_outputs["offset"] = pred_motions[:, i, :, :3, ...]  # (B, F_in, 3, H, W)
                        if pred_motion_splat is not None:
                            model_outputs.update(pred_motion_splat[i])  # a dict of (B, F_in, C, H, W)
                        if self.opt.dynamic_splat:
                            assert "dynamic_splat" in model_outputs
                            for k in model_outputs["dynamic_splat"][i].keys():
                                model_outputs[k] = model_outputs["dynamic_splat"][i][k]  # reset all gs attributes each out timestep
                        if self.opt.dynamic_depth:
                            assert "dynamic_depth" in model_outputs and model_outputs["dynamic_depth"].ndim == 6
                            model_outputs["depth"] = model_outputs["dynamic_depth"][:, i, :, :1, ...]  # (B, F_in, 1, H, W)
                        _render_outputs = self.gs_renderer.render(
                            model_outputs,  # a dict of (B, F_in, C, H, W)
                            input_C2W,  # (B, F_in, 4, 4)
                            input_fxfycxcy,  # (B, F_in, 4)
                            output_C2W[:, 0:1, ...],  # (B, 1, 4, 4): one view corresponding to one timestep
                            output_fxfycxcy[:, 0:1, ...],  # (B, 1, 4)
                        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
                        render_outputs_camera0_list.append(_render_outputs)
                    for k in render_outputs_camera0_list[0].keys():
                        if k not in ["gaussian_usage", "voxel_ratio"]:
                            render_outputs_camera0[k] = torch.cat([render_outputs_camera0_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
                    outputs["images_render_camera0"] = render_outputs_camera0["image"]  # (B, F_out, 3, H, W)

            for k in ["gaussian_usage", "voxel_ratio"]:
                if k in render_outputs:
                    outputs[k] = render_outputs[k]  # (B,)

        # Geometry outputs for visualization and/or supervision
        if not self.opt.dynamic_depth:
            pred_depths = self.gs_renderer.depth_activation(model_outputs["depth"].squeeze(2))
            outputs["images_depth"] = colorize_depth(1. / pred_depths, batch_mode=True)  # disparity for visualization
        else:
            pred_depths_time0 = self.gs_renderer.depth_activation(model_outputs["dynamic_depth"][:, 0, :, :1, ...].squeeze(2))  # (B, F_in, H, W)
            pred_depths_camera0 = self.gs_renderer.depth_activation(model_outputs["dynamic_depth"][:, :, 0, :1, ...].squeeze(2))  # (B, F_out, H, W)
            outputs["images_depth_time0"] = colorize_depth(1. / pred_depths_time0, batch_mode=True)  # disparity for visualization
            outputs["images_depth_camera0"] = colorize_depth(1. / pred_depths_camera0, batch_mode=True)  # disparity for visualization

        # Visualization
        if self.opt.rendering:
            if render_depths is not None:
                outputs["images_depth_render"] = colorize_depth(1. / render_depths, batch_mode=True)  # disparity for visualization
            outputs["images_render"] = render_images
        outputs["images_gt"] = output_images
        if pred_motions is not None:
            outputs["images_motion_time0"] = rearrange(normalize_among_last_dims(
                rearrange(pred_motions[:, 0, :, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")  # (B, F_in, 3, H, W)
            outputs["images_motion_camera0"] = rearrange(normalize_among_last_dims(
                rearrange(pred_motions[:, :, 0, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")  # (B, F_out, 3, H, W)

        ################################ Compute reconstruction losses/metrics ################################

        if self.opt.splat and self.opt.rendering:
            # MSE
            outputs["image_mse"] = image_mse = tF.mse_loss(render_images, output_images, reduction="none").mean(dim=(1, 2, 3, 4))  # (B,)
            loss = image_mse
            # LPIPS
            if self.opt.lpips_weight > 0.:
                lpips = self.lpips_loss(
                    rearrange(output_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    rearrange(render_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                )  # (B*F_out, 1, 1, 1)
                outputs["lpips"] = lpips = rearrange(lpips, "(b f) c h w -> b f c h w", f=F_out).mean(dim=(1, 2, 3, 4))  # (B,)
                loss += self.opt.lpips_weight * lpips
        else:  # no photometric loss
            loss = 0.

        # (Optional) Depth loss
        if self.opt.depth_weight > 0. and not self.opt.dynamic_depth:
            if "depth_conf" in model_outputs and self.opt.depth_conf_alpha != 0.:
                depth_confs = model_outputs["depth_conf"]
            else:
                depth_confs = torch.ones_like(input_depth_masks)  # (B, F_in, H, W)

            outputs["depth_loss"] = depth_loss = data["depth_weight"].to(dtype) * self.depth_loss(pred_depths, input_depths, input_depth_masks, depth_confs)  # (B,)
            loss += self.opt.depth_weight * depth_loss

            if self.opt.rendering and render_depths is not None:
                outputs["depth_render_loss"] = depth_render_loss = data["depth_weight"].to(dtype) * self.depth_loss(render_depths, output_depths, output_depth_masks)  # (B,)
                loss += self.opt.depth_weight * depth_render_loss

        # (Optional) Opacity loss
        if self.opt.rendering and self.opt.opacity_weight > 0.:
            if pred_motion_splat is not None:
                opacity = torch.stack([pred_motion_splat[i]["motion_opacity"] for i in range(F_out)], dim=1)  # (B, F_out, F_in, 1, H, W)
            elif self.opt.dynamic_splat:
                opacity = torch.stack([model_outputs["dynamic_splat"][i]["opacity"] for i in range(F_out)], dim=1)  # (B, F_out, F_in, 1, H, W)
            else:
                opacity = model_outputs["opacity"].unsqueeze(1)  # (B, 1, F_in, 1, H, W)
            outputs["opacity_loss"] = opacity_loss = torch.abs(self.gs_renderer.opacity_activation(opacity)).mean(dim=(1, 2, 3, 4, 5))  # (B,)
            loss += self.opt.opacity_weight * opacity_loss

        # (Optional) Motion loss
        if self.opt.motion_weight > 0. and pred_motions is not None:
            tracks_world = data["track_world"]  # a list of (F, N, 3)
            tracks_xy = data["track_xy"]  # a list of (F, N, 2)
            visibilities = data["visibility"]  # a list of (F, N)
            for ii in range(len(tracks_world)):
                tracks_world[ii] = tracks_world[ii].to(device=images.device, dtype=dtype)
                tracks_xy[ii] = tracks_xy[ii].to(device=images.device)
                visibilities[ii] = visibilities[ii].to(device=images.device)

            motion_loss_list, motion_reg_list = [], []
            for b in range(images.shape[0]):
                if tracks_world[b].shape[1] > 1:  # not dummy tracks
                    ## Get ground-truth motions
                    input_tracks_world, input_tracks_xy, input_vis = \
                        tracks_world[b][:F_in], tracks_xy[b][:F_in], visibilities[b][:F_in]  # (F_in, N, 3), (F_in, N, 2), (F_in, N)
                    if self.opt.num_output_frames > 0:
                        output_tracks_world = tracks_world[b][F_in:]  # (F_out, N, 2)
                    else:
                        output_tracks_world = tracks_world[b][:F_in]  # (F_in=F_out, N, 2)
                    # valid = output_vis.unsqueeze(1) & input_vis.unsqueeze(0)  # (F_out, F_in, N); more strict
                    valid = (~output_tracks_world.isnan().any(dim=-1)).unsqueeze(1) & input_vis.unsqueeze(0)  # (F_out, F_in, N); less strict
                    gt_motions = output_tracks_world.unsqueeze(1) - input_tracks_world.unsqueeze(0)  # (F_out, F_in, N, 3)
                    gt_motions = gt_motions[valid, :]  # (M, 3)
                    if gt_motions.shape[0] == 0:  # skip this sample if without valid points
                        motion_loss_list.append(torch.tensor(0., device=images.device))  # (,)
                        motion_reg_list.append(torch.tensor(0., device=images.device))  # (,)
                        continue

                    ## Gather valid motion predictions: (H, W) -> (M,)
                    linear_idxs = (input_tracks_xy[..., 1] * W + input_tracks_xy[..., 0]).unsqueeze(0).repeat(F_out, 1, 1)  # (F_out, F_in, N)
                    linear_idxs = linear_idxs.masked_fill(~valid, 0)  # not really supervise on these invalid points
                    pred_motions_3 = rearrange(pred_motions[b, :, :, :3, ...], "f_out f_in c h w -> f_out f_in (h w) c")  # (F_out, F_in, HW, 3)
                    pred_motions_3 = pred_motions_3.gather(dim=2, index=linear_idxs.unsqueeze(-1).repeat(1, 1, 1, 3))  # (F_out, F_in, N, 3)
                    pred_motions_3 = pred_motions_3[valid, :]  # (M, 3)

                    ## (Optional) Gather valid confidences: (H, W) -> (M,)
                    if pred_motions.shape[3] > 3 and self.opt.motion_conf_alpha != 0.:
                        motion_confs = pred_motions[b, :, :, 3, ...]  # (F_out, F_in, H, W)
                        motion_confs = rearrange(motion_confs, "f_out f_in h w -> f_out f_in (h w)")  # (F_out, F_in, HW)
                        motion_confs = motion_confs.gather(dim=2, index=linear_idxs)  # (F_out, F_in, N)
                        motion_confs = motion_confs[valid]  # (M,)
                    else:
                        motion_confs = torch.ones_like(pred_motions_3[..., 0])  # (M,)

                    motion_loss = self.motion_loss(pred_motions_3, gt_motions, motion_confs)  # (,)
                    motion_reg = pred_motions_3.norm(dim=1).mean()  # (,)

                else:  # dummy tracks: zero motions
                    motion_loss = pred_motions[b, :, :, :3, ...].norm(dim=2)  # (F_out, F_in, H, W)
                    if pred_motions.shape[3] > 3 and self.opt.motion_conf_alpha!= 0.:
                        motion_confs = pred_motions[b, :, :, 3, ...]  # (F_out, F_in, H, W)
                    else:
                        motion_confs = torch.ones_like(motion_loss)  # (F_out, F_in, H, W)
                    motion_loss = (motion_confs * motion_loss - self.opt.motion_conf_alpha * torch.log(motion_confs)).mean()  # (,)
                    motion_reg = torch.zeros_like(motion_loss)  # (,)

                motion_loss_list.append(motion_loss)  # (,)
                motion_reg_list.append(motion_reg)  # (,)

            outputs["motion_loss"] = motion_loss = data["motion_weight"].to(dtype) * torch.stack(motion_loss_list, dim=0)  # (B,)
            loss += (self.opt.motion_weight * motion_loss)

            if self.opt.motion_reg_weight > 0.:
                outputs["motion_reg"] = motion_reg = torch.stack(motion_reg_list, dim=0)  # (B,); motion regularization
                loss += (self.opt.motion_reg_weight * motion_reg)

        outputs["loss"] = loss  # (B,)

        # Metric: PSNR, SSIM and LPIPS
        if self.opt.rendering:
            with torch.no_grad():
                outputs["psnr"] = -10 * torch.log10(torch.mean((output_images - render_images) ** 2, dim=(1, 2, 3, 4)))  # (B,)
                outputs["ssim"] = SSIM(
                    rearrange(output_images, "b f c h w -> (b f) c h w"),
                    rearrange(render_images, "b f c h w -> (b f) c h w"),
                    data_range=1., size_average=False,
                )  # （B*F_out,)
                outputs["ssim"] = rearrange(outputs["ssim"], "(b f) -> b f", f=F_out).mean(dim=1)  # (B,)
                if "lpips" not in outputs:
                    lpips = self.lpips_loss(
                        rearrange(output_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                        rearrange(render_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    )  # (B*F_out, 1, 1, 1)
                    outputs["lpips"] = rearrange(lpips, "(b f) c h w -> b f c h w", f=F_out).mean(dim=(1, 2, 3, 4))  # (B,)
        else:
            outputs["psnr"] = torch.zeros_like(outputs["loss"])
            outputs["ssim"] = torch.zeros_like(outputs["loss"])
            outputs["lpips"] = torch.zeros_like(outputs["loss"])

        return outputs
