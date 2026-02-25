from typing import *
from torch import Tensor, BoolTensor

import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from src.options import Options
from src.utils import convert_to_buffer


class DepthLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.grad_loss = GradientLoss(scales=opt.depth_grad_loss_scale)
        convert_to_buffer(self.grad_loss, persistent=False)

    def forward(self,
        pred_depths: Tensor,
        gt_depths: Tensor,
        masks: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
    ):
        if masks is None:
            masks = torch.ones_like(pred_depths[:, :, :, :])  # (B, F, H, W)
        if confs is None:
            confs = torch.ones_like(pred_depths[:, :, :, :])  # (B, F, H, W)

        F, H, W = masks.shape[1:]

        pred_depths = rearrange(pred_depths, "b f h w -> b (f h w)")  # (B, N)
        gt_depths = rearrange(gt_depths, "b f h w -> b (f h w)")  # (B, N)
        masks = rearrange(masks, "b f h w -> b (f h w)")  # (B, N)
        confs = rearrange(confs, "b f h w -> b (f h w)")  # (B, N)

        # Confidence-weighted MSE
        depth_loss = tF.mse_loss(pred_depths, gt_depths, reduction="none")  # (B, N)
        depth_loss = (confs * depth_loss - self.opt.depth_conf_alpha * torch.log(confs))  # (B, N)
        depth_loss = (depth_loss * masks).sum(dim=-1) / (masks.sum(dim=1) + 1e-6)  # (B,)

        # (Optional) Gradient loss
        if self.opt.depth_grad_weight > 0.:
            pred_depths = rearrange(pred_depths, "b (f h w) -> b f h w", f=F, h=H, w=W)
            gt_depths = rearrange(gt_depths, "b (f h w) -> b f h w", f=F, h=H, w=W)
            masks = rearrange(masks, "b (f h w) -> b f h w", f=F, h=H, w=W)
            confs = rearrange(confs, "b (f h w) -> b f h w", f=F, h=H, w=W)
            depth_grad_loss = self.grad_loss(pred_depths, gt_depths, masks, confs)  # (B,)
            depth_loss = depth_loss + self.opt.depth_grad_weight * depth_grad_loss  # (B,)

        return depth_loss


class MotionLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

    def forward(self,
        pred_motions: Tensor,
        gt_motions: Tensor,
        confs: Optional[Tensor] = None,
    ):
        if confs is None:
            confs = torch.ones_like(pred_motions[:, 0])  # (M,)

        # Confidence-weighted L1
        motion_loss = tF.l1_loss(pred_motions, gt_motions, reduction="none")  # (M, 3)
        motion_loss = (confs.unsqueeze(-1) * motion_loss - self.opt.motion_conf_alpha * torch.log(confs.unsqueeze(-1)))  # (M, 3)
        motion_loss = motion_loss.mean()  # (,)

        # (Optional) Distribution loss
        if self.opt.motion_dist_weight > 0.:
            M = pred_motions.shape[0]
            idxs = torch.randperm(M)[:self.opt.motion_dist_sample_number] \
                if M > self.opt.motion_dist_sample_number else torch.arange(M)
            gt_motions = gt_motions[idxs]  # (M', 3)
            pred_motions = pred_motions[idxs]  # (M', 3)
            gt_motion_dist = gt_motions @ gt_motions.T  # (M, M)
            pred_motion_dist = pred_motions @ pred_motions.T  # (M, M)
            motion_dist_loss = tF.l1_loss(pred_motion_dist, gt_motion_dist, reduction="none")  # (M, M)
            motion_dist_loss = motion_dist_loss.mean()  # (,)
            motion_loss = motion_loss + self.opt.motion_dist_weight * motion_dist_loss  # (,)

        return motion_loss


################################ Gradient Loss ################################


# Copied from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/model_components/losses.py
# Modified:
    # (1) multi-view view: (B, H, W) -> (B, V, H, W)
    # (2) enable confidence-awareness; cf. DuSt3R, VGGT
    # (3) no `reduction_type`, so return batch-wise loss
    # (4) average loss over different scales


# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4):
        """
        Args:
            scales: number of scales to use
        """
        super().__init__()
        self.__scales = scales

    def forward(
        self,
        prediction: Tensor,             # (B, V, H, W)
        target: Tensor,                 # (B, V, H, W)
        mask: BoolTensor,               # (B, V, H, W)
        conf: Optional[Tensor] = None,  # (B, V, H, W)
    ) -> Tensor:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.__scales >= 1
        total = 0.

        for scale in range(self.__scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, :, ::step, ::step],
                target[:, :, ::step, ::step],
                mask[:, :, ::step, ::step],
                conf[:, :, ::step, ::step] if conf is not None else None,
            )
            total += grad_loss
        total /= self.__scales

        assert isinstance(total, Tensor)
        return total

    def gradient_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: BoolTensor,
        conf: Optional[Tensor] = None,
    ) -> Tensor:
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (-3, -2, -1))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
        if conf is not None:
            conf_x = torch.mean(conf[:, :, :, 1:] + conf[:, :, :, :-1])
            grad_x = torch.mul(conf_x, grad_x)
        mask_x = torch.mul(mask[:, :, :, 1:], mask[:, :, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
        if conf is not None:
            conf_y = torch.mean(conf[:, :, 1:, :] + conf[:, :, :-1, :])
            grad_y = torch.mul(conf_y, grad_y)
        mask_y = torch.mul(mask[:, :, 1:, :], mask[:, :, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = (torch.sum(grad_x, (-3, -2, -1)) + torch.sum(grad_y, (-3, -2, -1))) / 2.
        image_loss = image_loss / (summed_mask + 1e-6)

        return image_loss
