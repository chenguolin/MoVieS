from typing import *
from torch import Tensor

from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from extensions.vggt.vggt.heads.head_act import activate_head


class LinearHead(nn.Module):
    def __init__(self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 3,
        activation: str = "linear",
        conf_activation: str = "expp1",
        features: int = 256,
        shortcut_dim: int = -1,
        time_dim: int = 0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation

        # Refer to VGGT DPTHead
        head_features_1 = features
        head_features_2 = 32
        conv2_in_channels = head_features_1 // 2

        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, conv2_in_channels * patch_size ** 2)
        self.out = nn.Sequential(
            nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # (Optional) Shortcut
        if shortcut_dim > 0:
            self.shortcut = nn.Sequential(
                nn.Conv2d(shortcut_dim, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

        # (Optional) Time conditioning
        if time_dim > 0:
            self.time_embed = nn.Linear(time_dim, 2 * dim_in)
            # ## Zero initialize the time embedding weights for AdaLN
            # self.time_embed.weight.data.zero_()
            # self.time_embed.bias.data.zero_()

    def forward(self,
        aggregated_tokens_list: Union[Tensor, List[Tensor]],
        shortcut_info: Tensor,
        patch_start_idx: int,
        timestep_embeds: Optional[Tensor] = None,
        frames_chunk_size: int = 0,  # to compatible with compatible `DPTHead`
        only_frame0: bool = False,
    ) -> Tensor:
        if only_frame0:
            aggregated_tokens_list = [tokens[:, 0:1, :] for tokens in aggregated_tokens_list]
            shortcut_info = shortcut_info[:, 0:1, :, :, :]

        B, S, _, H, W = shortcut_info.shape

        if hasattr(self, "time_embed"):
            time_scale, time_bias = self.time_embed(timestep_embeds)[:, None, None, :].chunk(2, dim=-1)  # (B, 1, 1, D)
            time_scale, time_bias = time_scale.repeat(1, S, 1, 1), time_bias.repeat(1, S, 1, 1)  # (B, S, 1, D)
            time_scale, time_bias = time_scale.reshape(B * S, 1, -1), time_bias.reshape(B * S, 1, -1)  # (B*S, 1, D)
        else:
            time_scale, time_bias = 0., 0.

        if not isinstance(aggregated_tokens_list, Tensor):
            x = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        else:
            x = aggregated_tokens_list[:, :, patch_start_idx:]

        x = x.reshape(B * S, -1, x.shape[-1])  # x.view(B * S, -1, x.shape[-1]); for compatibility

        x = self.norm(x)
        x = x * (1. + time_scale) + time_bias  # conduct AdaLN

        y: Tensor = self.proj(x)  # (B*S, N, D)
        y = y.transpose(-2, -1).view(B*S, -1, H // self.patch_size, W // self.patch_size)
        y = tF.pixel_shuffle(y, self.patch_size)  # (B*S, C, H, W)

        out: Tensor = self.out(y)  # (B*S, C, H, W)

        # (Optional) Shortcut
        if hasattr(self, "shortcut"):
            shortcut_out = self.shortcut(shortcut_info.reshape(B * S, -1, H, W))
            out = out + shortcut_out

        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # To compatible with `DPTHead` outputs
        preds = rearrange(preds, "(b s) h w c -> b s h w c", b=B, s=S)
        conf = rearrange(conf, "(b s) h w -> b s h w", b=B, s=S)

        return preds, conf  # `None`: to compatible with compatible `DPTHead`
