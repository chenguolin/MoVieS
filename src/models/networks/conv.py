from typing import *
from torch import Tensor

import math
import torch
from torch import nn
import torch.nn.functional as tF


def conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int], Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int], Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
):
    if dims == 2:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
        else:
            return nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
    else:
        raise ValueError(f"Invalid number of dimensions: [{dims}]")


class CausalConv3d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        assert len(kernel_size) == 3
        self.time_kernel_size = kernel_size[0]

        if isinstance(stride, int):
            dilation = (dilation, 1, 1)
        assert len(stride) == 3

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        assert len(padding) == 3

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: Tensor, causal: bool = True):
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat(1, 1, self.time_kernel_size - 1, 1, 1)
            x = torch.cat([first_frame_pad, x], dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat(1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            last_frame_pad = x[:, :, -1:, :, :].repeat(1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            x = torch.cat([first_frame_pad, x, last_frame_pad], dim=2)

        return self.conv(x)


class DualConv3d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.bias = bias

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        assert len(kernel_size) == 3

        if isinstance(stride, int):
            dilation = (dilation, 1, 1)
        assert len(stride) == 3

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        assert len(padding) == 3

        intermediate_channels = out_channels if in_channels < out_channels else in_channels

        # First spatial convolution
        self.weight1 = nn.Parameter(torch.Tensor(
            intermediate_channels,
            in_channels // groups,
            1,
            kernel_size[1],
            kernel_size[2],
        ))
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Second temporal convolution
        self.weight2 = nn.Parameter(torch.Tensor(
            out_channels,
            intermediate_channels // groups,
            kernel_size[0],
            1,
            1,
        ))
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialization
        self.reset_parameters()

    def forward(self, x: Tensor, skip_time_conv: bool = False):
        # First spatial convolution
        x = tF.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
        )

        if skip_time_conv:
            return x

        # Second temporal convolution
        x = tF.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
        )
        return x

    def reset_parameters(self):  # the same as PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)
        if self.bias2 is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias2, -bound, bound)


class FeatureEmbed(nn.Module):
    def __init__(self,
        type: str = "causal3d",
        input_channels: int = 3,
        out_channels: int = 512,
        t_ratio: int = 4,
        s_ratio: int = 8,
    ):
        super().__init__()

        if type == "causal3d":
            self.net = conv_nd(
                causal=True,
                dims=3,
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=(t_ratio, s_ratio, s_ratio),
                stride=(t_ratio, s_ratio, s_ratio),
            )
        elif type == "3d":
            if t_ratio != 1:
                self.net = conv_nd(
                    causal=False,
                    dims=3,
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=(t_ratio+1, s_ratio, s_ratio),  # `+1` for the extra first frame
                    stride=(t_ratio, s_ratio, s_ratio),
                    padding=(t_ratio//2, 0, 0),  # for the extra first frame
                )
            else:  # 2D VAE
                self.net = conv_nd(
                    causal=False,
                    dims=3,
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=(1, s_ratio, s_ratio),
                    stride=(1, s_ratio, s_ratio),
                )
        else:
            raise ValueError(f"Invalid feature embedder type: [{type}]")

    def forward(self, x: Tensor):
        return self.net(x)  # (B, D, f, hh, ww)
