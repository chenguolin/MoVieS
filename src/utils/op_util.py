from typing import *
from torch import Tensor
from torch.nn import Module

from einops import rearrange


def zero_init_module(module: Module):
    for param in module.parameters():
        param.data.fill_(0.)  # set all parameters to 0


def convert_to_buffer(module: Module, persistent: bool = True):
    # Recurse over child modules
    for child in module.children():
        convert_to_buffer(child, persistent=persistent)

    # Also re-save buffers to change persistence
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def patchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    p1, p2 = patch_size
    if tokenize:
        return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p1, p2=p2)
    else:
        return rearrange(x, "b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=p1, p2=p2)


def unpatchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], input_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    (p1, p2), (h, w) = patch_size, input_size
    if tokenize:
        return rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p1, p2=p2)
    else:
        return rearrange(x, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=p1, p2=p2)


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has `target_dims` dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has [{x.ndim}] dims, but target_dims is [{target_dims}], which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]
