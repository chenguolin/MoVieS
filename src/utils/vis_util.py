from typing import *
from PIL.Image import Image as PILImage
from numpy import ndarray
from torch import Tensor
from wandb import Image as WandbImage

from PIL import Image
import numpy as np
import matplotlib
import torch
from torchvision.utils import flow_to_image
from einops import rearrange
import wandb
from plyfile import PlyData, PlyElement


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def overlay_mask_on_video(videos: Tensor, masks: Tensor, color: Tuple[float, float, float] = (250/255, 128/255, 114/255), alpha: float = 0.5):
    """Overlay a batch of masks `(B, F, H, W)` on a batch of videos `(B, F, 3, H, W)`."""
    B, F, C, H, W = videos.shape
    assert C == 3 and masks.shape == (B, F, H, W)

    mask_expanded = masks.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (B, F, 3, H, W)

    color_tensor = torch.tensor(color, dtype=videos.dtype, device=videos.device).view(1, 1, 3, 1, 1)
    color_mask = mask_expanded * color_tensor  # masked color

    overlaid = videos * (1. - alpha * mask_expanded) + color_mask * alpha

    return overlaid.clamp(0., 1.)  # (B, F, 3, H, W)


def colorize_flow(flows: Tensor, batch_mode: bool = False):
    """Colorize a sequence of flow maps: `((B, )F, 2, H, W) -> ((B, )F, 3, H, W)`."""
    if batch_mode:
        return torch.stack([
            colorize_flow(flow, batch_mode=False)
            for flow in flows
        ])

    assert flows.ndim == 4  # (F, 2, H, W)
    return flow_to_image(flows)


def save_xyz_rgb_as_ply(filename: str, xyz: Tensor, rgb: Optional[Tensor] = None):
    """Save an (F, 3, H, W) or (N, 3) XYZ tensor and corresponding RGB tensor as a PLY point cloud file."""
    if rgb is None:
        rgb = torch.ones_like(xyz)

    assert rgb.shape == xyz.shape

    if xyz.ndim == 4:
        assert xyz.shape[1] == 3
        xyz = rearrange(xyz, "f c h w -> (f h w) c")
        rgb = rearrange(rgb, "f c h w -> (f h w) c")

    assert xyz.ndim == 2 and xyz.shape[-1] == 3
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()

    points = []
    for (x, y, z), (r, g, b) in zip(xyz, rgb):
        points.append((x, y, z, (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)))

    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ('red', "u1"), ('green', "u1"), ('blue', "u1")])
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=True)
    ply.write(filename)


def normalize_among_last_dims(
        tensors: Tensor,
        num_dims: int,
        normalize_type: Literal[
            "min_max",
        ] = "min_max",
    ) -> Tensor:
    """Normalize (range in `[0, 1]`) among the last `num_dims` dimensions of a tensor and keep the original shape.

    For example, `(..., C, F, H, W) -> (..., C, F*H*W)`, then normalize on `C`, then reshape back to `(..., C, F, H, W)`.
    """
    last_shape = tensors.shape[-num_dims:]
    tensors = tensors.flatten(start_dim=-num_dims)  # e.g., (..., C, F, H, W) -> (..., C, F*H*W)

    if normalize_type == "min_max":
        tensors = (tensors - tensors.min(dim=-1, keepdim=True)[0]) / \
            (tensors.max(dim=-1, keepdim=True)[0] - tensors.min(dim=-1, keepdim=True)[0] + 1e-6)  # [0, 1]
    else:
        raise ValueError(f"Invalid normalize_type: [{normalize_type}]")

    tensors = tensors.reshape(list(tensors.shape[:-1]) + list(last_shape))  # e.g., (..., C, F*H*W) -> (..., C, F, H, W)
    return tensors


def colorize_depth(
        depths: Tensor,
        cmap: str = "inferno",
        normalize_num_dims: int = 3,
        normalize_type: Literal[
            "min_max",
        ] = "min_max",
        batch_mode: bool = False,
    ) -> Tensor:
    """Colorize a sequence of depth maps: `((B, )F(, 1), H, W) -> ((B, )F, 3, H, W)`."""
    if batch_mode:
        return torch.stack([
            colorize_depth(depth, cmap, normalize_num_dims, normalize_type, batch_mode=False)
            for depth in depths
        ])

    if depths.ndim == 4:  # (F, 1, H, W)
        depths = depths.squeeze(1)  # (F, H, W)
    assert depths.ndim == 3  # (F, H, W)

    depths = depths.float().nan_to_num()

    colormap = torch.tensor(matplotlib.colormaps[cmap].colors, dtype=depths.dtype, device=depths.device)
    depths = (normalize_among_last_dims(depths.unsqueeze(0), normalize_num_dims, normalize_type).squeeze(0) * 255.).long()  # (F, H, W)
    depths = colormap[depths].permute(0, 3, 1, 2)  # (F, H, W, 3) -> (F, 3, H, W)

    return depths


def wandb_video_log(outputs: Dict[str, Tensor], max_num: int = 4, max_frame: int = 256, fps: int = 4, format: str = "mp4") -> List[WandbImage]:
    """Organize videos in Dict `outputs` for wandb logging.

    Only process values in Dict `outputs` that have keys containing the word "images",
    which should be in the shape of (B, F, 3, H, W).
    """
    formatted_images = []
    for k in outputs.keys():
        if "images" in k and outputs[k] is not None:  # (B, F, 3, H, W)
            assert outputs[k].ndim == 5
            num, frame = outputs[k].shape[:2]
            num, frame = min(num, max_num), min(frame, max_frame)
            videos = rearrange(outputs[k][:num, :frame], "b f c h w -> f c h (b w)")
            formatted_images.append(
                wandb.Video(
                    tensor_to_video(videos.detach()).transpose(0, 3, 1, 2),  # (F, C, H, W) for wandb Video
                    caption=k,
                    fps=fps,
                    format=format,
                )
            )

    return formatted_images


def tensor_to_video(tensor: Tensor, return_pil: bool = False) -> Union[ndarray, List[PILImage]]:
    """Convert a tensor `((B, )F, C, H, W)` in `[0, 1]` to a numpy array or PIL Image `(F, H, (B *)W, C)` in `[0, 255]`."""
    if tensor.ndim == 5:  # (B, F, C, H, W)
        tensor = rearrange(tensor, "b f c h w -> f c h (b w)")
    assert tensor.ndim == 4  # (F, C, H, W)

    assert tensor.shape[1] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    video = (tensor.permute(0, 2, 3, 1).cpu().float().numpy() * 255.).astype(np.uint8)  # (F, H, W, C)
    if return_pil:
        video = [Image.fromarray(frame) for frame in video]
    return video


def wandb_mvimage_log(outputs: Dict[str, Tensor], max_num: int = 4, max_view: int = 8) -> List[WandbImage]:
    """Organize multi-view images in Dict `outputs` for wandb logging.

    Only process values in Dict `outputs` that have keys containing the word "images",
    which should be in the shape of (B, V, 3, H, W).
    """
    formatted_images = []
    for k in outputs.keys():
        if "images" in k and outputs[k] is not None:  # (B, V, 3, H, W)
            assert outputs[k].ndim == 5
            num, view = outputs[k].shape[:2]
            num, view = min(num, max_num), min(view, max_view)
            mvimages = rearrange(outputs[k][:num, :view], "b v c h w -> c (b h) (v w)")
            formatted_images.append(
                wandb.Image(
                    tensor_to_image(mvimages.detach()),
                    caption=k
                )
            )

    return formatted_images


def tensor_to_image(tensor: Tensor, return_pil: bool = False) -> Union[ndarray, PILImage]:
    """Convert a tensor `((B, )C, H, W)` in `[0, 1]` to a numpy array or PIL Image `(H, (B *)W, C)` in `[0, 255]`."""
    if tensor.ndim == 4:  # (B, C, H, W)
        tensor = rearrange(tensor, "b c h w -> c h (b w)")
    assert tensor.ndim == 3  # (C, H, W)

    assert tensor.shape[0] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    image = (tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)  # (H, W, C)
    if return_pil:
        image = Image.fromarray(image)
    return image


def load_image(image_path: str, rgba: bool = False, imagenet_norm: bool = False) -> Tensor:
    """Load an RGB(A) image from `image_path` as a tensor `(C, H, W)` in `[0, 1]`."""
    image = Image.open(image_path)
    tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.  # (C, H, W) in [0, 1]

    if not rgba and tensor_image.shape[0] == 4:
        mask = tensor_image[3:4]
        tensor_image = tensor_image[:3] * mask + (1. - mask)  # white background

    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        tensor_image = (tensor_image - mean) / std

    return tensor_image  # (C, H, W)
