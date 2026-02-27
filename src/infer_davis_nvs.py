import sys; sys.path.append("../extensions/vggt")

import os
import argparse
import numpy as np
import imageio.v2 as iio

import torch
from torch import Tensor
from safetensors.torch import load_file

import sys; sys.path.append(".")  # for src modules
from src.options import opt_dict
from src.models import SplatRecon
from src.utils import *


CKPT_PATH = "resources/movies_ckpt.safetensors"
DATA_DIR = "resources/DAVIS"


def concat_videos(video_paths: list[str], output_path: str) -> None:
    videos = [iio.mimread(video_path) for video_path in video_paths]
    videos = np.concatenate(videos, axis=0)  # (total_frames, H, W, 3)
    iio.mimwrite(output_path, videos, macro_block_size=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoVieS NVS inference")
    parser.add_argument("--name", type=str, required=True, help="DAVIS sample name")
    return parser.parse_args()


@torch.inference_mode()
def main(args: argparse.Namespace):
    os.makedirs(f"out/{args.name}", exist_ok=True)

    # NOTE: Load pretrained MoVieS model
    opt = opt_dict["movies"]
    model = SplatRecon(opt, load_lpips=False)  # lpips is not used for inference
    model.load_state_dict(load_file(CKPT_PATH), strict=True)
    model.eval()

    # NOTE: Load a preprocessed posed video for inference; camera is normalized similar to VGGT
    npz_path = f"{DATA_DIR}/{args.name}.npz"
    npz = np.load(npz_path)
    input_images = torch.from_numpy(npz["images"]).float().unsqueeze(0)  # (1, F_in, 3, H, W)
    input_C2W = torch.from_numpy(npz["C2W"]).float().unsqueeze(0)  # (1, F_in, 4, 4)
    input_fxfycxcy = torch.from_numpy(npz["fxfycxcy"]).float().unsqueeze(0)  # (1, F_in, 4); normalized intrinsics
    input_timesteps = torch.linspace(0, 1, steps=13).unsqueeze(0)  # (1, F_in)
    output_timesteps = torch.linspace(0, 1, steps=13).unsqueeze(0)  # (1, F_out)

    iio.mimwrite(f"out/{args.name}/input_video.mp4", tensor_to_video(input_images), macro_block_size=1)

    F_in, F_out = input_timesteps.shape[1], output_timesteps.shape[1]
    output_fxfycxcy = input_fxfycxcy[:, 0:1, :].repeat(1, F_out, 1)  # same intrinsics for all output frames

    device = "cuda"
    model = model.to(device)

    # NOTE: Inference with MoVieS
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # `backbone_outputs`: static attributes, including "depth", "color", "scale", "rotation", "opacity", etc.
        # `pred_motions`: (B, F_out, F_in, 3 (xyz) + 1 (conf), H, W)
        # `pred_motion_gs`: a list of `F_out` dict of (B, F_in, C, H, W)
        backbone_outputs, pred_motions, pred_motion_gs = \
            model.backbone(
                input_images.to(device=device, dtype=torch.bfloat16),
                input_C2W.to(device=device, dtype=torch.bfloat16),
                input_fxfycxcy.to(device=device, dtype=torch.bfloat16),
                input_timesteps.to(device=device, dtype=torch.bfloat16),
                output_timesteps.to(device=device, dtype=torch.bfloat16),
                frames_chunk_size=16,
            )

    iio.mimwrite(f"out/{args.name}/output_depth.mp4", tensor_to_video(
        colorize_depth(1./backbone_outputs["depth"], batch_mode=True)), macro_block_size=1)

    # NOTE: Render the output dynamic 3DGS with desired timestep and camera parameters

    # 1. Fix at the first camera, moving timesteps
    render_outputs: Dict[str, Tensor] = {}
    render_outputs_list: List[Dict[str, Tensor]] = []
    for i in range(F_out):  # for different output timesteps
        if pred_motions is not None:
            backbone_outputs["offset"] = pred_motions[:, i, :, :3, ...]  # (B, F_in, 3, H, W); moving timesteps here
        if pred_motion_gs is not None:
            backbone_outputs.update(pred_motion_gs[i])  # a dict of (B, F_in, C, H, W); moving timesteps here
        _render_outputs = model.gs_renderer.render(
            backbone_outputs,  # a dict of (B, F_in, C, H, W)
            input_C2W.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4, 4)
            input_fxfycxcy.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4)
            # NOTE: Target C2W and fxfycxcy for rendering; fixed to the first camera here
            input_C2W[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4, 4): one view corresponding to one timestep
            output_fxfycxcy[:, 0:1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4)
        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
        render_outputs_list.append(_render_outputs)

    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat([render_outputs_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
    render_images = render_outputs["image"]  # (B, F_out, 3, H, W)

    iio.mimwrite(f"out/{args.name}/output_render_camera0.mp4", tensor_to_video(render_images), macro_block_size=1)
    iio.mimwrite(f"out/{args.name}/output_motion_camera0.mp4", tensor_to_video(rearrange(normalize_among_last_dims(
        rearrange(pred_motions[:, 0, :, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")), macro_block_size=1)
    iio.mimwrite(f"out/{args.name}/output_motion_time0.mp4", tensor_to_video(rearrange(normalize_among_last_dims(
        rearrange(pred_motions[:, :, 0, :3, ...], "b f c h w -> b c f h w"), num_dims=3), "b c f h w -> b f c h w")), macro_block_size=1)

    # 2. Fix at the last timestep, moving cameras
    render_outputs: Dict[str, Tensor] = {}
    render_outputs_list: List[Dict[str, Tensor]] = []
    for i in range(F_out):  # for different output timesteps
        if pred_motions is not None:
            backbone_outputs["offset"] = pred_motions[:, -1, :, :3, ...]  # (B, F_in, 3, H, W); fixed to the last timestep here
        if pred_motion_gs is not None:
            backbone_outputs.update(pred_motion_gs[-1])  # a dict of (B, F_in, C, H, W); fixed to the last timestep here
        _render_outputs = model.gs_renderer.render(
            backbone_outputs,  # a dict of (B, F_in, C, H, W)
            input_C2W.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4, 4)
            input_fxfycxcy.to(device=device, dtype=torch.bfloat16),  # (B, F_in, 4)
            # NOTE: Target C2W and fxfycxcy for rendering; moving cameras here
            input_C2W[:, i:i+1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4, 4): one view corresponding to one timestep
            output_fxfycxcy[:, i:i+1, ...].to(device=device, dtype=torch.bfloat16),  # (B, 1, 4)
        )  # a dict of (B, 1, C, H, W): one view corresponding to one timestep
        render_outputs_list.append(_render_outputs)

    for k in render_outputs_list[0].keys():
        if k not in ["gaussian_usage", "voxel_ratio"]:
            render_outputs[k] = torch.cat([render_outputs_list[i][k] for i in range(F_out)], dim=1)  # (B, F_out, C, H, W)
    render_images = render_outputs["image"]  # (B, F_out, 3, H, W)

    iio.mimwrite(f"out/{args.name}/output_render_time-1.mp4", tensor_to_video(render_images), macro_block_size=1)
    concat_videos(
        [
            f"out/{args.name}/output_render_camera0.mp4",
            f"out/{args.name}/output_render_time-1.mp4",
        ],
        f"out/{args.name}/output_render.mp4",
    )


if __name__ == "__main__":
    main(parse_args())
