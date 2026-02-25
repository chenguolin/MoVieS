from typing import *

import os
from dataclasses import dataclass


@dataclass
class Options:
    # Data
    input_res: Tuple[int, int] = (224, 224)
    size_divisor: int = 14
    num_input_frames: int = 5
    num_output_frames: int = 8
    min_valid_ratio: float = 0.01  # minimum valid point ratio for filtering invalid samples
    max_rotation_diff: float = 45.  # maximum rotation difference for filtering invalid samples
    crop_resize_ratio: Tuple[float, float] = (0.77, 1.)  # cf. AnySplat
    norm_xyz: bool = True  # cf. VGGT
        ## Dynamic (random) sampling
    random_video_size: bool = True
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.)  # (0.33, 1.)  # cf. VGGT
    min_num_input_frames: int = 2  # cf. VGGT
        ## Camera normalization
    camera_norm_type: Literal[
        "none",
        "canonical",
    ] = "canonical"
    camera_norm_unit: float = 1.
        ## View sampling
    view_sampling_type: Literal[
        "uniform",
        "arbitrary_output_include_input",
    ] = "arbitrary_output_include_input"
    include_input_prob: float = 0.25  # only for `arbitrary_output_include_input`
        ## Dataset selection
    dataset_name: Literal[
        "re10k",
        "tartanair",
        "matrixcity",
        "dynamicreplica",
        "pointodyssey",
        "vkitti2",
        "spring",
        "stereo4d",
        "mix_static",
        "mix_all",
    ] = "mix_static"
        ## Post initialization (`__post_init__`)
    root: str = "./resources"
    all_file_dirs_train: Dict[str, str] = None
    all_file_dirs_test: Dict[str, str] = None
    dataset_args: Dict[str, Dict[str, Any]] = None

    # Backbone
    vggt_init: bool = True
    freeze_dino_encoder: bool = True
    freeze_attn_backbone: bool = False
    use_dpt_splat_head: bool = True  # for static-3dgs head
    use_dpt_motion_splat_head: bool = True  # for motion-3dgs head
    memory_efficient_attention: bool = False
    input_plucker: bool = True
    input_pose_token: bool = True
    input_timestep: bool = False
    time_dim: int = 20  # sin-cos
    output_motion: bool = False
    motion_splat: bool = True  # time-conditioning all 3DGS attributes, distinguish static and dynamic parts via motion values
    mask_by_motion: float = 0.005  # mask by motion
    mask_by_motion_opacity: float = 0.005  # mask by motion for opacity
    dynamic_splat: bool = False  # time-conditioning all 3DGS attributes, not distinguish static and dynamic parts, cf. BTimer
    dynamic_depth: bool = False  # time-conditioning depth, cf. BTimer
    frames_chunk_size: int = 8  # for DPTHead

    # Splatting
    splat: bool = True
    rendering: bool = True
    voxel_size: float = 0.
        ## Color
    color_act_type: Literal[
        "identity",
        "tanh",
    ] = "tanh"
        ## Scale
    scale_act_type: Literal[
        "identity",
        "range",  # cf. GRM
        "exp",  # cf. GS-LRM
        "softplus",  # cf. NoPoSplat
    ] = "softplus"
            ### Scale: range
    scale_act_min: float = 0.
    scale_act_max: float = 0.3
            ### Scale: exp
    scale_act_bias: float = -2.3  # `-2.3` cf. GS-LRM Appendix A.4
            ### Scale: softplus
    scale_act_scale: float = 0.001  # `0.001` cf. NoPoSplat
        ## Depth
    depth_act_type: Literal[
        "identity",
        "range",  # cf. GS-LRM
        "norm_exp",  # cf. DUSt3R
        "exp",  # cf. VGGT
    ] = "identity"
    znear: float = 0.001
    zfar: float = 100
        ## XYZ
    xyz_act_type: Literal[
        "identity",
        "norm_exp",  # cf. DUSt3R
        "inv_log",  # cf. VGGT
    ] = "identity"
        ## Opacity
    opacity_act_bias: float = -1.  # `-1.` for tanh (i.e., `-2.` for sigmoid) cf. GS-LRM Appendix A.4
        ## Advanced settings
    sh_degree: int = 0
    gsplat_radius_clip: float = 0.
    gsplat_render_mode: Literal[
        "RGB",
        "RGB+D",
        "RGB+ED",
    ] = "RGB"
    gsplat_rasterize_mode: Literal[
        "classic",
        "antialiased",
    ] = "classic"
    gsplat_distributed: bool = False

    # Training
        ## LPIPS
    lpips_weight: float = 0.5
        ## Depth loss
    depth_weight: float = 1.
    depth_grad_weight: float = 1.  # cf. VGGT
    depth_grad_loss_scale: int = 4
            ### Confidence-aware
    depth_conf_alpha: float = 0.  # cf. DUSt3R
        ## Motion loss
    motion_weight: float = 10.
    motion_dist_weight: float = 10.
    motion_dist_sample_number: int = 1000
            ### Confidence-aware
    motion_conf_alpha: float = 0.  # cf. DUSt3R
            ### regularization
    motion_reg_weight: float = 0.
        ## Opacity loss
    opacity_weight: float = 0.
    prune_ratio: float = 0.6
    random_ratio: float = 0.25
    opacity_threshold: float = 0.001
        ## LR scheduler
    name_lr_mult: Optional[str] = None
    exclude_name_lr_mult: Optional[str] = "splat_head,extra_embed"
    lr_mult: float = 0.1

    def __post_init__(self):
        # Data directory
        assert os.path.exists(self.root), f"Invalid data path: [{self.root}]"

        # Data directory arguments
        self.all_file_dirs_train = {
            "re10k": os.path.join(self.root, "RealEstate/720P/train"),
            "tartanair": os.path.join(self.root, "TartanAir/train"),
            "matrixcity": os.path.join(self.root, "MatrixCity/train"),
            "dynamicreplica": os.path.join(self.root, "DynamicReplica/train"),
            "pointodyssey": os.path.join(self.root, "PointOdyssey/train"),
            "vkitti2": os.path.join(self.root, "VKITTI2/train"),
            "spring": os.path.join(self.root, "Spring/train"),
            "stereo4d": os.path.join(self.root, "Stereo4D/train/processed"),
            "tapvid3d": os.path.join(self.root, "TAPVid3D"),
        }
        self.all_file_dirs_test = {
            "re10k": os.path.join(self.root, "RealEstate/720P/test"),
            "tartanair": os.path.join(self.root, "TartanAir/test"),
            "matrixcity": os.path.join(self.root, "MatrixCity/test"),
            "dynamicreplica": os.path.join(self.root, "DynamicReplica/test"),
            "pointodyssey": os.path.join(self.root, "PointOdyssey/test"),
            "vkitti2": os.path.join(self.root, "VKITTI2/test"),
            "spring": os.path.join(self.root, "Spring/test"),
            "stereo4d": os.path.join(self.root, "Stereo4D/test/processed"),
            "tapvid3d": os.path.join(self.root, "TAPVid3D"),
        }

        # Set up the number of gap frames
        self.num_gap_frames = 4 * (self.num_input_frames - 1) + 1  # TODO: make `4` configurable

        # Dataset-specific arguments
        self.default_dataset_args = {
            "min_bounded_gap": self.num_input_frames,
            "max_bounded_gap": self.num_gap_frames,
            "test_max_bounded_gap": None,  # different gap for the test dataset
            "min_depth_quantile": None,
            "max_depth_quantile": None,
            "is_static": False,
        }
        self.dataset_args = {
            "re10k": {
                "max_bounded_gap": self.num_gap_frames * 10,
                "min_depth_quantile": 0.2,  # not GT (predicted by VDA and scaled by Depth Pro)
                "max_depth_quantile": 0.8,  # not GT (predicted by VDA and scaled by Depth Pro)
                "is_static": True,
            },
            "tartanair": {
                "max_bounded_gap": self.num_gap_frames * 2,
                "is_static": True,
            },
            "matrixcity": {
                "max_bounded_gap": self.num_gap_frames,
                "test_max_bounded_gap": self.num_input_frames,  # train & test FPS are different
                "is_static": True,
            },
            "dynamicreplica": {
                "max_bounded_gap": self.num_gap_frames * 2,
            },
            "pointodyssey": {
                "max_bounded_gap": self.num_gap_frames * 2,
            },
            "vkitti2": {
                "max_bounded_gap": self.num_gap_frames,
            },
            "spring": {
                "max_bounded_gap": self.num_gap_frames * 5,
            },
            "stereo4d": {
                "max_bounded_gap": self.num_gap_frames * 5,
            },
        }


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}


# MoVieS
opt_dict["movies_static"] = Options()

opt_dict["movies"] = Options(
    dataset_name="mix_all",
    freeze_dino_encoder=False,
    memory_efficient_attention=True,
    input_timestep=True,
    output_motion=True,
    motion_weight=10.,
    exclude_name_lr_mult="splat_head,motion_splat_head,extra_embed,time_embed",  # ,motion_head
)
