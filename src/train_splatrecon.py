import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from typing import *
from torch import Tensor

import os
import argparse
import logging
import math
import gc
from contextlib import nullcontext

from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file as safetensors_load_file
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin

from src.options import opt_dict
from src.data import *  # import all dataset classes and `yield_forever`
from src.models import SplatRecon, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util


def main():
    PROJECT_NAME = "SplatRecon"

    parser = argparse.ArgumentParser(
        description="Train 3DGS reconstruction model for scenes"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default="wandb/token",
        help="Path to the WandB login token"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=5,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of the model pretrained in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained model checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    opt.__post_init__()

    # Create an experiment directory using the `tag`
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir}")

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set args by configs
    args.gradient_accumulation_steps = max(
        args.gradient_accumulation_steps,
        configs["train"].get("gradient_accumulation_steps", 1),
    )
    args.use_deepspeed = (
        args.use_deepspeed or
        configs["train"].get("use_deepspeed", False)
    )
    args.zero_stage = max(
        args.zero_stage,
        configs["train"].get("zero_stage", 2),
    )

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    if opt.random_video_size:
        accelerator.even_batches = False  # `False` if use `DynamicDataLoader`
        if deepspeed_plugin is not None:
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1  # manually set to 1 if use `DynamicBatchSampler`, not really used
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Allow the current step to be shared across processes in the model and dataloader
    step_tracker = util.StepTracker()

    if opt.dataset_name == "mix_static":
        train_dataset = (
            Re10kDataset(opt, training=True, step_tracker=step_tracker)
            + 100 * TartanairDataset(opt, training=True, step_tracker=step_tracker)
            + 10 * MatrixcityDataset(opt, training=True, step_tracker=step_tracker)
        )
    elif opt.dataset_name == "mix_all":
        train_dataset = (
            Stereo4dDataset(opt, training=True, step_tracker=step_tracker)
            + 100 * DynamicreplicaDataset(opt, training=True, step_tracker=step_tracker)
            + 1000 * PointodysseyDataset(opt, training=True, step_tracker=step_tracker)
            + 500 * Vkitti2Dataset(opt, training=True, step_tracker=step_tracker)
            + 2000 * SpringDataset(opt, training=True, step_tracker=step_tracker)
            # Static
            + Re10kDataset(opt, training=True, step_tracker=step_tracker)
            + 100 * TartanairDataset(opt, training=True, step_tracker=step_tracker)
            + 10 * MatrixcityDataset(opt, training=True, step_tracker=step_tracker)
        )
    elif opt.dataset_name == "re10k":  # 63134
        train_dataset = Re10kDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "tartanair":  # 686 (left & right)
        train_dataset = 100 * TartanairDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "matrixcity":  # 4266
        train_dataset = 10 * MatrixcityDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "dynamicreplica":  # 966 (left & right)
        train_dataset = 100 * DynamicreplicaDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "pointodyssey":  # 109 (131 filtered)
        train_dataset = 1000 * PointodysseyDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "vkitti2":  # 90
        train_dataset = 500 * Vkitti2Dataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "spring":  # 34
        train_dataset = 2000 * SpringDataset(opt, training=True, step_tracker=step_tracker)
    elif opt.dataset_name == "stereo4d":  # 98262 (98259 actually)
        train_dataset = Stereo4dDataset(opt, training=True, step_tracker=step_tracker)
    else:
        raise ValueError(f"Invalid dataset name [{opt.dataset_name}]")
    if opt.random_video_size:
        train_loader = DynamicDataLoader(
            opt,
            train_dataset,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory,
            drop_last=True,
            collate_fn=BaseDataset.collate_fn,
            persistent_workers=True,
            seed=args.seed,
            max_img_per_gpu=configs["train"]["batch_size_per_gpu"] * opt.num_input_frames,
        ).get_loader(epoch=0)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=configs["train"]["batch_size_per_gpu"],
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=True,
            drop_last=True,
            collate_fn=BaseDataset.collate_fn,
        )
    if opt.dataset_name == "mix_static":
        val_dataset = (
            Re10kDataset(opt, training=False, step_tracker=step_tracker)
            + 100 * TartanairDataset(opt, training=False, step_tracker=step_tracker)
            + 10 * MatrixcityDataset(opt, training=False, step_tracker=step_tracker)
        )
    elif opt.dataset_name == "mix_all":
        val_dataset = (
            Stereo4dDataset(opt, training=False, step_tracker=step_tracker)
            + 100 * DynamicreplicaDataset(opt, training=False, step_tracker=step_tracker)
            + 500 * PointodysseyDataset(opt, training=False, step_tracker=step_tracker)
            + 500 * Vkitti2Dataset(opt, training=False, step_tracker=step_tracker)
            + 1000 * SpringDataset(opt, training=False, step_tracker=step_tracker)
        )
    elif opt.dataset_name == "re10k":  # 7137
        val_dataset = Re10kDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "tartanair":  # 52 (left & right)
        val_dataset = 100 * TartanairDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "matrixcity":  # 321
        val_dataset = 10 * MatrixcityDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "dynamicreplica":  # 40 / 40 (val) (left & right)
        val_dataset = 100 * DynamicreplicaDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "pointodyssey":  # 13 / 15 (val)
        val_dataset = 500 * PointodysseyDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "vkitti2":  # 10
        val_dataset = 500 * Vkitti2Dataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "spring":  # 3
        val_dataset = 1000 * SpringDataset(opt, training=False, step_tracker=step_tracker)
    elif opt.dataset_name == "stereo4d":  # 9928
        val_dataset = Stereo4dDataset(opt, training=False, step_tracker=step_tracker)
    else:
         raise ValueError(f"Invalid dataset name [{opt.dataset_name}]")
    if opt.random_video_size:
        val_loader = DynamicDataLoader(
            opt,
            val_dataset,
            num_workers=args.num_workers,
            shuffle=True,  # shuffle for various visualization
            pin_memory=args.pin_memory,
            drop_last=False,
            collate_fn=BaseDataset.collate_fn,
            persistent_workers=True,
            seed=args.seed,
            max_img_per_gpu=configs["val"]["batch_size_per_gpu"] * opt.num_input_frames,
        ).get_loader(epoch=0)
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=configs["val"]["batch_size_per_gpu"],
            shuffle=True,  # shuffle for various visualization
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=True,
            drop_last=False,
            collate_fn=BaseDataset.collate_fn,
        )

    logger.info(f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # Initialize the model, optimizer and lr scheduler
    if accelerator.is_main_process:  # download pretrained module weights
        _ = SplatRecon(opt)
        del _
    model = SplatRecon(opt, step_tracker)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

    if opt.name_lr_mult is not None or opt.exclude_name_lr_mult is not None:
        name_params, name_params_lr_mult = {}, {}
        for name, param in model.named_parameters():
            # Include
            if opt.name_lr_mult is not None:
                assert opt.exclude_name_lr_mult is None
                for k in opt.name_lr_mult.split(","):
                    if k in name:
                        name_params_lr_mult[name] = param
            if opt.name_lr_mult is not None and name not in name_params_lr_mult:
                name_params[name] = param
            # Exclude
            if opt.exclude_name_lr_mult is not None:
                assert opt.name_lr_mult is None
                for k in opt.exclude_name_lr_mult.split(","):
                    if k in name:
                        name_params[name] = param
            if opt.exclude_name_lr_mult is not None and name not in name_params:
                name_params_lr_mult[name] = param
        optimizer = get_optimizer(
            params=[
                # Sorted names to ensure the same order for optimizer resuming
                {"params": list([name_params[name] for name in sorted(name_params.keys())]), "lr": configs["optimizer"]["lr"]},
                {"params": list([name_params_lr_mult[name] for name in sorted(name_params_lr_mult.keys())]), "lr": configs["optimizer"]["lr"] * opt.lr_mult}
            ],
            **configs["optimizer"]
        )
        if opt.exclude_name_lr_mult is not None:
            logger.info(f"Learning rate x [1.0] parameter names: {sorted(name_params.keys())}\n")
        else:
            logger.info(f"Learning rate x [{opt.lr_mult}] parameter names: {sorted(name_params_lr_mult.keys())}\n")
    else:
        optimizer = get_optimizer(params=params_to_optimize, **configs["optimizer"])

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # (Optional) Load a pretrained model
    if args.load_pretrained_model is not None:
        logger.info(f"Load pretrained checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        model, args.load_pretrained_model_ckpt = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_model, "checkpoints"),
            args.load_pretrained_model_ckpt,
            os.path.join(args.project_hdfs_dir, args.load_pretrained_model),
            model, accelerator, strict=False,
        )

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Download from HDFS
        if not os.path.exists(os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}')):
            # Load model before `accelerator.prepare()`
            model, args.resume_from_iter = util.load_ckpt(
                ckpt_dir,
                args.resume_from_iter,
                args.hdfs_dir,
                model, accelerator, strict=True,
            )
        # # Load everything
        # accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"), load_kwargs={"weights_only": False})

        global_update_step = int(args.resume_from_iter) + 1
    step_tracker.set_step(global_update_step)

    # Prepare everything with `accelerator`
    model, optimizer, lr_scheduler, train_loader, val_loader = \
        accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    # Cast input dataset to the appropriate dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    # assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # # (Optional) Load checkpoint
    # global_update_step = 0
    # if args.resume_from_iter is not None:
    #     logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
    #     # Download from HDFS
    #     if not os.path.exists(os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}')):
    #         args.resume_from_iter = util.load_ckpt(
    #             ckpt_dir,
    #             args.resume_from_iter,
    #             args.hdfs_dir,
    #             None,  # `None`: not load model ckpt here
    #             accelerator,  # manage the process states
    #         )
    #     # Load everything
    #     accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"), load_kwargs={"weights_only": False})
    #     global_update_step = int(args.resume_from_iter) + 1
    # step_tracker.set_step(global_update_step)
    # Load optimizer and lr_scheduler
    if args.resume_from_iter is not None:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "optimizer.bin"), weights_only=False))
        lr_scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "scheduler.bin"), weights_only=False))

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(model), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        with open(args.wandb_token_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    # Start training
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for epoch in range(configs["train"]["epochs"]):

        if opt.random_video_size:
            train_loader = DynamicDataLoader(
                opt,
                train_dataset,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=args.pin_memory,
                drop_last=True,
                collate_fn=BaseDataset.collate_fn,
                persistent_workers=True,
                seed=args.seed,
                max_img_per_gpu=configs["train"]["batch_size_per_gpu"] * opt.num_input_frames,
            ).get_loader(epoch=epoch)
            val_loader = DynamicDataLoader(
                opt,
                val_dataset,
                num_workers=args.num_workers,
                shuffle=True,  # shuffle for various visualization
                pin_memory=args.pin_memory,
                drop_last=False,
                collate_fn=BaseDataset.collate_fn,
                persistent_workers=True,
                seed=args.seed,
                max_img_per_gpu=configs["val"]["batch_size_per_gpu"] * opt.num_input_frames,
            ).get_loader(epoch=epoch)

        for batch in train_loader:

            if global_update_step == args.max_train_steps:
                progress_bar.close()
                logger.logger.propagate = True  # propagate to the root logger (console)
                if accelerator.is_main_process:
                    wandb.finish()
                logger.info("Training finished!\n")
                return

            model.train()

            for k in batch:
                if isinstance(batch[k], Tensor):
                    batch[k] = batch[k].to(device=accelerator.device, dtype=weight_dtype)

            with accelerator.accumulate(model):

                outputs = model(batch, dtype=weight_dtype)  # `step` starts from 1

                psnr = outputs["psnr"]
                ssim = outputs["ssim"]
                lpips = outputs["lpips"]
                loss = outputs["loss"]

                # Some extra outputs for logging
                depth_loss = outputs["depth_loss"] if "depth_loss" in outputs else None
                depth_render_loss = outputs["depth_render_loss"] if "depth_render_loss" in outputs else None
                opacity_loss = outputs["opacity_loss"] if "opacity_loss" in outputs else None
                motion_loss = outputs["motion_loss"] if "motion_loss" in outputs else None
                motion_reg = outputs["motion_reg"] if "motion_reg" in outputs else None
                gaussian_usage = outputs["gaussian_usage"] if "gaussian_usage" in outputs else None
                voxel_ratio = outputs["voxel_ratio"] if "voxel_ratio" in outputs else None

                # Backpropagate
                accelerator.backward(loss.mean())

                # Gradient clip
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Gather the losses across all processes for logging (if we use distributed training)
                psnr = accelerator.gather(psnr.detach()).mean()
                ssim = accelerator.gather(ssim.detach()).mean()
                lpips = accelerator.gather(lpips.detach()).mean()
                loss = accelerator.gather(loss.detach()).mean()

                if depth_loss is not None:
                    depth_loss = accelerator.gather(depth_loss.detach()).mean()
                if depth_render_loss is not None:
                    depth_render_loss = accelerator.gather(depth_render_loss.detach()).mean()
                if opacity_loss is not None:
                    opacity_loss = accelerator.gather(opacity_loss.detach()).mean()
                if motion_loss is not None:
                    motion_loss = accelerator.gather(motion_loss.detach()).mean()
                if motion_reg is not None:
                    motion_reg = accelerator.gather(motion_reg.detach()).mean()
                if gaussian_usage is not None:
                    gaussian_usage = accelerator.gather(gaussian_usage.detach()).mean()
                if voxel_ratio is not None:
                    voxel_ratio = accelerator.gather(voxel_ratio.detach()).mean()

                logs = {
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "lpips": lpips.item(),
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                logger.info(
                    f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                    f"psnr: {logs['psnr']:.4f}, ssim: {logs['ssim']:.4f}, lpips: {logs['lpips']:.4f}, " +
                    f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}"
                )

                # Log the training progress
                if (global_update_step % configs["train"]["log_freq"] == 0  # 1. every `log_freq` steps
                    or global_update_step % updated_steps_per_epoch == 0):  # 2. every epoch
                    if accelerator.is_main_process:
                        wandb.log({
                            "training/psnr": psnr.item(),
                            "training/ssim": ssim.item(),
                            "training/lpips": lpips.item(),
                            "training/loss": loss.item(),
                            "training/lr": lr_scheduler.get_last_lr()[0]
                        }, step=global_update_step)

                        if depth_loss is not None:
                            wandb.log({"training/depth_loss": depth_loss.item()}, step=global_update_step)
                        if depth_render_loss is not None:
                            wandb.log({"training/depth_render_loss": depth_render_loss.item()}, step=global_update_step)
                        if opacity_loss is not None:
                            wandb.log({"training/opacity_loss": opacity_loss.item()}, step=global_update_step)
                        if motion_loss is not None:
                            wandb.log({"training/motion_loss": motion_loss.item()}, step=global_update_step)
                        if motion_reg is not None:
                            wandb.log({"training/motion_reg": motion_reg.item()}, step=global_update_step)
                        if gaussian_usage is not None:
                            wandb.log({"training/gaussian_usage": gaussian_usage.item()}, step=global_update_step)
                        if voxel_ratio is not None:
                            wandb.log({"training/voxel_ratio": voxel_ratio.item()}, step=global_update_step)

                # Save checkpoint
                if global_update_step != 0 and (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                    or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                    or global_update_step == args.max_train_steps-1):  # 3. last step
                    gc.collect()
                    if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    elif accelerator.is_main_process:
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    accelerator.wait_for_everyone()  # ensure all processes have finished saving
                    if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                        if accelerator.is_local_main_process:
                            if args.hdfs_dir is not None:
                                util.save_ckpt(ckpt_dir, global_update_step, args.hdfs_dir, accelerator.process_index)
                    elif accelerator.is_main_process:
                        if args.hdfs_dir is not None:
                            util.save_ckpt(ckpt_dir, global_update_step, args.hdfs_dir, accelerator.process_index)
                    gc.collect()

                # Evaluate on the validation set
                if ((global_update_step % configs["train"]["early_eval_freq"] == 0 and
                    global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                    or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                    or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                    or global_update_step == args.max_train_steps-1):  # 4. last step

                    torch.cuda.empty_cache()
                    gc.collect()

                    with torch.no_grad():
                        model.eval()

                        all_val_matrics, val_steps = {}, 0
                        val_progress_bar = tqdm(
                            range(len(val_loader)) if args.max_val_steps is None \
                                else range(args.max_val_steps),
                            desc="Validation",
                            ncols=125,
                            disable=not accelerator.is_main_process
                        )
                        for val_batch in val_loader:

                            for k in val_batch:
                                if isinstance(val_batch[k], Tensor):
                                    val_batch[k] = val_batch[k].to(device=accelerator.device, dtype=weight_dtype)

                            val_outputs = model(val_batch, dtype=weight_dtype)

                            val_psnr = val_outputs["psnr"]
                            val_ssim = val_outputs["ssim"]
                            val_lpips = val_outputs["lpips"]
                            val_loss = val_outputs["loss"]

                            # Some extra outputs for logging
                            val_depth_loss = val_outputs["depth_loss"] if "depth_loss" in val_outputs else None
                            val_depth_render_loss = val_outputs["depth_render_loss"] if "depth_render_loss" in val_outputs else None
                            val_opacity_loss = val_outputs["opacity_loss"] if "opacity_loss" in val_outputs else None
                            val_motion_loss = val_outputs["motion_loss"] if "motion_loss" in val_outputs else None
                            val_motion_reg = val_outputs["motion_reg"] if "motion_reg" in val_outputs else None
                            val_gaussian_usage = val_outputs["gaussian_usage"] if "gaussian_usage" in val_outputs else None
                            val_voxel_ratio = val_outputs["voxel_ratio"] if "voxel_ratio" in val_outputs else None

                            val_psnr = accelerator.gather_for_metrics(val_psnr).mean()
                            val_ssim = accelerator.gather_for_metrics(val_ssim).mean()
                            val_lpips = accelerator.gather_for_metrics(val_lpips).mean()
                            val_loss = accelerator.gather_for_metrics(val_loss).mean()

                            if val_depth_loss is not None:
                                val_depth_loss = accelerator.gather_for_metrics(val_depth_loss).mean()
                            if val_depth_render_loss is not None:
                                val_depth_render_loss = accelerator.gather_for_metrics(val_depth_render_loss).mean()
                            if val_opacity_loss is not None:
                                val_opacity_loss = accelerator.gather_for_metrics(val_opacity_loss).mean()
                            if val_motion_loss is not None:
                                val_motion_loss = accelerator.gather_for_metrics(val_motion_loss).mean()
                            if val_motion_reg is not None:
                                val_motion_reg = accelerator.gather_for_metrics(val_motion_reg).mean()
                            if val_gaussian_usage is not None:
                                val_gaussian_usage = accelerator.gather_for_metrics(val_gaussian_usage).mean()
                            if val_voxel_ratio is not None:
                                val_voxel_ratio = accelerator.gather_for_metrics(val_voxel_ratio).mean()

                            val_logs = {
                                "psnr": val_psnr.item(),
                                "ssim": val_ssim.item(),
                                "lpips": val_lpips.item(),
                                "loss": val_loss.item()
                            }
                            val_progress_bar.set_postfix(**val_logs)
                            val_progress_bar.update(1)
                            val_steps += 1

                            all_val_matrics.setdefault("psnr", []).append(val_psnr)
                            all_val_matrics.setdefault("ssim", []).append(val_ssim)
                            all_val_matrics.setdefault("lpips", []).append(val_lpips)
                            all_val_matrics.setdefault("loss", []).append(val_loss)

                            if val_depth_loss is not None:
                                all_val_matrics.setdefault("depth_loss", []).append(val_depth_loss)
                            if val_depth_render_loss is not None:
                                all_val_matrics.setdefault("depth_render_loss", []).append(val_depth_render_loss)
                            if val_opacity_loss is not None:
                                all_val_matrics.setdefault("opacity_loss", []).append(val_opacity_loss)
                            if val_motion_loss is not None:
                                all_val_matrics.setdefault("motion_loss", []).append(val_motion_loss)
                            if val_motion_reg is not None:
                                all_val_matrics.setdefault("motion_reg", []).append(val_motion_reg)
                            if val_gaussian_usage is not None:
                                all_val_matrics.setdefault("gaussian_usage", []).append(val_gaussian_usage)
                            if val_voxel_ratio is not None:
                                all_val_matrics.setdefault("voxel_ratio", []).append(val_voxel_ratio)

                            if args.max_val_steps is not None and val_steps == args.max_val_steps:
                                break

                    val_progress_bar.close()

                    for k, v in all_val_matrics.items():
                        all_val_matrics[k] = torch.tensor(v).mean()

                    logger.info(
                        f"Eval [{global_update_step:06d} / {total_updated_steps:06d}] " +
                        f"psnr: {all_val_matrics['psnr'].item():.4f}, " +
                        f"ssim: {all_val_matrics['ssim'].item():.4f}, " +
                        f"lpips: {all_val_matrics['lpips'].item():.4f}, " +
                        f"loss: {all_val_matrics['loss'].item():.4f}\n"
                    )

                    outputs = accelerator.gather(outputs)
                    val_outputs = accelerator.gather_for_metrics(val_outputs)

                    if accelerator.is_main_process:
                        wandb.log({
                            "validation/psnr": all_val_matrics["psnr"].item(),
                            "validation/ssim": all_val_matrics["ssim"].item(),
                            "validation/lpips": all_val_matrics["lpips"].item(),
                            "validation/loss": all_val_matrics["loss"].item()
                        }, step=global_update_step)

                        for name, val in all_val_matrics.items():
                            if name not in {"psnr", "ssim", "lpips", "loss"}:
                                wandb.log({
                                    f"validation/{name}": val.item()
                                }, step=global_update_step)

                        # Visualize rendering
                        wandb.log({
                            "videos/training": vis_util.wandb_video_log(outputs)
                        }, step=global_update_step)
                        wandb.log({
                            "videos/validation": vis_util.wandb_video_log(val_outputs)
                        }, step=global_update_step)

                    torch.cuda.empty_cache()
                    gc.collect()

                # Update training step
                global_update_step += 1
                step_tracker.set_step(global_update_step)


if __name__ == "__main__":
    main()
