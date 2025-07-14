from typing import *
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch import optim
from torch.optim import lr_scheduler
from diffusers.optimization import get_scheduler

from src.models.splatrecon import SplatRecon


def get_optimizer(name: str, params: Parameter, **kwargs) -> Optimizer:
    if name == "adamw":
        return optim.AdamW(params=params, **kwargs)
    else:
        raise NotImplementedError(f"Not implemented optimizer: [{name}]")


def get_lr_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    if name == "one_cycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs["max_lr"],
            total_steps=kwargs["total_steps"],
            pct_start=kwargs["pct_start"],
        )
    elif name == "cosine_warmup":
        return get_scheduler(
            "cosine", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant_warmup":
        return get_scheduler(
            "constant_with_warmup", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant":
        return lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda _: 1)
    elif name == "linear_decay":
        return lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: max(0., 1. - epoch / kwargs["total_epochs"]),
        )
    else:
        raise NotImplementedError(f"Not implemented lr scheduler: [{name}]")
