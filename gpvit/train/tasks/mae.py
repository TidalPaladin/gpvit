from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.mae.task import MAE as MAEBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from ..import BACKBONES
from .helpers import mask_fn


class MAE(MAEBase):
    r"""Implements MAE style pretraining.

    Args:
        backbone: Backbone architecture for the model.
        mask_ratio: Ratio of tokens to mask. Defaults to 0.4.
        mask_scale: Scale of the mask. Increasing this will mask tokens in larger groups. Defaults to 2.
        optimizer_init: Initial configuration for the optimizer.
        lr_scheduler_init: Initial configuration for the learning rate scheduler. 
        lr_interval: Interval for learning rate update. Defaults to "epoch".
        lr_monitor: Metric to monitor for learning rate scheduler. Defaults to "train/total_loss_epoch".
        checkpoint: Path to the checkpoint file. 
        strict_checkpoint: If True, loading checkpoint is strict. 
        log_train_metrics_interval: Interval for logging training metrics. 
        log_train_metrics_on_epoch: If True, logs training metrics on epoch end. 
        weight_decay_exemptions: Set of exemptions for weight decay. 

    """

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        dim = cast(Any, self.backbone).dim
        out_dim = cast(Any, self.backbone).in_channels
        patch_h, patch_w = cast(Any, self.backbone).patch_size
        outputs_per_token = out_dim * patch_h * patch_w
        return nn.Conv2d(dim, outputs_per_token, kernel_size=1)

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        x, _ = self.backbone(x)
        x = self.mae_head(x)
        x = cast(Any, self.backbone).unpatch(x)

        if mask_hook is not None:
            mask_hook.remove()
        return {"mae": x}
