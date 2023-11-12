from typing import Any, Dict, Optional, Set, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.query_patch.task import QueryPatch as QueryPatchBase
from torch import Tensor

from gpvit import GPViT, MLPMixerPooling
from gpvit.layers import MLPMixer

from .. import BACKBONES


class QueryPatch(QueryPatchBase):
    r"""Implements query patch pretraining.

    Args:
        backbone: Backbone architecture for the model.
        decoder_depth: Number of MLPMixer blocks in the decoder.
        augment_batches: Number of unique augmentations per batch. Defaults to 4.
        optimizer_init: Initial configuration for the optimizer. Defaults to {}.
        lr_scheduler_init: Initial configuration for the learning rate scheduler. Defaults to {}.
        lr_interval: Interval for learning rate update. Defaults to "epoch".
        lr_monitor: Metric to monitor for learning rate scheduler. Defaults to "train/total_loss_epoch".
        named_datasets: If True, uses named datasets. Defaults to False.
        checkpoint: Path to the checkpoint file. Defaults to None.
        strict_checkpoint: If True, loading checkpoint is strict. Defaults to True.
        log_train_metrics_interval: Interval for logging training metrics. Defaults to 1.
        log_train_metrics_on_epoch: If True, logs training metrics on epoch end. Defaults to False.
        weight_decay_exemptions: Set of exemptions for weight decay. Defaults to set().
    """

    def __init__(
        self,
        backbone: str,
        decoder_depth: int = 4,
        augment_batches: int = 4,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        weight_decay_exemptions: Set[str] = set(),
    ):
        super().__init__(
            backbone=backbone,
            augment_batches=augment_batches,
            optimizer_init=optimizer_init,
            lr_scheduler_init=lr_scheduler_init,
            lr_interval=lr_interval,
            lr_monitor=lr_monitor,
            named_datasets=named_datasets,
            checkpoint=checkpoint,
            strict_checkpoint=strict_checkpoint,
            log_train_metrics_interval=log_train_metrics_interval,
            log_train_metrics_on_epoch=log_train_metrics_on_epoch,
            weight_decay_exemptions=weight_decay_exemptions,
        )

        self.save_hyperparameters()
        backbone_model = cast(GPViT, self.backbone)
        dim = backbone_model.dim
        group_tokens = 2 * backbone_model.num_group_tokens
        self.box_decoder = nn.Sequential(
            *[MLPMixer(dim, 4 * group_tokens, group_tokens, 4 * dim, dropout=0.1) for _ in range(decoder_depth)],
            self.create_head(),
        )

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        backbone = cast(GPViT, self.backbone)
        dim = backbone.dim
        num_tokens = 2 * backbone.num_group_tokens
        return nn.Sequential(
            MLPMixerPooling(dim, 4 * num_tokens, num_tokens, 4 * dim, dropout=0.1, output_tokens=1),
            nn.Linear(dim, 4),
        )

    def forward(self, x: Tensor, x_box: Tensor) -> Dict[str, Tensor]:
        x.shape[0]

        # Encode the full image
        _, groups = self.backbone(x)

        # Encode the box crop
        _, groups_box = self.backbone(x_box)

        #  Decode
        all_group_tokens = torch.cat([groups, groups_box], dim=-2)
        box_pred = self.box_decoder(all_group_tokens)

        return {"box": box_pred}
