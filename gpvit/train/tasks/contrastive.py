from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.contrastive.task import ContrastiveEmbedding as ContrastiveEmbeddingBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from gpvit import GPViT, MLPMixerPooling

from .. import BACKBONES
from .helpers import mask_fn


class ContrastiveEmbedding(ContrastiveEmbeddingBase):
    r"""Implements Contrastive Embedding pretraining.

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        backbone = cast(GPViT, self.backbone)
        dim = backbone.dim
        num_tokens = backbone.num_group_tokens
        return nn.Sequential(
            MLPMixerPooling(dim, 4 * num_tokens, num_tokens, 4 * dim, dropout=0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, elementwise_affine=False),
        )

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        N = x.shape[0]
        _, cls = self.backbone(x)
        cls = self.embed_head(cls).view(N, -1)

        if mask_hook is not None:
            mask_hook.remove()

        return {"embed": cls}
