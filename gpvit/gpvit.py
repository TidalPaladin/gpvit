from typing import Optional, Tuple

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .layers import GroupPropagation, WindowAttention


class GPViT(nn.Module):
    """
    GPViT is a class for Vision Transformer with Group Propagation.

    Attributes:
        dim: The dimension of the input data.
        num_group_tokens: The number of group tokens.
        depth: The depth of the network.
        img_size: The size of the input image.
        patch_size: The size of the patches to be extracted from the image.
        window_size: The size of the window for the WindowAttention layer.
        in_channels: The number of input channels.
        group_interval: The interval for group propagation.
        dropout: The dropout rate.
        activation: The activation function to use.
        nhead: The number of heads in the multihead attention models. Defaults to
            ``dim // 64`` for Flash Attention.
    """

    def __init__(
        self,
        dim: int,
        num_group_tokens: int,
        depth: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int] = (16, 16),
        window_size: Tuple[int, int] = (7, 7),
        in_channels: int = 3,
        group_interval: int = 3,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        nhead: Optional[int] = None,
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 64
        self._img_size = img_size
        self._patch_size = patch_size
        self._window_size = window_size
        self._in_channels = in_channels

        H, W = self.tokenized_size
        L = H * W
        self.position = nn.Parameter(torch.randn(1, L, dim))
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(dim),
        )
        self.group_tokens = nn.Parameter(torch.randn(1, num_group_tokens, dim))

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            is_group_propagation = i % group_interval == group_interval - 1
            if is_group_propagation:
                token_hidden_dim = num_group_tokens * 4
                channel_hidden_dim = dim * 4
                block = GroupPropagation(
                    dim, self.nhead, num_group_tokens, token_hidden_dim, channel_hidden_dim, dropout, activation
                )
            else:
                block = WindowAttention(
                    dim,
                    self.nhead,
                    window_size=self.window_size,
                    grid_size=self.tokenized_size,
                    dropout=dropout,
                    activation=activation,
                )
            self.blocks.append(block)

        self.tokens_to_grid = Rearrange("b (h w) d -> b d h w", h=H, w=W)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.patch_embed(x)
        x = x + self.position

        B = x.shape[0]
        group_tokens = self.group_tokens.expand(B, -1, -1)
        for block in self.blocks:
            if isinstance(block, GroupPropagation):
                _x, _group_tokens = block(x, group_tokens)
                x = x + _x
                group_tokens = group_tokens + _group_tokens
            else:
                x = x + block(x)

        x = self.tokens_to_grid(x)
        return x, group_tokens

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def img_size(self) -> Tuple[int, int]:
        return self._img_size

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def window_size(self) -> Tuple[int, int]:
        return self._window_size

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    @property
    def tokenized_size(self) -> Tuple[int, int]:
        H = self.img_size[0] // self.patch_size[0]
        W = self.img_size[1] // self.patch_size[1]
        return H, W
