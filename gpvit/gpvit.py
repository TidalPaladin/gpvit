from typing import Callable, Optional, Tuple

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .layers import GroupPropagation, WindowAttention
from .pos_enc import FourierLogspace


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
        conv: Whether to use a DW conv in each GP block. The kernel size is set to
            ``window_size``. Defaults to True.
        pos_enc: Position encoding to use. Can be one of ``"fourier"``, ``"learned"``.
        reshape_output: Whether to reshape the output to a grid. Defaults to True.

    Shapes:
        * Input - :math:`(B, C, H, W)`
        * Output - :math:`(B, D, H', W')`, where :math:`H' = H // patch_size[0]` and
            :math:`W' = W // patch_size[1]`, and :math:`D` is the dimension of the input data.
            If ``reshape_output`` is False, then the output shape is :math:`(B, H'*W', D)`.
        * Group tokens - :math:`(B, L, D)` where :math:`L` is the number of group tokens
            and :math:`D` is the dimension of the input data.

    Returns:
        Tuple[Tensor, Tensor]: The output of the model and the output of the group tokens.
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
        conv: bool = True,
        pos_enc: str = "learned",
        reshape_output: bool = True,
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 64
        self._img_size = img_size
        self._patch_size = patch_size
        self._window_size = window_size
        self._in_channels = in_channels
        self._conv = conv
        self._reshape_output = reshape_output

        # Initialize group tokens
        self.group_tokens = nn.Parameter(torch.randn(1, num_group_tokens, dim))

        # Patch embedding
        H, W = self.tokenized_size
        L = H * W
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(dim),
        )
        self.stem_norm = nn.LayerNorm(dim)

        # Position encoding
        if pos_enc == "fourier":
            self.position = FourierLogspace(2, dim, L, dim // 2, zero_one_norm=False, dropout=dropout)
        else:
            self.position = nn.Parameter(torch.randn(1, L, dim))

        # Body
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            is_group_propagation = i % group_interval == group_interval - 1
            if is_group_propagation:
                token_hidden_dim = num_group_tokens * 4
                channel_hidden_dim = dim * 4
                block = GroupPropagation(
                    dim,
                    self.nhead,
                    num_group_tokens,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dropout,
                    activation=activation,
                    kernel_size=self.kernel_size,
                    tokenized_size=self.tokenized_size,
                )
            else:
                # Ensure we use an activation form that will be accelerated by TransformerEncoderLayer
                act = "gelu" if activation == nn.GELU() else "relu" if activation == nn.ReLU() else activation
                block = WindowAttention(
                    dim,
                    self.nhead,
                    window_size=self.window_size,
                    grid_size=self.tokenized_size,
                    dropout=dropout,
                    activation=act,
                )
            self.blocks.append(block)

        self.tokens_to_grid = Rearrange("b (h w) d -> b d h w", h=H, w=W)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Patch embed
        x = self.patch_embed(x)

        # Position encoding
        B = x.shape[0]
        if isinstance(self.position, nn.Parameter):
            pos = self.position
        else:
            pos = self.position.from_grid(list(self.tokenized_size), proto=x, batch_size=B)
        x = self.stem_norm(x + pos)

        # Body
        group_tokens = self.group_tokens.expand(B, -1, -1)
        for block in self.blocks:
            if isinstance(block, GroupPropagation):
                x, group_tokens = block(x, group_tokens)
            else:
                x = block(x)

        # Unpatch and return
        if self.reshape_output:
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

    @property
    def conv(self) -> bool:
        return self._conv

    @property
    def kernel_size(self) -> Optional[Tuple[int, int]]:
        # Ensure kernel size is odd
        if self.conv:
            # Not scriptable
            # return tuple(k if k % 2 != 0 else k + 1 for k in self.window_size)
            H, W = self.window_size
            return (
                H if H % 2 != 0 else H + 1,
                W if W % 2 != 0 else W + 1,
            )
        return None

    @property
    def reshape_output(self) -> bool:
        return self._reshape_output

    def register_mask_hook(self, func: Callable, *args, **kwargs) -> RemovableHandle:
        r"""Register a token masking hook to be applied after the patch embedding step.

        Args:
            func: Callable token making hook with signature given in :func:`register_forward_hook`

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``.
        """
        return self.patch_embed.register_forward_hook(func, *args, **kwargs)

    def unpatch(self, x: Tensor) -> Tensor:
        """
        Unpatches the input tensor into the original image. It is assumed that the
        second dimension is of size ``H_p W_p C`` where ``H_p`` and ``W_p`` are the
        patch size and ``C`` is the number of channels. This function is used to invert
        the patch embedding step for tasks like MAE.

        Args:
            x: The input tensor to be unpatched.

        Shapes:
            * ``x`` - :math:`(B, H_p W_p C, H, W)`
            - output: :math:`(B, C, H H_p, W W_p)`

        Returns:
            Tensor: The unpatched tensor.
        """
        Hp, Wp = self.patch_size
        H, W = self.tokenized_size
        return rearrange(
            x,
            "b (hp wp c) h w -> b c (h hp) (w wp)",
            hp=Hp,
            wp=Wp,
            h=H,
            w=W,
        )
