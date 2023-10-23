from copy import copy
from typing import Optional, Tuple

from einops.layers.torch import Rearrange
from torch import Tensor, nn


class MLPMixer(nn.Module):
    """
    MLP Mixer class for mixing tokens and channels in an input tensor.

    Args:
        dim: The dimension of the input tensor.
        token_dim: The hidden dimension of the token-mixing MLP.
        num_patches: The number of patches to be considered for token mixing.
        channel_dim: The hidden dimension of the channel-mixing MLP.
        dropout: Dropout rate for the MLPs. Defaults to 0.0.
        activation: Activation function for the MLPs. Defaults to nn.GELU().
    """

    def __init__(
        self,
        dim: int,
        token_dim: int,
        num_patches: int,
        channel_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            Rearrange("b l d -> b d l", d=dim, l=num_patches),
            nn.Linear(num_patches, token_dim),
            copy(activation),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_patches),
            Rearrange("b d l -> b l d", d=dim, l=num_patches),
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(dim, channel_dim),
            copy(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_mixing(x)
        x = self.norm(x + self.channel_mixing(x))
        return x


class WindowAttention(nn.Module):
    """
    Window Attention class for applying window-based attention to an input tensor.

    Args:
        grid_size: The size of the grid to be considered for window attention.
        window_size: The size of the window to be considered for window attention.

    Raises:
        ValueError: If the window size does not divide the grid size.
    """

    def __init__(
        self,
        *args,
        grid_size: Tuple[int, int],
        window_size: Tuple[int, int],
        **kwargs,
    ):
        # TODO: It is cleaner to inherit from TransformerEncoderLayer but calling super().forward()
        # is not supported in torchscript. See https://github.com/pytorch/pytorch/issues/42885
        super().__init__()
        kwargs.setdefault("batch_first", "True")
        self.transformer_layer = nn.TransformerEncoderLayer(*args, **kwargs)

        Hw, Ww = window_size
        Hi, Wi = grid_size

        if not Hi % Hw == 0:
            raise ValueError(f"Window height {Hw} does not divide grid height {Hi}")
        if not Wi % Ww == 0:
            raise ValueError(f"Window width {Ww} does not divide grid width {Wi}")

        H, W = Hi // Hw, Wi // Ww
        self.grid_size = grid_size
        self.window = Rearrange("n (h hw w ww) c -> (n h w) (hw ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.unwindow = Rearrange("(n h w) (hw ww) c -> n (h hw w ww) c", hw=Hw, ww=Ww, h=H, w=W)

    def forward(self, x: Tensor) -> Tensor:
        x = self.window(x)
        x = self.transformer_layer(x)
        x = self.unwindow(x)
        return x


class GroupPropagation(nn.Module):
    """
    Group Propagation class for applying group propagation to an input tensor.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        num_tokens: The number of tokens.
        token_dim: The dimension of the tokens.
        channel_dim: The dimension of the channels.
        dropout: The dropout value.
        activation: The activation function to use.
        kernel_size: The kernel size for the DW convolutional layer. Defaults to None,
            meaning no convolutional layer is used.
        tokenized_size: The size of the tokenized image. Only required when using a
            convolutional layer.

    Returns:
        Tuple[Tensor, Tensor]: The tokens and group tokens after propagation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_tokens: int,
        token_dim: int,
        channel_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        kernel_size: Optional[Tuple[int, int]] = None,
        tokenized_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.q1 = nn.Linear(d_model, d_model)
        self.kv1 = nn.Linear(d_model, 2 * d_model)

        self.mixer = MLPMixer(d_model, token_dim, num_tokens, channel_dim, dropout=dropout, activation=activation)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.q2 = nn.Linear(d_model, d_model)
        self.kv2 = nn.Linear(d_model, 2 * d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if kernel_size is not None:
            if tokenized_size is None:
                raise ValueError("tokenized_size must be provided when using a convolutional layer")
            padding = tuple((k - 1) // 2 for k in kernel_size)
            H, W = tokenized_size
            self.conv = nn.Sequential(
                Rearrange("b (h w) d -> b d h w", h=H, w=W, d=d_model),
                nn.Conv2d(d_model, d_model, kernel_size, groups=d_model, padding=padding),
                Rearrange("b d h w -> b (h w) d", h=H, w=W, d=d_model),
            )
        else:
            self.conv = None

    def forward(self, tokens: Tensor, group_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        # Cross attention - group tokens to tokens
        q = self.q1(group_tokens)
        k, v = self.kv1(tokens).chunk(2, dim=-1)
        _group_tokens, _ = self.attn1(q, k, v)
        group_tokens = self.norm1(group_tokens + _group_tokens)

        # Group token mixing
        group_tokens = self.mixer(group_tokens)

        # Cross attention - tokens to group tokens
        q = self.q2(tokens)
        k, v = self.kv2(group_tokens).chunk(2, dim=-1)
        _tokens, _ = self.attn2(q, k, v)
        tokens = self.norm2(tokens + _tokens)

        # Maybe DW conv
        if self.conv is not None:
            tokens = tokens + self.conv(tokens)

        return tokens, group_tokens
