from abc import ABC
from copy import copy
from typing import Optional, Tuple

import torch
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
            nn.Conv1d(num_patches, token_dim, 1),
            copy(activation),
            nn.Dropout(dropout),
            nn.Conv1d(token_dim, num_patches, 1),
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(dim, channel_dim),
            copy(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.token_mixing(self.norm1(x))
        x = x + self.channel_mixing(self.norm2(x))
        return x


class WindowMixer(nn.Module, ABC):
    """
    Window class for applying window-based mixing to an input tensor.

    Args:
        grid_size: The size of the grid to be considered for window attention.
        window_size: The size of the window to be considered for window attention.

    Raises:
        ValueError: If the window size does not divide the grid size.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        window_size: Tuple[int, int],
    ):
        super().__init__()
        self._grid_size = grid_size
        self._window_size = window_size

        Hw, Ww = window_size
        Hi, Wi = grid_size
        if not Hi % Hw == 0:
            raise ValueError(f"Window height {Hw} does not divide grid height {Hi}")  # pragma: no cover
        if not Wi % Ww == 0:
            raise ValueError(f"Window width {Ww} does not divide grid width {Wi}")  # pragma: no cover

        H, W = self.size_after_windowing
        self.window = Rearrange("n (h hw w ww) c -> (n h w) (hw ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.unwindow = Rearrange("(n h w) (hw ww) c -> n (h hw w ww) c", hw=Hw, ww=Ww, h=H, w=W)

    @property
    def grid_size(self) -> Tuple[int, int]:
        """The size of the total grid to be considered for window attention."""
        return self._grid_size

    @property
    def window_size(self) -> Tuple[int, int]:
        """The size of the window to be considered for window attention."""
        return self._window_size

    @property
    def size_after_windowing(self) -> Tuple[int, int]:
        """The size of the grid once windowed"""
        Hg, Wg = self.grid_size
        Hw, Ww = self.window_size
        return (Hg // Hw), (Wg // Ww)

    @property
    def num_windows(self) -> int:
        """The number of windows in the grid."""
        H, W = self.size_after_windowing
        return H * W

    @property
    def tokens_per_window(self) -> int:
        """The number of tokens in a window."""
        H, W = self.window_size
        return H * W

    def forward(self, x: Tensor) -> Tensor:
        x = self.window(x)
        x = self.mixer(x)
        x = self.unwindow(x)
        return x

    def group_forward(self, tokens: Tensor, group_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        return self(tokens), group_tokens


class WindowAttention(WindowMixer):
    """
    Window mixing class for applying window-based attention to an input tensor.

    Args:
        dim: The dimension of the input tensor.
        nhead: The number of heads in the multihead attention models.
        grid_size: The size of the grid to be considered for window attention.
        window_size: The size of the window to be considered for window attention.

    Keyword Args:
        Forwarded to :class:`nn.TransformerEncoderLayer`.

    Raises:
        ValueError: If the window size does not divide the grid size.
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        grid_size: Tuple[int, int],
        window_size: Tuple[int, int],
        **kwargs,
    ):
        # TODO: It is cleaner to inherit from TransformerEncoderLayer but calling super().forward()
        # is not supported in torchscript. See https://github.com/pytorch/pytorch/issues/42885
        super().__init__(grid_size, window_size)
        kwargs.setdefault("batch_first", "True")
        self.mixer = nn.TransformerEncoderLayer(dim, nhead, **kwargs)


class WindowMLPMixer(WindowMixer):
    """
    Window mixing class for applying window-based MLP-mixing to an input tensor.

    Args:
        dim: The dimension of the input tensor.
        token_dim: The hidden dimension of the token-mixing MLP. If ``None``, defaults to
            ``4 * num_tokens``.
        channel_dim: The hidden dimension of the channel-mixing MLP. If ``None``, defaults to
            ``4 * dim``.
        grid_size: The size of the grid to be considered for window attention.
        window_size: The size of the window to be considered for window attention.

    Keyword Args:
        Forwarded to :class:`MLPMixer`.

    Raises:
        ValueError: If the window size does not divide the grid size.
    """

    def __init__(
        self,
        dim: int,
        token_dim: Optional[int],
        channel_dim: Optional[int],
        grid_size: Tuple[int, int],
        window_size: Tuple[int, int],
        **kwargs,
    ):
        super().__init__(grid_size, window_size)
        token_dim = token_dim or 4 * self.tokens_per_window
        channel_dim = channel_dim or 4 * dim
        self.mixer = MLPMixer(dim, token_dim, self.tokens_per_window, channel_dim, **kwargs)


class GroupPropagation(nn.Module):
    """
    Group Propagation class for applying group propagation to an input tensor.
    Subclass to implement group token mixing.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        dropout: The dropout value.
        mlp_hidden_dim: Hidden dimension of the MLPs. Defaults to ``4 * d_model``.
        activation: The activation function to use.
        kernel_size: The kernel size for the DW convolutional layer. Defaults to None,
            meaning no convolutional layer is used.
        tokenized_size: The size of the tokenized image. Only required when using a
            convolutional layer.
        group_tokens_as_kv: Whether to use the group tokens as the key and value in the
            first cross attention layer. Setting this to ``True`` allows the group tokens
            to attend to both the tokens and the group tokens. Otherwise the group tokens
            attend only to the tokens. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: The tokens and group tokens after propagation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        mlp_hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        kernel_size: Optional[Tuple[int, int]] = None,
        tokenized_size: Optional[Tuple[int, int]] = None,
        group_tokens_as_kv: bool = False,
    ):
        super().__init__()
        self.mixer = nn.Identity()
        self.group_tokens_as_kv = group_tokens_as_kv

        # Cross attention 1
        self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.q1 = nn.Linear(d_model, d_model)
        self.kv1 = nn.Linear(d_model, 2 * d_model)
        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_kv = nn.LayerNorm(d_model)

        # Cross attention 2
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.q2 = nn.Linear(d_model, d_model)
        self.kv2 = nn.Linear(d_model, 2 * d_model)
        self.norm2_q = nn.LayerNorm(d_model)
        self.norm2_kv = nn.LayerNorm(d_model)

        # Possible DW conv for tokens
        if kernel_size is not None:
            if tokenized_size is None:
                raise ValueError("tokenized_size must be provided when using a convolutional layer")  # pragma: no cover
            padding = tuple((k - 1) // 2 for k in kernel_size)
            H, W = tokenized_size
            self.conv = nn.Sequential(
                Rearrange("b (h w) d -> b d h w", h=H, w=W, d=d_model),
                nn.Conv2d(d_model, d_model, kernel_size, groups=d_model, padding=padding),
                Rearrange("b d h w -> b (h w) d", h=H, w=W, d=d_model),
            )
        else:
            self.conv = None

        # MLP for tokens
        mlp_hidden_dim = d_model * 4 if mlp_hidden_dim is None else mlp_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            copy(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, group_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        # Cross attention - group tokens to tokens
        q = self.q1(self.norm1_q(group_tokens))
        _tokens = torch.cat([tokens, group_tokens], dim=1) if self.group_tokens_as_kv else tokens
        k, v = self.kv1(self.norm1_kv(_tokens)).chunk(2, dim=-1)
        _group_tokens, _ = self.attn1(q, k, v)
        group_tokens = group_tokens + _group_tokens

        # Group token mixing
        group_tokens = self.mixer(group_tokens)

        # Cross attention - tokens to group tokens
        q = self.q2(self.norm2_q(tokens))
        k, v = self.kv2(self.norm2_kv(group_tokens)).chunk(2, dim=-1)
        _tokens, _ = self.attn2(q, k, v)
        tokens = tokens + _tokens

        # Maybe DW conv
        _tokens = tokens
        if self.conv is not None and self.norm3 is not None:
            _tokens = self.conv(_tokens)

        # MLP
        tokens = tokens + self.mlp(self.norm3(_tokens))

        return tokens, group_tokens

    def group_forward(self, tokens: Tensor, group_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        return self(tokens, group_tokens)


class GroupPropagationMLPMixer(GroupPropagation):
    """
    Group Propagation class for applying group propagation to an input tensor.
    Uses MLPMixer for group token mixing.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        num_tokens: The number of tokens.
        token_dim: Dimension of the token mixing MLP. Defaults to ``4 * num_tokens``.
        channel_dim: Dimension of the channel mixing MLP. Defaults to ``4 * d_model``.
        mixer_repeats: The number of times to repeat the mixer.
        dropout: The dropout value.
        mlp_hidden_dim: Hidden dimension of the MLPs. Defaults to ``4 * d_model``.
        activation: The activation function to use.
        kernel_size: The kernel size for the DW convolutional layer. Defaults to None,
            meaning no convolutional layer is used.
        tokenized_size: The size of the tokenized image. Only required when using a
            convolutional layer.
        group_tokens_as_kv: Whether to use the group tokens as the key and value in the
            first cross attention layer. Setting this to ``True`` allows the group tokens
            to attend to both the tokens and the group tokens. Otherwise the group tokens
            attend only to the tokens. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: The tokens and group tokens after propagation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_tokens: int,
        token_dim: Optional[int] = None,
        channel_dim: Optional[int] = None,
        mixer_repeats: int = 1,
        dropout: float = 0.0,
        mlp_hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        kernel_size: Optional[Tuple[int, int]] = None,
        tokenized_size: Optional[Tuple[int, int]] = None,
        group_tokens_as_kv: bool = False,
    ):
        super().__init__(
            d_model,
            nhead,
            dropout,
            mlp_hidden_dim,
            activation,
            kernel_size,
            tokenized_size,
            group_tokens_as_kv,
        )
        token_dim = num_tokens * 4 if token_dim is None else token_dim
        channel_dim = d_model * 4 if channel_dim is None else channel_dim

        # MLPMixer for group tokens
        self.mixer = nn.Sequential(
            *[
                MLPMixer(d_model, token_dim, num_tokens, channel_dim, dropout=dropout, activation=activation)
                for _ in range(mixer_repeats)
            ]
        )


class GroupPropagationTransformer(GroupPropagation):
    """
    Group Propagation class for applying group propagation to an input tensor.
    Uses a transformer for group token mixing.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        mixer_repeats: Number of times to repeat the mixer.
        dropout: The dropout value.
        mlp_hidden_dim: Hidden dimension of the MLPs. Defaults to ``4 * d_model``.
        activation: The activation function to use.
        kernel_size: The kernel size for the DW convolutional layer. Defaults to None,
            meaning no convolutional layer is used.
        tokenized_size: The size of the tokenized image. Only required when using a
            convolutional layer.
        group_tokens_as_kv: Whether to use the group tokens as the key and value in the
            first cross attention layer. Setting this to ``True`` allows the group tokens
            to attend to both the tokens and the group tokens. Otherwise the group tokens
            attend only to the tokens. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: The tokens and group tokens after propagation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        mixer_repeats: int = 1,
        dropout: float = 0.0,
        mlp_hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        kernel_size: Optional[Tuple[int, int]] = None,
        tokenized_size: Optional[Tuple[int, int]] = None,
        group_tokens_as_kv: bool = False,
    ):
        super().__init__(
            d_model,
            nhead,
            dropout,
            mlp_hidden_dim,
            activation,
            kernel_size,
            tokenized_size,
            group_tokens_as_kv,
        )

        # Transformer for group tokens
        self.mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
                activation=(
                    "gelu"
                    if isinstance(activation, nn.GELU)
                    else "relu"
                    if isinstance(activation, nn.ReLU)
                    else activation
                ),
                norm_first=True,
            ),
            mixer_repeats,
        )


class MLPMixerPooling(MLPMixer):
    """
    MLP Mixer pooling head for group propagation tokens.

    Args:
        dim: The dimension of the input tensor.
        token_dim: The hidden dimension of the token-mixing MLP.
        num_patches: The number of patches to be considered for token mixing.
        channel_dim: The hidden dimension of the channel-mixing MLP.
        dropout: Dropout rate for the MLPs. Defaults to 0.0.
        activation: Activation function for the MLPs. Defaults to nn.GELU().
        output_tokens: The number of output tokens. Defaults to 1.
    """

    def __init__(
        self,
        dim: int,
        token_dim: int,
        num_patches: int,
        channel_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        output_tokens: int = 1,
    ):
        super().__init__(dim, token_dim, num_patches, channel_dim, dropout, activation)
        self.token_mixing = nn.Sequential(
            nn.Conv1d(num_patches, token_dim, 1),
            copy(activation),
            nn.Dropout(dropout),
            nn.Conv1d(token_dim, output_tokens, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_mixing(self.norm1(x))
        x = x + self.channel_mixing(self.norm2(x))
        return x
