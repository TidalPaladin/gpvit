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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.token_mixing(x))
        x = self.norm2(x + self.channel_mixing(x))
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
            raise ValueError(f"Window height {Hw} does not divide grid height {Hi}")  # pragma: no cover
        if not Wi % Ww == 0:
            raise ValueError(f"Window width {Ww} does not divide grid width {Wi}")  # pragma: no cover

        H, W = Hi // Hw, Wi // Ww
        self.grid_size = grid_size
        self.window = Rearrange("n (h hw w ww) c -> (n h w) (hw ww) c", hw=Hw, ww=Ww, h=H, w=W)
        self.unwindow = Rearrange("(n h w) (hw ww) c -> n (h hw w ww) c", hw=Hw, ww=Ww, h=H, w=W)

    def forward(self, x: Tensor) -> Tensor:
        x = self.window(x)
        x = self.transformer_layer(x)
        x = self.unwindow(x)
        return x


class GroupPropagation(nn.Module, ABC):
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
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention 2
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.q2 = nn.Linear(d_model, d_model)
        self.kv2 = nn.Linear(d_model, 2 * d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        q = self.q1(group_tokens)
        _tokens = torch.cat([tokens, group_tokens], dim=1) if self.group_tokens_as_kv else tokens
        k, v = self.kv1(_tokens).chunk(2, dim=-1)
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
        _tokens = tokens
        if self.conv is not None and self.norm3 is not None:
            _tokens = self.conv(_tokens)

        # MLP
        tokens = self.norm3(tokens + self.mlp(_tokens))

        return tokens, group_tokens


class GroupPropagationMLPMixer(GroupPropagation):
    """
    Group Propagation class for applying group propagation to an input tensor.
    Uses MLPMixer for group token mixing.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        num_tokens: The number of tokens.
        token_dim: The dimension of the tokens.
        channel_dim: The dimension of the channels.
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
        token_dim: int,
        channel_dim: int,
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

        # MLPMixer for group tokens
        self.mixer = MLPMixer(d_model, token_dim, num_tokens, channel_dim, dropout=dropout, activation=activation)


class GroupPropagationTransformer(GroupPropagation):
    """
    Group Propagation class for applying group propagation to an input tensor.
    Uses a transformer for group token mixing.

    Args:
        d_model: The dimension of the model.
        nhead: The number of heads in the multihead attention models.
        num_transformer_layers: The number of transformer layers to use in group token mixing.
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
        num_transformer_layers: int = 1,
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
            ),
            num_transformer_layers,
        )
