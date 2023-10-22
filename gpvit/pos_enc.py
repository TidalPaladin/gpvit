import math
from typing import Iterable, Optional, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class PositionEncoder(nn.Module):
    r"""Base class for positional encodings"""

    def __init__(self):
        super().__init__()

    def relative_forward(self, x: Tensor, neighbors: Tensor) -> Tensor:
        r"""Computes the positional encoding using relative distances between coordinates in
        ``x`` and coordinates in ``neighbors``.
        Shapes:
            * ``x`` - :math:`(L, N, C)`
            * ``neighbors`` - :math:`(K, L, N, C)` where :math:`K` is the neighborhood of relative coordinates
              for a given coordinate in ``x``.
            * Output - :math:`(K, L, N, C)`
        """
        L, N, C = x.shape
        K, L, N, C = neighbors.shape
        delta = neighbors - x.view(1, L, N, C)
        return self(delta.view(-1, N, C)).view(K, L, N, C)

    def from_grid(
        self,
        dims: Iterable[int],
        batch_size: int = 1,
        proto: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        r"""Creates positional encodings for a coordinate space with lengths given in ``dims``.
        Args:
            dims:
                Forwarded to :func:`create_grid`
            batch_size:
                Batch size, for matching the coordinate grid against a batch of vectors that need
                positional encoding.
            proto:
                Forwarded to :func:`create_grid`
        Keyword Args:
            Forwarded to :func:`create_grid`
        Shapes:
            * Output - :math:`(L, N, D)` where :math:`D` is the embedding size, :math:`L` is ``product(dims)``,
              and :math:`N` is ``batch_size``.
        """
        grid = self.create_grid(dims, proto, *args, **kwargs)
        pos_enc = self(grid).expand(batch_size, -1, -1)
        return pos_enc

    @staticmethod
    def create_grid(
        dims: Iterable[int],
        proto: Optional[Tensor] = None,
        requires_grad: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> Tensor:
        r"""Create a grid of coordinate values given the size of each dimension.
        Args:
            dims:
                The length of each dimension
            proto:
                If provided, a source tensor with which to match device / requires_grad
            requires_grad:
                Optional override for requires_grad
            normalize:
                If true, normalize coordinate values on the range :math:`\[-1, 1\]`
        Keyword Args:
            ``"device"`` or ``"dtype"``, used to set properties of the created grid tensor
        Shapes:
            * Output - :math:`(L, 1, C)` where :math:`C` is ``len(dims)`` and :math:`L` is ``product(dims)``
        """
        if proto is not None:
            device = proto.device
            dtype = proto.dtype
            _kwargs = {"device": device, "dtype": dtype}
            _kwargs.update(kwargs)
            kwargs = _kwargs

        with torch.no_grad():
            lens = tuple(torch.arange(d, **kwargs) for d in dims)
            grid = torch.stack(torch.meshgrid(*lens, indexing="ij"), dim=0)

            if normalize:
                C = grid.shape[0]
                scale = grid.view(C, -1).amax(dim=-1).view(C, *((1,) * (grid.ndim - 1)))
                grid.div_(scale).sub_(0.5).mul_(2)

            grid = rearrange(grid, "c ... -> () (...) c")

        requires_grad = requires_grad or (proto is not None and proto.requires_grad)
        grid.requires_grad = requires_grad
        return grid


class FourierLogspace(PositionEncoder):
    scales: Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_freq: float,
        num_bands: int,
        zero_one_norm: bool = True,
        base: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        start = 0 if zero_one_norm else -1
        stop = math.log(max_freq / 2) / math.log(2)
        self.scales = nn.Parameter(math.pi * torch.logspace(start, stop, num_bands, base=base))
        d_mlp = self.scales.numel() * d_in * 2
        dim_ff = 4 * max(d_out, d_mlp)
        self.mlp = nn.Sequential(
            nn.Linear(d_mlp, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_out),
        )

    @property
    def d_out(self) -> int:
        return cast(nn.Linear, self.mlp[-1]).out_features

    @property
    def num_bands(self) -> int:
        return self.scales.numel()

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "n l c -> n l c ()")
        x = x * self.scales.view(1, 1, 1, -1)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = rearrange(x, "n l c d -> n l (c d)")
        x = self.mlp(x)
        return x
