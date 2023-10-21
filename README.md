# GP-ViT

Implementation of [GP-ViT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation](https://arxiv.org/abs/2212.06795)

## Setup

Run `make init` to install the project to a `pdm` virtual environment

## Usage

```python
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
    ...
```