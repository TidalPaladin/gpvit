import pytest
import torch

from gpvit import GPViT


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_pass(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4)).to(
        device
    )
    x = torch.randn(1, 3, 64, 64).to(device)

    output, group_output = model(x)
    assert output.shape == (1, 128, 8, 8)
    assert group_output.shape == (1, 16, 128)


def test_nan_propagation():
    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4))
    x = torch.randn(2, 3, 64, 64)
    x[0, :, :, :] = float("nan")

    output, group_output = model(x)
    assert not torch.isnan(output[1]).any()
    assert not torch.isnan(group_output[1]).any()


def test_backprop():
    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4))
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    y = torch.randn(1, 128, 8, 8)

    output, group_output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()

    assert x.grad is not None
