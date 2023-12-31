import pytest
import torch
import torch.nn as nn

from gpvit import GPViT
from gpvit.layers import GroupPropagationMLPMixer, GroupPropagationTransformer, WindowAttention, WindowMLPMixer


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


def test_register_mask_hook():
    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4))
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    output1, group_output1 = model(x)

    def hook(module, input, output):
        return output * 0.0

    handle = model.register_mask_hook(hook)
    output2, group_output2 = model(x)

    assert (output1 != output2).any()
    handle.remove()


def test_unpatch():
    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4))
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    output, _ = model(x)

    head = nn.Linear(128, 3 * 8 * 8)
    output = head(output.movedim(1, -1)).movedim(-1, 1)

    unpatched = model.unpatch(output)
    assert unpatched.shape == (1, 3, 64, 64)


def test_torch_script():
    model = GPViT(dim=128, num_group_tokens=16, depth=12, img_size=(64, 64), patch_size=(8, 8), window_size=(4, 4))
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    # Convert the model to TorchScript
    script_model = torch.jit.script(model)

    # Verify the script model works as expected
    output_script, group_output_script = script_model(x)  # type: ignore
    output, group_output = model(x)

    assert torch.allclose(output, output_script)
    assert torch.allclose(group_output, group_output_script)


def test_gp_only():
    model = GPViT(
        dim=128,
        num_group_tokens=16,
        depth=12,
        img_size=(64, 64),
        patch_size=(8, 8),
        window_size=(4, 4),
        group_interval=1,
    )
    for child in model.children():
        assert not isinstance(child, WindowAttention)
    for block in model.blocks:
        assert isinstance(block, GroupPropagationMLPMixer)


def test_transformer():
    model = GPViT(
        dim=128,
        num_group_tokens=16,
        depth=12,
        img_size=(64, 64),
        patch_size=(8, 8),
        window_size=(4, 4),
        group_interval=1,
        group_token_mixer="transformer",
        mixer_repeats=2,
    )
    for child in model.children():
        assert not isinstance(child, WindowAttention)
    for block in model.blocks:
        assert isinstance(block, GroupPropagationTransformer)

    x = torch.randn(1, 3, 64, 64)
    output, group_output = model(x)
    assert output.shape == (1, 128, 8, 8)
    assert group_output.shape == (1, 16, 128)


def test_window_mlpmixer():
    model = GPViT(
        dim=128,
        num_group_tokens=16,
        depth=12,
        img_size=(64, 64),
        patch_size=(8, 8),
        window_size=(4, 4),
        token_mixer="mlpmixer",
    )
    for child in model.children():
        assert not isinstance(child, WindowAttention)
    for block in model.blocks:
        assert isinstance(block, (GroupPropagationMLPMixer, WindowMLPMixer))

    x = torch.randn(1, 3, 64, 64)
    output, group_output = model(x)
    assert output.shape == (1, 128, 8, 8)
    assert group_output.shape == (1, 16, 128)
