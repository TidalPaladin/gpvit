import pytest
import torch

from gpvit.layers import (
    GroupPropagationMLPMixer,
    GroupPropagationTransformer,
    MLPMixer,
    MLPMixerPooling,
    WindowAttention,
    WindowMLPMixer,
)


class TestMLPMixer:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward_pass(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mixer = MLPMixer(64, 32, 16, 32).to(device)
        x = torch.randn(1, 16, 64).to(device)
        output = mixer(x)
        assert output.shape == x.shape

    def test_nan_propagation(self):
        mixer = MLPMixer(64, 32, 16, 32)
        x = torch.randn(2, 16, 64)
        x[0, :, :] = float("nan")
        output = mixer(x)
        assert not torch.isnan(output[1]).any()

    def test_backprop(self):
        mixer = MLPMixer(64, 32, 16, 32)
        x = torch.randn(1, 16, 64, requires_grad=True)
        output = mixer(x)
        output.sum().backward()
        assert x.grad is not None

    def test_torch_scriptable(self):
        mixer = MLPMixer(64, 32, 16, 32)
        x = torch.randn(1, 16, 64)
        scripted_mixer = torch.jit.script(mixer)
        output = scripted_mixer(x)  # type: ignore
        assert output.shape == x.shape


class TestWindowMixer:
    @pytest.fixture(params=[WindowMLPMixer, WindowAttention])
    def layer(self, request):
        if request.param == WindowAttention:
            return WindowAttention(64, 4, grid_size=(16, 16), window_size=(4, 4))
        elif request.param == WindowMLPMixer:
            return WindowMLPMixer(64, 16, 64, grid_size=(16, 16), window_size=(4, 4))
        else:
            raise ValueError(f"Unknown layer {request.param}")

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward_pass(self, device, layer):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        layer = layer.to(device)
        x = torch.randn(1, 256, 64).to(device)
        output = layer(x)
        assert output.shape == x.shape

    def test_nan_propagation(self, layer):
        x = torch.randn(2, 256, 64)
        x[0, :, :] = float("nan")
        output = layer(x)
        assert not torch.isnan(output[1]).any()

    def test_backprop(self, layer):
        x = torch.randn(1, 256, 64, requires_grad=True)
        output = layer(x)
        output.sum().backward()
        assert x.grad is not None

    def test_torch_scriptable(self, layer):
        x = torch.randn(1, 256, 64)
        scripted_attn = torch.jit.script(layer)
        output = scripted_attn(x)  # type: ignore
        assert output.shape == x.shape


class TestGroupPropagationMLPMixer:
    @pytest.fixture
    def layer(self):
        return GroupPropagationMLPMixer(64, 4, 16, 32, 32)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward_pass(self, device, layer):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        gp = layer.to(device)
        tokens = torch.randn(1, 16, 64).to(device)
        group_tokens = torch.randn(1, 16, 64).to(device)
        output_tokens, output_group_tokens = gp(tokens, group_tokens)
        assert output_tokens.shape == tokens.shape
        assert output_group_tokens.shape == group_tokens.shape

    def test_nan_propagation(self, layer):
        gp = layer
        tokens = torch.randn(2, 16, 64)
        group_tokens = torch.randn(2, 16, 64)
        tokens[0, :, :] = float("nan")
        output_tokens, output_group_tokens = gp(tokens, group_tokens)
        assert not torch.isnan(output_tokens[1]).any()
        assert not torch.isnan(output_group_tokens[1]).any()

    def test_backprop(self, layer):
        gp = layer
        tokens = torch.randn(1, 16, 64, requires_grad=True)
        group_tokens = torch.randn(1, 16, 64, requires_grad=True)
        output_tokens, output_group_tokens = gp(tokens, group_tokens)
        (output_tokens.sum() + output_group_tokens.sum()).backward()
        assert tokens.grad is not None
        assert group_tokens.grad is not None

    def test_torch_scriptable(self, layer):
        gp = layer
        tokens = torch.randn(1, 16, 64)
        group_tokens = torch.randn(1, 16, 64)
        scripted_gp = torch.jit.script(gp)
        output_tokens, output_group_tokens = scripted_gp(tokens, group_tokens)  # type: ignore
        assert output_tokens.shape == tokens.shape
        assert output_group_tokens.shape == group_tokens.shape


class TestGroupPropagationTransformer(TestGroupPropagationMLPMixer):
    @pytest.fixture
    def layer(self):
        return GroupPropagationTransformer(64, 4, 2)


class TestMLPMixerPooling:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward_pass(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        mixer = MLPMixerPooling(64, 32, 16, 32).to(device)
        x = torch.randn(1, 16, 64).to(device)
        output = mixer(x)
        assert output.shape == (1, 1, 64)

    def test_nan_propagation(self):
        mixer = MLPMixerPooling(64, 32, 16, 32)
        x = torch.randn(2, 16, 64)
        x[0, :, :] = float("nan")
        output = mixer(x)
        assert not torch.isnan(output[1]).any()

    def test_backprop(self):
        mixer = MLPMixerPooling(64, 32, 16, 32)
        x = torch.randn(1, 16, 64, requires_grad=True)
        output = mixer(x)
        output.sum().backward()
        assert x.grad is not None

    def test_torch_scriptable(self):
        mixer = MLPMixerPooling(64, 32, 16, 32)
        x = torch.randn(1, 16, 64)
        scripted_mixer = torch.jit.script(mixer)
        output = scripted_mixer(x)  # type: ignore
        assert output.shape == (1, 1, 64)
