import pytest
import torch
from torch import Tensor

from gpvit.pos_enc import FourierLogspace


class TestFourierLogspace:
    @pytest.fixture
    def pos_enc(self):
        return FourierLogspace(2, 128, 10, 64, zero_one_norm=False, dropout=0.0)

    @pytest.fixture
    def input_tensor(self):
        return torch.randn(1, 16, 128, requires_grad=True)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward_pass(self, pos_enc, input_tensor, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Cuda is not available.")

        pos_enc = pos_enc.to(device)
        input_tensor = input_tensor.to(device)

        output = pos_enc.from_grid((4, 4), proto=input_tensor)
        assert output.shape == (1, 16, 128)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_backpropagation(self, pos_enc, input_tensor, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Cuda is not available.")

        pos_enc = pos_enc.to(device)
        input_tensor = input_tensor.to(device)

        output = pos_enc.from_grid((4, 4), proto=input_tensor)
        output.sum().backward()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_torch_scriptability(self, pos_enc, input_tensor, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Cuda is not available.")

        pos_enc.to(device)
        input_tensor.to(device)

        pos_enc = pos_enc.to(device)
        input_tensor = input_tensor.to(device)

        script_pos_enc = torch.jit.script(pos_enc)
        script_output = script_pos_enc.from_grid((4, 4), proto=input_tensor)  # type: ignore
        assert isinstance(script_output, Tensor)
