import math

import pytest
import torch
import torch.nn.functional as F

from rl_games.algos_torch.layers import NoisyFactorizedLinear, NoisyLinear
from rl_games.algos_torch.network_builder import NetworkBuilder


@pytest.mark.parametrize("layer_type", [NoisyLinear, NoisyFactorizedLinear])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_shape", [(3,), (4, 3), (2, 4, 3)])
def test_zero_noise_matches_linear_for_supported_input_shapes(layer_type, bias, input_shape):
    layer = layer_type(3, 2, bias=bias)
    with torch.no_grad():
        layer.sigma_weight.zero_()
        if bias:
            layer.sigma_bias.zero_()
    inputs = torch.arange(math.prod(input_shape)).reshape(input_shape).float()

    result = layer(inputs)

    expected = F.linear(inputs, layer.weight, layer.bias)
    assert result.shape == input_shape[:-1] + (2,)
    torch.testing.assert_close(result, expected)


def test_builder_noisy_dense_trains_noise_parameters():
    layer = NetworkBuilder.BaseNetwork()._noisy_dense(3, 2)
    inputs = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])

    torch.manual_seed(123)
    first = layer(inputs)
    first.square().sum().backward()

    for parameter in (layer.sigma_weight, layer.sigma_bias):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == parameter.numel()

    second = layer(inputs)
    assert not torch.equal(first, second)
