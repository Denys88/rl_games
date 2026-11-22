"""Signed value transforms preserve small signals and train through zero."""

import math

import pytest
import torch

from rl_games.algos_torch.layers import SymExp, SymLog, symexp, symlog
from rl_games.common.layers.action import OneHotEncodedAction


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('transform,reference', [(symlog, math.log1p), (symexp, math.expm1)])
def test_small_signed_values_keep_their_precision(dtype, transform, reference):
    values = [-1e-8, -1e-12, 0.0, 1e-12, 1e-8]
    x = torch.tensor(values, dtype=dtype)
    expected = torch.tensor(
        [math.copysign(reference(abs(value)), value) for value in values],
        dtype=dtype,
    )
    actual = transform(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize('transform', [symlog, symexp, SymLog(), SymExp()])
def test_signed_transforms_have_unit_gradient_at_zero(transform):
    x = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    transform(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('transform', [symlog, symexp])
def test_signed_transforms_match_numerical_gradients(transform):
    x = torch.tensor([-2.0, -0.01, 0.0, 0.01, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(transform, (x,))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_signed_transforms_round_trip_small_and_large_values(dtype):
    x = torch.tensor([-1e4, -2.0, -1e-8, 0.0, 1e-8, 2.0, 1e4], dtype=dtype)
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=1e-6, atol=0)


def test_encoded_action_can_learn_from_zero_output():
    model = OneHotEncodedAction(2, 1)
    torch.nn.init.zeros_(model.value_linear.weight)
    torch.nn.init.zeros_(model.value_linear.bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observations = torch.ones(3, 2)
    loss = (model(observations) - 1.0).square().mean()
    loss.backward()
    torch.testing.assert_close(model.value_linear.weight.grad, torch.full((1, 2), -2.0))
    optimizer.step()
    assert (model(observations) - 1.0).square().mean() < loss
