"""Two-hot targets project onto the actual, bounded categorical support."""

import math

import pytest
import torch

from rl_games.common.extensions.distributions import TwoHotDist
from rl_games.common.layers.action import TwoHotEncodedAction
from rl_games.common.layers.value import TwoHotEncodedValue


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_uniform_likelihood_is_normalized_at_and_beyond_support(dtype):
    targets = torch.tensor([[-100.0], [-20.0], [0.0], [20.0], [100.0]], dtype=dtype)
    logits = torch.zeros(5, 255, dtype=dtype, requires_grad=True)
    distribution = TwoHotDist(logits=logits)
    actual = distribution.log_prob(targets)
    torch.testing.assert_close(actual, torch.full_like(targets, -math.log(255)))
    (-actual.sum()).backward()
    torch.testing.assert_close(logits.grad.sum(-1), torch.zeros(5, dtype=dtype), atol=1e-6, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_exact_support_points_select_their_own_log_probabilities(dtype):
    indices = torch.tensor([0, 1, 126, 127, 128, 253, 254])
    logits = torch.linspace(-2.0, 2.0, 255, dtype=dtype).square().repeat(7, 1)
    distribution = TwoHotDist(logits=logits)
    targets = distribution.buckets[indices].unsqueeze(-1)
    expected = logits.log_softmax(-1).gather(1, indices.unsqueeze(-1))
    torch.testing.assert_close(distribution.log_prob(targets), expected)


def test_interpolation_has_the_expected_cross_entropy_gradient():
    logits = torch.tensor([[1., 2., 3., 4., 5.], [5., 4., 3., 2., 1.]],
                          dtype=torch.float64, requires_grad=True)
    distribution = TwoHotDist(logits=logits, min_space=-2.0, max_space=2.0)
    targets = torch.tensor([[-1.5], [0.25]], dtype=logits.dtype)
    weights = torch.tensor([[0.5, 0.5, 0., 0., 0.], [0., 0., 0.75, 0.25, 0.]],
                           dtype=logits.dtype)
    loss = -distribution.log_prob(targets).sum()
    expected = -(weights * logits.log_softmax(-1)).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - weights)


@pytest.mark.parametrize('buckets', [3, 32, 255])
@pytest.mark.parametrize('parameter', ['logits', 'probs'])
def test_support_matches_categorical_size_dtype_and_device(buckets, parameter):
    probabilities = torch.arange(1, buckets + 1, dtype=torch.float64).unsqueeze(0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    kwargs = {parameter: probabilities.log() if parameter == 'logits' else probabilities}
    distribution = TwoHotDist(**kwargs, min_space=2.0, max_space=5.0)
    support = torch.linspace(2.0, 5.0, buckets, dtype=probabilities.dtype)
    torch.testing.assert_close(distribution.buckets, support)
    torch.testing.assert_close(distribution.mode(), (probabilities * support).sum(-1, keepdim=True))


def test_encoded_action_uses_its_configured_support():
    model = TwoHotEncodedAction(2, 1, backets=32, min_space=2.0, max_space=5.0).double()
    torch.nn.init.zeros_(model.value_linear.weight)
    torch.nn.init.zeros_(model.value_linear.bias)
    observations = torch.ones(3, 2, dtype=torch.float64)
    torch.testing.assert_close(model(observations), torch.full((3, 1), 3.5, dtype=torch.float64))
    loss = model.loss(return_batch=torch.tensor([[2.0], [3.0], [5.0]], dtype=torch.float64)).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.value_linear.weight.grad).all()
    assert model.value_linear.weight.grad.abs().sum() > 0


def test_encoded_value_clamps_extreme_return_targets():
    model = TwoHotEncodedValue(2, 1).double()
    torch.nn.init.zeros_(model.value_linear.weight)
    torch.nn.init.zeros_(model.value_linear.bias)
    model(torch.ones(3, 2, dtype=torch.float64))
    targets = torch.tensor([[-math.expm1(30)], [0.0], [math.expm1(30)]], dtype=torch.float64)
    loss = model.loss(return_batch=targets)
    torch.testing.assert_close(loss, torch.full_like(targets, math.log(255)))
    loss.mean().backward()
    assert torch.isfinite(model.value_linear.weight.grad).all()
