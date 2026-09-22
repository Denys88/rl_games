"""Autocast policy heads must retain fp32 Gaussian probability arithmetic."""

import math
from types import SimpleNamespace

import pytest
import torch

from rl_games.algos_torch.models import (
    ModelA2CContinuousLogStd,
    apply_sigma_parametrization,
)


class _LowPrecisionHead(torch.nn.Module):
    """Representable outputs isolate probability error from matmul rounding."""

    sigma_parametrization = 'softplus'
    min_sigma = 0.2

    def __init__(self, dtype):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.linspace(-1, 1, 20).to(dtype))
        self.raw_sigma = torch.nn.Parameter(torch.full((20,), -10., dtype=dtype))

    def forward(self, inputs):
        n = inputs['obs'].shape[0]
        return (self.mu.expand(n, -1), self.raw_sigma.expand(n, -1),
                torch.zeros(n, 1), None)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_head_matches_same_gaussian_in_float32(dtype):
    torch.manual_seed(7)
    head = _LowPrecisionHead(dtype)
    model = ModelA2CContinuousLogStd.Network(
        head, obs_shape=(3,), normalize_value=False,
        normalize_input=False, value_size=1)
    # This is the identical distribution, with no optimizer or RMS update.
    reference = torch.distributions.Normal(
        head.mu.float(), torch.nn.functional.softplus(head.raw_sigma.float()) + 0.2)
    actions = reference.sample((128,))
    result = model({'obs': torch.zeros(128, 3), 'prev_actions': actions})
    expected_logp = reference.log_prob(actions).sum(-1)
    ratio = (expected_logp + result['prev_neglogp']).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result['entropy'], reference.entropy().sum().expand(128))
    assert result['sigmas'].dtype == torch.float32
    assert result['mus'].dtype == torch.float32

    (-result['prev_neglogp'].mean() + result['entropy'].mean() * .001).backward()
    assert torch.isfinite(head.mu.grad).all()
    assert torch.isfinite(head.raw_sigma.grad).all()
    assert head.mu.grad.abs().sum() > 0

    sampled = model({'obs': torch.zeros(128, 3), 'is_train': False})
    expected = -reference.log_prob(sampled['actions']).sum(-1)
    torch.testing.assert_close(sampled['neglogpacs'], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('parametrization', ['exp', 'softplus', 'linear'])
def test_sigma_transform_preserves_double_precision(parametrization):
    raw = torch.tensor([math.log(.2), .1], dtype=torch.float64)
    sigma, logstd = apply_sigma_parametrization(
        raw, SimpleNamespace(sigma_parametrization=parametrization, min_sigma=.2))
    assert sigma.dtype == logstd.dtype == torch.float64
    torch.testing.assert_close(sigma.log(), logstd)
