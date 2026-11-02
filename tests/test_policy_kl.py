"""The adaptive LR controller needs an unbiased Gaussian KL in every dtype."""

import pytest
import torch

from rl_games.algos_torch.torch_ext import policy_kl


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sigma', [0.2, 0.01, 0.001])
def test_identical_policies_have_zero_kl(dtype, sigma):
    mu = torch.zeros(8, 20, dtype=dtype)
    std = torch.full_like(mu, sigma)
    actual = policy_kl(mu, std, mu, std, reduce=False)
    assert actual.shape == (8,)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert policy_kl(mu, std, mu, std).item() == 0.0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('vary_sigma', [False, True])
def test_policy_kl_matches_torch_distributions(dtype, vary_sigma):
    p0_mu = torch.tensor([[0.1, -0.7, 0.0], [0.5, -0.3, 0.8]], dtype=dtype)
    p1_mu = torch.tensor([[0.2, -0.2, 0.001], [0.4, -0.8, 0.3]], dtype=dtype)
    p0_sigma = torch.tensor([[0.2, 0.7, 0.001], [0.3, 0.6, 1.0]], dtype=dtype)
    p1_sigma = (
        torch.tensor([[0.5, 0.3, 0.002], [0.8, 0.2, 0.7]], dtype=dtype)
        if vary_sigma else p0_sigma.clone()
    )
    # Compare the same quantized inputs, evaluated at least in fp32.
    ref_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    p0 = torch.distributions.Normal(p0_mu.to(ref_dtype), p0_sigma.to(ref_dtype))
    p1 = torch.distributions.Normal(p1_mu.to(ref_dtype), p1_sigma.to(ref_dtype))
    expected = torch.distributions.kl_divergence(p0, p1).sum(dim=-1)

    actual = policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=False)
    assert actual.dtype == ref_dtype
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)
    reduced = policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma)
    assert reduced.ndim == 0
    torch.testing.assert_close(reduced, expected.mean())


def test_policy_kl_preserves_orientation_and_batch_dimensions():
    p0_mu = torch.zeros(2, 4, 3)
    p1_mu = torch.ones_like(p0_mu)
    p0_sigma = torch.ones_like(p0_mu)
    p1_sigma = torch.full_like(p0_mu, 2.0)
    p0 = torch.distributions.Normal(p0_mu, p0_sigma)
    p1 = torch.distributions.Normal(p1_mu, p1_sigma)
    expected = torch.distributions.kl_divergence(p0, p1).sum(dim=-1)
    reversed_kl = torch.distributions.kl_divergence(p1, p0).sum(dim=-1)

    actual = policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=False)
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, reversed_kl)
