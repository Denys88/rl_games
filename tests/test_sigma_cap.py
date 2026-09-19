import math
import types

import torch

from rl_games.algos_torch.models import apply_sigma_parametrization


def _net(**kw):
    return types.SimpleNamespace(**kw)


def test_softplus_without_cap_unchanged():
    net = _net(min_sigma=0.2, sigma_parametrization='softplus', max_sigma=0.0)
    raw = torch.tensor([-1.05, 0.0, 5.0, 50.0])
    sigma, logstd = apply_sigma_parametrization(raw, net)
    assert torch.allclose(sigma, torch.nn.functional.softplus(raw) + 0.2)
    assert torch.allclose(logstd, torch.log(sigma))


def test_cap_bounds_sigma_and_keeps_small_values():
    net = _net(min_sigma=0.2, sigma_parametrization='softplus', max_sigma=1.0)
    raw = torch.tensor([-1.05, 0.0, 5.0, 50.0, 500.0])
    sigma, logstd = apply_sigma_parametrization(raw, net)
    assert bool((sigma <= 1.0).all()) and bool((sigma >= 0.2).all())
    # the Wuji init (raw -1.05 -> 0.5 uncapped) is compressed to about 0.42
    assert 0.40 < float(sigma[0]) < 0.5
    # far above the cap it saturates at the cap
    assert abs(float(sigma[-1]) - 1.0) < 5e-3
    assert torch.allclose(logstd, torch.log(sigma))


def test_cap_keeps_gradient_far_above_cap_in_fp32():
    net = _net(min_sigma=0.2, sigma_parametrization='softplus', max_sigma=1.0)
    # uncapped sigma 200: a tanh squash would have exactly zero gradient here
    raw = torch.tensor([200.0], requires_grad=True)
    sigma, _ = apply_sigma_parametrization(raw, net)
    sigma.sum().backward()
    assert raw.grad is not None and float(raw.grad) > 1e-6


def test_cap_applies_to_exp_parametrization_without_floor():
    net = _net(min_sigma=0.0, sigma_parametrization='exp', max_sigma=1.0)
    raw = torch.tensor([math.log(0.3), 4.0])
    sigma, logstd = apply_sigma_parametrization(raw, net)
    assert 0.2 < float(sigma[0]) < 0.3 and float(sigma[1]) < 1.0 + 1e-6
    assert torch.allclose(logstd, torch.log(sigma))


def test_cap_below_floor_rejected():
    net = _net(min_sigma=0.5, sigma_parametrization='softplus', max_sigma=0.4)
    try:
        apply_sigma_parametrization(torch.zeros(2), net)
    except ValueError:
        return
    raise AssertionError('expected ValueError for max_sigma <= min_sigma')
