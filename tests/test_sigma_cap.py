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
    # the Wuji init (raw -1.05 -> 0.5 uncapped) stays close to 0.5
    assert abs(float(sigma[0]) - 0.5) < 0.02
    # far above the cap it saturates at the cap
    assert abs(float(sigma[-1]) - 1.0) < 1e-4
    assert torch.allclose(logstd, torch.log(sigma))


def test_cap_keeps_gradient_above_cap():
    net = _net(min_sigma=0.2, sigma_parametrization='softplus', max_sigma=1.0)
    raw = torch.tensor([3.0], requires_grad=True)   # uncapped sigma 3.2, above the cap
    sigma, _ = apply_sigma_parametrization(raw, net)
    sigma.sum().backward()
    assert raw.grad is not None and float(raw.grad.abs()) > 0.0


def test_cap_applies_to_exp_parametrization_without_floor():
    net = _net(min_sigma=0.0, sigma_parametrization='exp', max_sigma=1.0)
    raw = torch.tensor([math.log(0.3), 4.0])
    sigma, logstd = apply_sigma_parametrization(raw, net)
    assert abs(float(sigma[0]) - 0.3) < 0.02 and float(sigma[1]) < 1.0 + 1e-6
    assert torch.allclose(logstd, torch.log(sigma))


def test_cap_below_floor_rejected():
    net = _net(min_sigma=0.5, sigma_parametrization='softplus', max_sigma=0.4)
    try:
        apply_sigma_parametrization(torch.zeros(2), net)
    except ValueError:
        return
    raise AssertionError('expected ValueError for max_sigma <= min_sigma')
