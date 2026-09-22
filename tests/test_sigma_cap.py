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


def _params(**space):
    cont = dict(mu_activation='None', sigma_activation='None', mu_init={'name': 'default'},
                sigma_init={'name': 'const_initializer', 'val': 0.0}, fixed_sigma=False)
    cont.update(space)
    return {'model': {'name': 'continuous_a2c_logstd'},
            'network': {'name': 'actor_critic', 'separate': False, 'space': {'continuous': cont},
                        'mlp': {'units': [8], 'activation': 'elu', 'initializer': {'name': 'default'}}}}


BUILD_KWARGS = dict(actions_num=3, input_shape=(5,), num_seqs=1, value_size=1,
                    normalize_value=False, normalize_input=False)


def test_max_sigma_read_through_builder_and_applied_in_forward():
    from rl_games.algos_torch.model_builder import ModelBuilder
    model = ModelBuilder().load(_params(sigma_parametrization='softplus', min_sigma=0.2, max_sigma=1.0)).build(dict(BUILD_KWARGS))
    net = model.a2c_network
    assert (net.min_sigma, net.max_sigma) == (0.2, 1.0)
    with torch.no_grad():
        net.sigma.bias.fill_(500.0)  # raw head far above the cap: x/(1+x) reaches 0.998
    out = model({'is_train': False, 'prev_actions': None, 'obs': torch.zeros(2, 5)})
    assert torch.all(out['sigmas'] < 1.0) and torch.all(out['sigmas'] > 0.99)


def test_cap_below_floor_rejected_at_build_time():
    from rl_games.algos_torch.model_builder import ModelBuilder
    try:
        ModelBuilder().load(_params(sigma_parametrization='softplus', min_sigma=0.5, max_sigma=0.4)).build(dict(BUILD_KWARGS))
    except ValueError as e:
        assert 'max_sigma' in str(e)
        return
    raise AssertionError('expected ValueError for max_sigma <= min_sigma')
