"""Tests for population_actor_critic: N stacked MLP policies routed by a slot
one-hot in the last N observation dims."""

import numpy as np
import pytest
import torch


def _params(N=3, units=(16, 8), seed=5):
    return {'population_size': N, 'seed': seed,
            'mlp': {'units': list(units), 'activation': 'elu', 'initializer': {'name': 'default'}},
            'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                                     'mu_init': {'name': 'default'},
                                     'sigma_init': {'name': 'const_initializer', 'val': 0},
                                     'fixed_sigma': True}}}


def _build(N=3, D=6, A=2, units=(16, 8), seed=5):
    from rl_games.algos_torch.population_network import PopulationBuilder
    b = PopulationBuilder()
    b.load(_params(N, units, seed))
    return b.build('pop', actions_num=A, input_shape=(D + N,), value_size=1, num_seqs=1)


def _obs(B, D, N, slots):
    obs = torch.randn(B, D)
    onehot = torch.nn.functional.one_hot(slots, N).float()
    return torch.cat([obs, onehot], dim=1)


def _slot_mlp(net, k):
    """Standard nn.Sequential equal to slot k of the population net."""
    layers = []
    for W, b in zip(net.weights, net.biases):
        lin = torch.nn.Linear(W.shape[1], W.shape[2])
        lin.weight.data = W[k].t().clone()
        lin.bias.data = b[k].clone()
        layers += [lin, torch.nn.ELU()]
    trunk = torch.nn.Sequential(*layers)
    mu = torch.nn.Linear(net.mu_w.shape[1], net.mu_w.shape[2])
    mu.weight.data = net.mu_w[k].t().clone()
    mu.bias.data = net.mu_b[k].clone()
    val = torch.nn.Linear(net.value_w.shape[1], net.value_w.shape[2])
    val.weight.data = net.value_w[k].t().clone()
    val.bias.data = net.value_b[k].clone()
    return trunk, mu, val


def test_population_forward_matches_per_slot_mlps():
    N, D, A, B = 3, 6, 2, 40
    net = _build(N, D, A)
    slots = torch.randint(0, N, (B,))
    obs = _obs(B, D, N, slots)
    mu, logstd, value, states = net({'obs': obs})
    assert mu.shape == (B, A) and logstd.shape == (B, A) and value.shape == (B, 1) and states is None
    for k in range(N):
        trunk, mu_k, val_k = _slot_mlp(net, k)
        rows = slots == k
        h = trunk(obs[rows, :D])
        assert torch.allclose(mu[rows], mu_k(h), atol=1e-5)
        assert torch.allclose(value[rows], val_k(h), atol=1e-5)
        assert torch.allclose(logstd[rows], net.sigma[k].expand(int(rows.sum()), A))


def test_slots_are_initialised_differently():
    net = _build()
    assert not torch.allclose(net.weights[0][0], net.weights[0][1])
    net2 = _build()                                   # same seed -> same init
    assert torch.allclose(net.weights[0][1], net2.weights[0][1])


def test_gradient_isolation_between_slots():
    N, D, A, B = 3, 6, 2, 30
    net = _build(N, D, A)
    slots = torch.randint(0, N, (B,))
    obs = _obs(B, D, N, slots)
    mu, _, value, _ = net({'obs': obs})
    loss = (mu[slots == 0] ** 2).sum() + (value[slots == 0] ** 2).sum()
    loss.backward()
    for W in list(net.weights) + [net.mu_w, net.value_w]:
        assert W.grad[0].abs().sum() > 0
        assert W.grad[1].abs().sum() == 0 and W.grad[2].abs().sum() == 0


def test_forward_handles_missing_slots_and_single_row():
    N, D, A = 4, 5, 3
    net = _build(N, D, A)
    obs = _obs(1, D, N, torch.tensor([2]))         # only slot 2 present
    mu, logstd, value, _ = net({'obs': obs})
    assert mu.shape == (1, A)
    trunk, mu_k, _ = _slot_mlp(net, 2)
    assert torch.allclose(mu, mu_k(trunk(obs[:, :D])), atol=1e-5)


def test_extract_slot_round_trip_through_standard_model():
    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.algos_torch.population_network import PopulationBuilder, extract_slot
    from rl_games.algos_torch import model_builder
    N, D, A = 3, 6, 2
    units = [16, 8]
    model_builder.register_network('population_actor_critic', PopulationBuilder)
    pop_cfg = {'model': {'name': 'continuous_a2c_logstd'},
               'network': dict(_params(N, units, seed=5), name='population_actor_critic')}
    std_cfg = {'model': {'name': 'continuous_a2c_logstd'},
               'network': {'name': 'actor_critic', 'separate': False,
                           'mlp': {'units': units, 'activation': 'elu', 'd2rl': False,
                                   'initializer': {'name': 'default'}},
                           'space': pop_cfg['network']['space']}}

    def build(cfg, dim):
        return ModelBuilder().load(cfg).build(
            {'actions_num': A, 'input_shape': (dim,), 'num_seqs': 1, 'value_size': 1,
             'normalize_value': True, 'normalize_input': True}).eval()

    pop = build(pop_cfg, D + N)
    with torch.no_grad():                               # make normalisation non-trivial
        pop.running_mean_std.running_mean[:D].add_(torch.randn(D, dtype=torch.float64))
        pop.running_mean_std.running_var[:D].mul_(3.0)
        pop.running_mean_std.running_mean[D:].copy_(torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64))
        pop.running_mean_std.running_var[D:].copy_(torch.tensor([0.09, 0.05, 0.05], dtype=torch.float64))
        pop.value_mean_std.running_mean.fill_(3.0)
    std = build(std_cfg, D)
    std.load_state_dict(extract_slot(pop.state_dict(), 1, D), strict=True)
    slots = torch.full((7,), 1)
    obs = _obs(7, D, N, slots)
    with torch.no_grad():
        ref = pop({'obs': obs, 'is_train': False})
        got = std({'obs': obs[:, :D], 'is_train': False})
    assert torch.allclose(ref['mus'], got['mus'], atol=1e-5)
    assert torch.allclose(ref['values'], got['values'], atol=1e-5)
    assert torch.allclose(ref['sigmas'], got['sigmas'], atol=1e-6)
    prefixed = {'_orig_mod.' + k: v for k, v in pop.state_dict().items()}
    assert set(extract_slot(prefixed, 1, D)) == set(extract_slot(pop.state_dict(), 1, D))
