"""First-layer-only normalization must preserve changing hidden widths."""

import pytest
import torch
from torch import nn

from rl_games.algos_torch.network_builder import A2CBuilder, SACBuilder


def network_parameters(units, normalization, first_only):
    return {
        "separate": False, "space": {"discrete": {}},
        "normalization": normalization,
        "mlp": {"units": units, "activation": "elu", "initializer": {"name": "default"},
                "norm_only_first_layer": first_only},
    }


def check_normalization_layers(trunk, normalization, first_only, depth):
    layers = [m for m in trunk if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d))]
    expected = 0 if normalization is None else 1 if first_only else depth
    assert len(layers) == expected


@pytest.mark.parametrize("units", [[16, 8, 4], [8, 8, 8]])
@pytest.mark.parametrize("normalization", [None, "layer_norm", "batch_norm"])
@pytest.mark.parametrize("first_only", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_ppo_network_with_first_layer_normalization(units, normalization, first_only, separate):
    params = network_parameters(units, normalization, first_only)
    params["separate"] = separate
    builder = A2CBuilder()
    builder.load(params)
    network = builder.build("a2c", actions_num=3, input_shape=(5,), num_seqs=4)
    observations = torch.randn(4, 5, requires_grad=True)
    logits, values, _ = network({"obs": observations})
    assert logits.shape == (4, 3)
    assert values.shape == (4, 1)
    (logits.square().mean() + values.square().mean()).backward()
    assert torch.isfinite(observations.grad).all()
    for parameter in network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    check_normalization_layers(network.actor_mlp, normalization, first_only, len(units))
    if separate:
        check_normalization_layers(network.critic_mlp, normalization, first_only, len(units))


@pytest.mark.parametrize("units", [[16, 8, 4], [8, 8, 8]])
@pytest.mark.parametrize("normalization", [None, "layer_norm", "batch_norm"])
@pytest.mark.parametrize("first_only", [False, True])
def test_sac_network_with_first_layer_normalization(units, normalization, first_only):
    params = network_parameters(units, normalization, first_only)
    params.update(separate=True, log_std_bounds=[-5, 1])
    builder = SACBuilder()
    builder.load(params)
    network = builder.build("sac", actions_num=2, input_shape=(5,), obs_dim=5, action_dim=2)
    observations = torch.randn(4, 5, requires_grad=True)
    distribution = network.actor(observations)
    actions = distribution.rsample()
    q1, q2 = network.critic(observations, actions)
    assert actions.shape == (4, 2)
    assert q1.shape == q2.shape == (4, 1)
    (actions.square().mean() + q1.square().mean() + q2.square().mean()).backward()
    assert torch.isfinite(observations.grad).all()
    for module in (network.actor, network.critic):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    for trunk in (network.actor.trunk, network.critic.Q1, network.critic.Q2):
        check_normalization_layers(trunk, normalization, first_only, len(units))
