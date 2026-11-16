"""SAC output heads preserve skip-connected and empty MLP trunks."""

import copy

import pytest
import torch

from rl_games.algos_torch.network_builder import SACBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build_network(units, d2rl, normalization):
    builder = SACBuilder()
    builder.load({
        "separate": True, "normalization": normalization, "log_std_bounds": [-5, 1],
        "mlp": {"units": units, "activation": "elu", "d2rl": d2rl,
                "initializer": {"name": "default"}},
    })
    return builder.build("sac", actions_num=2, input_shape=(5,), obs_dim=5, action_dim=2)


@pytest.mark.parametrize("units", [[8], [16, 8, 4]])
@pytest.mark.parametrize("normalization", [None, "layer_norm", "batch_norm"])
@pytest.mark.parametrize("d2rl", [False, True])
def test_sac_trunks_train_and_roundtrip(units, normalization, d2rl):
    torch.manual_seed(1)
    network = build_network(units, d2rl, normalization)
    observations = torch.randn(6, 5, requires_grad=True)
    distribution = network.actor(observations)
    actions = distribution.rsample()
    q1, q2 = network.critic(observations, actions)
    assert actions.shape == (6, 2)
    assert q1.shape == q2.shape == (6, 1)
    loss = q1.square().mean() + q2.square().mean() + actions.square().mean()
    loss.backward()
    assert torch.isfinite(observations.grad).all()
    for module in (network.actor, network.critic):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    state = copy.deepcopy(network.state_dict())
    restored = build_network(units, d2rl, normalization)
    restored.load_state_dict(state)
    network.eval()
    restored.eval()
    expected_actions = network.actor(observations).mean
    actual_actions = restored.actor(observations).mean
    torch.testing.assert_close(actual_actions, expected_actions)
    for expected, actual in zip(network.critic(observations, expected_actions),
                                restored.critic(observations, actual_actions)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("normalization", [None, "layer_norm", "batch_norm"])
def test_sac_without_hidden_layers_is_a_linear_actor_and_critic(normalization):
    network = build_network([], False, normalization)
    observations = torch.randn(4, 5)
    actions = network.actor(observations).rsample()
    q1, q2 = network.critic(observations, actions)
    (actions.square().sum() + q1.square().sum() + q2.square().sum()).backward()
    assert network.actor.trunk[0].in_features == 5
    assert network.critic.Q1[0].in_features == 7
    assert network.critic.Q2[0].in_features == 7


def test_sequential_trunk_checkpoint_keys_remain_compatible():
    network = build_network([16, 8, 4], False, None)
    assert set(network.actor.trunk.state_dict()) == {
        "0.weight", "0.bias", "2.weight", "2.bias",
        "4.weight", "4.bias", "6.weight", "6.bias",
    }
