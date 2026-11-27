"""Impala recurrent policies only require enabled auxiliary observation keys."""

import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
@pytest.mark.parametrize("before_mlp", [False, True])
@pytest.mark.parametrize("permute", [False, True])
def test_impala_auxiliary_inputs_are_independently_optional(flags, recurrent, before_mlp, permute):
    rewards, actions = flags
    shape = (8, 8, 3) if permute else (3, 8, 8)
    observation = torch.randn(8, *shape)
    obs = {"observation": observation} if rewards or actions else observation
    shapes = {"observation": shape} if rewards or actions else shape
    if rewards:
        obs["reward"] = torch.randn(8)
        shapes["reward"] = ()
    if actions:
        obs["last_action"] = torch.arange(8) % 3
        shapes["last_action"] = ()
    params = {
        "model": {"name": "discrete_a2c"},
        "network": {
            "name": "resnet_actor_critic", "space": {"discrete": {}},
            "require_rewards": rewards, "require_last_actions": actions,
            "cnn": {"conv_depths": [4], "permute_input": permute},
            "mlp": {"units": [8], "activation": "relu", "initializer": {"name": "default"}},
            "rnn": {"name": recurrent, "units": 8, "layers": 1, "before_mlp": before_mlp},
        },
    }
    model = ModelBuilder().load(params).build({
        "actions_num": 3, "input_shape": shapes, "num_seqs": 2, "value_size": 1,
        "normalize_value": False, "normalize_input": False,
    })
    inputs = {"obs": obs, "seq_length": 4, "rnn_states": model.get_default_rnn_state(),
              "is_train": True, "prev_actions": torch.arange(8) % 3}
    result = model(dict(inputs))
    assert result["prev_neglogp"].shape == (8,)
    assert result["values"].shape == (8, 1)
    if rewards or actions:
        # Supplying the unused key cannot change a policy's outputs.
        full_obs = dict(obs)
        full_obs.setdefault("reward", torch.zeros(8))
        full_obs.setdefault("last_action", torch.zeros(8, dtype=torch.long))
        expected = model(dict(inputs, obs=full_obs))
        torch.testing.assert_close(result["prev_neglogp"], expected["prev_neglogp"])
        torch.testing.assert_close(result["values"], expected["values"])
    (result["prev_neglogp"].mean() + result["values"].square().mean()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
