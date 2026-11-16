"""Impala policies expose all heads of multi-discrete action spaces."""

import copy

import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build_model(recurrent, permute, last_actions=False):
    image_shape = (8, 8, 3) if permute else (3, 8, 8)
    params = {
        "model": {"name": "multi_discrete_a2c"},
        "network": {
            "name": "resnet_actor_critic", "space": {"multi_discrete": {}},
            "cnn": {"conv_depths": [4], "permute_input": permute},
            "mlp": {"units": [8], "activation": "relu", "initializer": {"name": "default"}},
        },
    }
    if recurrent:
        params["network"]["rnn"] = {
            "name": recurrent, "units": 8, "layers": 1, "before_mlp": False,
        }
    if last_actions:
        params["network"].update(require_rewards=True, require_last_actions=True)
    shape = {"observation": image_shape, "reward": (), "last_action": (2,)} if last_actions else image_shape
    model = ModelBuilder().load(params).build({
        "actions_num": [3, 5], "input_shape": shape, "num_seqs": 2,
        "value_size": 1, "normalize_value": False, "normalize_input": False,
    })
    return model, image_shape


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_impala_multi_discrete_sampling_and_training(recurrent, permute, masked):
    model, shape = build_model(recurrent, permute)
    batch = 8
    observations = torch.randn(batch, *shape)
    inputs = {"obs": observations, "seq_length": 4}
    if recurrent:
        inputs["rnn_states"] = model.get_default_rnn_state()
    if masked:
        mask = torch.zeros(batch, 8, dtype=torch.bool)
        mask[:, 1] = True
        mask[:, 6] = True  # action 3 in the second head
        inputs["action_masks"] = mask
    sampled = model(dict(inputs, is_train=False))
    assert sampled["actions"].shape == (batch, 2)
    assert [x.shape for x in sampled["logits"]] == [(batch, 3), (batch, 5)]
    if masked:
        torch.testing.assert_close(sampled["actions"], torch.tensor([[1, 3]]).expand(batch, -1))
    trained = model(dict(inputs, is_train=True, prev_actions=sampled["actions"]))
    loss = trained["prev_neglogp"].mean() + trained["values"].square().mean() - 0.01 * trained["entropy"].mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    restored, _ = build_model(recurrent, permute)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval()
    model.eval()
    expected = model(dict(inputs, is_train=True, prev_actions=sampled["actions"]))
    actual = restored(dict(inputs, is_train=True, prev_actions=sampled["actions"]))
    torch.testing.assert_close(actual["prev_neglogp"], expected["prev_neglogp"])


@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
def test_impala_multi_discrete_previous_actions_reach_recurrent_trunk(recurrent):
    model, shape = build_model(recurrent, True, last_actions=True)
    obs = {
        "observation": torch.randn(8, *shape),
        "reward": torch.randn(8),
        "last_action": torch.tensor([[1, 4]]).expand(8, -1),
    }
    result = model({"obs": obs, "is_train": False, "seq_length": 4,
                    "rnn_states": model.get_default_rnn_state()})
    assert result["actions"].shape == (8, 2)
    assert torch.isfinite(result["neglogpacs"]).all()
