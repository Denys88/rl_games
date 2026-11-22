"""The recurrent Impala LayerNorm option creates a trainable module."""

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


@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
@pytest.mark.parametrize("before_mlp", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_impala_recurrent_layer_norm_forward_backward_and_checkpoint(
        recurrent, before_mlp, permute, layer_norm):
    shape = (8, 8, 3) if permute else (3, 8, 8)
    params = {
        "model": {"name": "discrete_a2c"},
        "network": {
            "name": "resnet_actor_critic", "space": {"discrete": {}},
            "cnn": {"conv_depths": [4], "permute_input": permute},
            "mlp": {"units": [6], "activation": "relu", "initializer": {"name": "default"}},
            "rnn": {"name": recurrent, "units": 8, "layers": 1,
                    "before_mlp": before_mlp, "layer_norm": layer_norm},
        },
    }
    config = {"actions_num": 3, "input_shape": shape, "num_seqs": 2, "value_size": 1,
              "normalize_value": False, "normalize_input": False}
    model = ModelBuilder().load(params).build(dict(config))
    inputs = {"obs": torch.randn(8, *shape), "seq_length": 4,
              "rnn_states": model.get_default_rnn_state(), "is_train": True,
              "prev_actions": torch.arange(8) % 3}
    result = model(dict(inputs))
    assert result["prev_neglogp"].shape == (8,)
    assert result["values"].shape == (8, 1)
    (result["prev_neglogp"].mean() + result["values"].square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    state = copy.deepcopy(model.state_dict())
    norm_keys = {key for key in state if "layer_norm." in key}
    assert norm_keys == ({"a2c_network.layer_norm.weight", "a2c_network.layer_norm.bias"}
                         if layer_norm else set())
    if layer_norm:
        norm = model.a2c_network.layer_norm
        assert norm.normalized_shape == (8,)
        assert norm.weight.grad.abs().sum() > 0
        assert norm.bias.grad.abs().sum() > 0
    restored = ModelBuilder().load(params).build(dict(config))
    restored.load_state_dict(state)
    actual = restored(dict(inputs))
    torch.testing.assert_close(actual["prev_neglogp"], result["prev_neglogp"])
    torch.testing.assert_close(actual["values"], result["values"])
