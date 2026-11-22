"""The identity recurrent layer supports the same call contract as GRU/LSTM."""

import copy

import pytest
import torch

from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.algos_torch.torch_ext import IdentityRNN


@pytest.mark.parametrize("before_mlp", [False, True])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("seq_length", [1, 4])
@pytest.mark.parametrize("with_dones", [False, True])
def test_identity_recurrence_matches_feedforward_policy(before_mlp, separate, seq_length, with_dones):
    params = {
        "separate": separate, "space": {"discrete": {}},
        "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
    }
    reference_builder = A2CBuilder()
    reference_builder.load(copy.deepcopy(params))
    reference = reference_builder.build("a2c", actions_num=3, input_shape=(5,), num_seqs=2)
    params["rnn"] = {
        "name": "identity", "units": 5 if before_mlp else 8, "layers": 1,
        "before_mlp": before_mlp,
    }
    builder = A2CBuilder()
    builder.load(params)
    network = builder.build("a2c", actions_num=3, input_shape=(5,), num_seqs=2)
    network.load_state_dict(reference.state_dict())
    states = tuple(torch.randn_like(s) for s in network.get_default_rnn_state())
    observations = torch.randn(2 * seq_length, 5, requires_grad=True)
    dones = torch.arange(2 * seq_length) % 2 if with_dones else None
    result = network({"obs": observations, "rnn_states": states,
                      "seq_length": seq_length, "dones": dones, "bptt_len": 2})
    expected = reference({"obs": observations})
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])
    assert len(result[2]) == len(states)
    for state, initial in zip(result[2], states):
        torch.testing.assert_close(state, initial)
    actual_grad = torch.autograd.grad(result[0].sum() + result[1].sum(), observations)[0]
    expected_grad = torch.autograd.grad(expected[0].sum() + expected[1].sum(), observations)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_identity_rnn_preserves_two_argument_calls():
    layer = IdentityRNN(3, 3)
    inputs = torch.randn(4, 2, 3, requires_grad=True)
    states = torch.randn(1, 2, 1)
    output, final = layer(inputs, states)
    assert output is inputs
    assert final is states
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
