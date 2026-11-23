"""Regression tests for dones handling on the recurrent training paths:
RnnWithDones with a single sequence, and the multi-agent central value dataset."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_games.algos_torch.central_value import CentralValueTrain
from rl_games.common.layers.recurrent import RnnWithDones


def _stepwise_reference(rnn, inputs, states, dones):
    # the rollout's behaviour: zero the state of a done env, then step
    outs = []
    for t in range(inputs.size(0)):
        not_done = (1.0 - dones[t]).unsqueeze(0)
        if isinstance(states, tuple):
            states = tuple(s * not_done for s in states)
        else:
            states = states * not_done
        out, states = rnn(inputs[t:t + 1], states)
        outs.append(out)
    return torch.cat(outs, dim=0), states


@pytest.mark.parametrize('rnn_cls', [nn.GRU, nn.LSTM])
@pytest.mark.parametrize('num_seqs', [1, 2, 5])
@pytest.mark.parametrize('seq_length', [1, 2, 8])
def test_rnn_with_dones_matches_stepwise_reference(rnn_cls, num_seqs, seq_length):
    torch.manual_seed(0)
    in_size, hidden = 3, 4
    rnn = rnn_cls(in_size, hidden)
    inputs = torch.randn(seq_length, num_seqs, in_size)
    # a done in the middle of the sequence and one at the very first step
    dones = torch.zeros(seq_length, num_seqs, 1)
    dones[seq_length // 2, 0, 0] = 1.0
    dones[0, num_seqs - 1, 0] = 1.0
    h0 = torch.randn(1, num_seqs, hidden)
    states = (h0, torch.randn(1, num_seqs, hidden)) if rnn_cls is nn.LSTM else h0

    out, out_states = RnnWithDones(rnn)(inputs, states, dones)
    ref_out, ref_states = _stepwise_reference(rnn, inputs, states, dones)

    assert torch.allclose(out, ref_out, atol=1e-6)
    if rnn_cls is nn.LSTM:
        for s, r in zip(out_states, ref_states):
            assert torch.allclose(s, r, atol=1e-6)
    else:
        assert torch.allclose(out_states, ref_states, atol=1e-6)


def test_rnn_with_dones_single_sequence_resets_mid_sequence():
    # Regression: with one sequence the batch dim was squeezed away and the done was
    # ignored, so the output equalled the no-reset one.
    torch.manual_seed(0)
    rnn = nn.GRU(3, 4)
    inputs = torch.randn(8, 1, 3)
    states = torch.randn(1, 1, 4)
    dones = torch.zeros(8, 1, 1)
    dones[4, 0, 0] = 1.0

    out, _ = RnnWithDones(rnn)(inputs, states, dones)
    no_reset_out, _ = rnn(inputs, states)

    assert torch.allclose(out[:4], no_reset_out[:4], atol=1e-6)
    assert not torch.allclose(out[4:], no_reset_out[4:], atol=1e-4)


def _central_value_stub(num_actors, num_agents, horizon, value_size):
    return SimpleNamespace(num_actors=num_actors, num_agents=num_agents, horizon_length=horizon,
                           value_size=value_size, batch_size=num_actors * horizon)


@pytest.mark.parametrize('num_agents', [2, 3])
@pytest.mark.parametrize('value_size', [1, 2])
def test_multiagent_central_value_rows_keep_their_own_env(num_agents, value_size):
    # Rollout tensors are env-major: row = env * num_agents + agent, each with `horizon`
    # steps. The central value has one row per env, and every tensor of that row has to
    # come from the same env.
    num_actors, horizon = 4, 6
    stub = _central_value_stub(num_actors, num_agents, horizon, value_size)
    rows = num_actors * num_agents

    env_of_row = torch.arange(rows) // num_agents
    step = torch.arange(horizon)
    # every element encodes (env, step), shared by the agents of an env
    per_row = (env_of_row[:, None] * 100 + step[None, :]).float()
    dones = per_row.reshape(-1)
    value_preds = per_row.reshape(-1, 1).repeat(1, value_size)
    returns = value_preds.clone()

    res_values, res_returns, _, res_dones = CentralValueTrain.update_multiagent_tensors(
        stub, value_preds, returns, None, dones)

    expected = (torch.arange(num_actors)[:, None] * 100 + step[None, :]).float().reshape(-1)
    assert res_dones.shape == (stub.batch_size,)
    assert torch.equal(res_dones, expected)
    assert torch.equal(res_values[:, 0], expected)
    assert torch.equal(res_returns[:, 0], expected)


def test_multiagent_central_value_dones_only_env_zero_terminates():
    # Regression: env 1 used to receive env 0's dones (row e read env e // num_agents).
    num_actors, num_agents, horizon = 4, 2, 8
    stub = _central_value_stub(num_actors, num_agents, horizon, 1)
    dones = torch.zeros(num_actors, num_agents, horizon)
    dones[0, :, 3] = 1.0
    values = torch.zeros(num_actors * num_agents * horizon, 1)

    res_dones = CentralValueTrain.update_multiagent_tensors(
        stub, values, values.clone(), None, dones.reshape(-1))[3].view(num_actors, horizon)

    assert res_dones[0, 3] == 1.0
    assert res_dones.sum() == 1.0
