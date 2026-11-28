"""PPO minibatches must contain complete recurrent sequences."""

import pytest
import torch

from rl_games.common.datasets import PPODataset


@pytest.mark.parametrize('batch_size,minibatch_size,sequence_length',
                         [(12, 6, 4), (12, 3, 4), (24, 8, 6)])
def test_recurrent_minibatches_reject_partial_sequences(
        batch_size, minibatch_size, sequence_length):
    with pytest.raises(ValueError, match='[Mm]inibatch size.*sequence length'):
        PPODataset(batch_size, minibatch_size, False, True, 'cpu', sequence_length)


@pytest.mark.parametrize('recurrent,minibatch_size', [(True, 4), (True, 8), (False, 6)])
def test_valid_minibatches_visit_every_transition(recurrent, minibatch_size):
    batch_size = 24
    sequence_length = 4
    dataset = PPODataset(
        batch_size, minibatch_size, False, recurrent, 'cpu', sequence_length)
    values = torch.arange(batch_size)
    state = torch.arange(batch_size // sequence_length).reshape(1, -1, 1)
    dataset.update_values_dict({'obs': {'position': values}, 'rnn_states': [state]})

    batches = list(dataset)

    assert all(batch['obs']['position'].numel() == minibatch_size for batch in batches)
    torch.testing.assert_close(
        torch.cat([batch['obs']['position'] for batch in batches]), values)
    if recurrent:
        torch.testing.assert_close(
            torch.cat([batch['rnn_states'][0] for batch in batches], dim=1), state)


@pytest.mark.parametrize('batch_size,minibatch_size', [(9, 3), (10, 5)])
@pytest.mark.parametrize('permute', [False, True])
def test_feedforward_batches_do_not_need_sequence_alignment(batch_size, minibatch_size, permute):
    dataset = PPODataset(batch_size, minibatch_size, False, False, 'cpu', 4, permute=permute)
    rows = torch.arange(batch_size)
    dataset.update_values_dict({'obs': {'row': rows}, 'returns': rows * 2})
    dataset.apply_permutation()

    batches = list(dataset)
    observed = torch.cat([batch['obs']['row'] for batch in batches])
    torch.testing.assert_close(observed.sort().values, rows)
    assert all(batch['returns'].numel() == minibatch_size for batch in batches)
    for batch in batches:
        torch.testing.assert_close(batch['returns'], batch['obs']['row'] * 2)


def test_feedforward_cartpole_trains_with_five_step_rollout(tmp_path):
    from tests.test_critical_fixes import make_cartpole_agent

    agent = make_cartpole_agent(
        horizon_length=5, minibatch_size=5, max_epochs=1, train_dir=str(tmp_path))
    try:
        assert not agent.is_rnn
        assert agent.seq_length == 4  # default for agents without an RNN too
        before = [p.detach().clone() for p in agent.model.parameters()]
        agent.train()
        after = list(agent.model.parameters())
        assert agent.epoch_num == 1
        assert all(torch.isfinite(p).all() for p in after)
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
    finally:
        agent.writer.close()
        agent.vec_env.env.close()
