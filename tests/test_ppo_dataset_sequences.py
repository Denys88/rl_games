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
