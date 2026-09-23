"""Masked moment estimation feeds PPO advantage normalization (rnn_masks /
next-step autoreset), the masked RunningMeanStd and the masked diagnostics.
The one-pass E[x^2] - E[x]^2 form went negative in float32 for nearly
constant inputs and NaN'd the update."""

import sys

import pytest
import torch

from rl_games.algos_torch.torch_ext import (get_mean_var_with_masks,
                                            normalization_with_masks)


def _row_mask(n, n_valid):
    m = torch.zeros(n)
    m[:n_valid] = 1.0
    return m


def test_constant_masked_values_give_zero_variance_not_negative():
    v = torch.full((16,), 0.1)
    m = _row_mask(16, 15)
    mean, var = get_mean_var_with_masks(v, m)
    assert torch.allclose(mean, torch.tensor(0.1))
    assert var.item() >= 0.0, var
    assert torch.isfinite(normalization_with_masks(v, m)).all()


@pytest.mark.parametrize('offset', [0.0, 100.0, 1000.0])
def test_masked_variance_matches_direct_unbiased_variance(offset):
    torch.manual_seed(0)
    v = offset + 0.1 * torch.randn(4096)
    n_valid = 3000
    m = _row_mask(4096, n_valid)
    mean, var = get_mean_var_with_masks(v, m)
    assert torch.allclose(mean, v[:n_valid].mean(), rtol=1e-5, atol=1e-6)
    assert torch.allclose(var, v[:n_valid].var(unbiased=True), rtol=1e-3), (var, v[:n_valid].var())


def test_masked_variance_counts_valid_elements_for_multi_column_input():
    # an (N, 1) row mask expanded over V columns must yield the pooled
    # variance over n_valid * V elements, as torch.var does on the valid rows
    torch.manual_seed(1)
    y = torch.randn(256, 2) + torch.tensor([0.0, 5.0])
    n_valid = 200
    m = _row_mask(256, n_valid).unsqueeze(1).expand_as(y)
    mean, var = get_mean_var_with_masks(y, m)
    assert torch.allclose(mean, y[:n_valid].mean(), atol=1e-5)
    assert torch.allclose(var, y[:n_valid].var(unbiased=True), rtol=1e-4)


def test_degenerate_masks_stay_finite():
    v = torch.randn(8)
    for n_valid in (0, 1, 2):
        mean, var = get_mean_var_with_masks(v, _row_mask(8, n_valid))
        assert torch.isfinite(mean) and torch.isfinite(var), (n_valid, mean, var)
        assert var.item() >= 0.0


@pytest.mark.parametrize('const', [0.7, 1.0])
def test_constant_advantages_survive_prepare_dataset_and_one_update(const):
    # a real agent on the validity-masked fixture: constant advantages made
    # the one-pass masked variance negative -> sqrt -> NaN advantages -> NaN
    # parameters after a single optimizer step
    sys.path.insert(0, __file__.rsplit('/tests/', 1)[0])
    from tests.test_ppo_masking import make_ppo_agent, _rollout_batch
    agent, _ = make_ppo_agent(seed=5)
    batch = _rollout_batch(agent)
    assert batch['rnn_masks'] is not None and (batch['rnn_masks'] == 0).any()
    batch['returns'] = batch['values'] + const
    agent.prepare_dataset(batch)
    adv = agent.dataset.values_dict['advantages']
    assert torch.isfinite(adv).all(), 'masked advantage normalization produced non-finite values'
    for i in range(len(agent.dataset)):
        agent.train_actor_critic(agent.dataset[i])
    for name, p in agent.model.state_dict().items():
        assert torch.isfinite(p).all(), f'{name} is non-finite after one update'
