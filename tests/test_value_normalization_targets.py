"""Value clipping must compare predictions and targets in one coordinate system."""

import copy
from types import SimpleNamespace

import pytest
import torch

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.a2c_common import ContinuousA2CBase, DiscreteA2CBase
from rl_games.common.common_losses import default_critic_loss
from rl_games.common.datasets import PPODataset


@pytest.mark.parametrize('base', [ContinuousA2CBase, DiscreteA2CBase])
@pytest.mark.parametrize('masked', [False, True])
def test_old_values_and_returns_use_final_normalization_stats(base, masked):
    n = 64
    rms = RunningMeanStd((1,))
    expected_rms = copy.deepcopy(rms)
    central_batch = {}
    agent = SimpleNamespace(
        config={}, normalize_value=True, normalize_advantage=False,
        has_central_value=True, use_action_masks=False, value_mean_std=rms,
        dataset=PPODataset(n, n, False, False, 'cpu', 1),
        central_value_net=SimpleNamespace(update_dataset=central_batch.update),
    )
    values = torch.linspace(-1, 1, n).unsqueeze(-1)
    returns = values + 20.0
    batch = {
        'values': values, 'returns': returns,
        'obses': torch.zeros(n, 3), 'states': torch.zeros(n, 5),
        'actions': torch.zeros(n, 2), 'neglogpacs': torch.zeros(n),
        'dones': torch.zeros(n), 'mus': torch.zeros(n, 2),
        'sigmas': torch.ones(n, 2),
    }
    if masked:
        batch['rnn_masks'] = torch.ones(n)

    # Preserve the existing two updates, but use their final statistics for
    # both tensors. The return mean shift makes a stale transform observable.
    expected_rms.train()
    expected_rms(values)
    expected_rms(returns)
    expected_rms.eval()
    base.prepare_dataset(agent, batch)
    assert rms.count.item() == 1 + 2 * n
    for key, expected in expected_rms.state_dict().items():
        torch.testing.assert_close(rms.state_dict()[key], expected)

    old_normalized = expected_rms(values)
    returns_normalized = expected_rms(returns)
    for dataset in (agent.dataset.values_dict, central_batch):
        torch.testing.assert_close(dataset['old_values'], old_normalized)
        torch.testing.assert_close(dataset['returns'], returns_normalized)

        # An improvement beyond the value clip should have zero gradient.
        # Using the earlier normalization for old_values misses this boundary.
        prediction = (old_normalized + 0.3).detach().requires_grad_()
        loss = default_critic_loss(
            dataset['old_values'], prediction, 0.2, dataset['returns'], True).mean()
        loss.backward()
        assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))
