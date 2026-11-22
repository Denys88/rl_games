"""Unmasked singleton rollouts must not create NaN advantages."""

from types import SimpleNamespace

import pytest
import torch

from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import ContinuousA2CBase, DiscreteA2CBase
from rl_games.common.common_losses import normalize_advantage
from rl_games.common.datasets import PPODataset
from tests.test_ppo_masking import make_ppo_agent, _rollout_batch
from tests.test_sac_correctness import SameStepFakeVecEnv


@pytest.mark.parametrize("normalizer", [
    normalize_advantage,
    lambda values: torch_ext.normalization_with_masks(values, None),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("constant", [False, True])
def test_unmasked_normalization_is_finite(normalizer, dtype, size, constant):
    values = torch.full((size,), 2.0, dtype=dtype) if constant else (
        torch.arange(size, dtype=dtype) * 2.0 - 3.0)
    actual = normalizer(values)
    expected = torch.zeros_like(values) if size == 1 else (
        (values - values.mean()) / (values.std() + 1e-8))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("base", [ContinuousA2CBase, DiscreteA2CBase])
@pytest.mark.parametrize("size", [1, 2, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_prepare_dataset_preserves_finite_advantages(base, size, masked):
    central_batch = {}
    agent = SimpleNamespace(
        config={}, normalize_value=False, normalize_advantage=True,
        normalize_rms_advantage=False, has_central_value=True,
        use_action_masks=False,
        dataset=PPODataset(size, size, False, False, "cpu", 1),
        central_value_net=SimpleNamespace(update_dataset=central_batch.update),
    )
    values = torch.zeros(size, 1)
    advantages = torch.arange(size, dtype=torch.float32) * 2.0 + 1.0
    batch = {
        "values": values, "returns": advantages.unsqueeze(-1),
        "obses": torch.zeros(size, 3), "states": torch.zeros(size, 5),
        "actions": torch.zeros(size, 2), "neglogpacs": torch.zeros(size),
        "dones": torch.zeros(size), "mus": torch.zeros(size, 2),
        "sigmas": torch.ones(size, 2),
    }
    if masked:
        batch["rnn_masks"] = torch.ones(size)
    base.prepare_dataset(agent, batch)
    expected = torch.zeros(size) if size == 1 else (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8))
    for prepared in (agent.dataset.values_dict, central_batch):
        assert torch.isfinite(prepared["advantages"]).all()
        torch.testing.assert_close(prepared["advantages"], expected)


def test_one_transition_ppo_update_keeps_parameters_finite(tmp_path):
    agent, _ = make_ppo_agent(
        env_cls=SameStepFakeVecEnv, num_envs=1, horizon_length=1,
        minibatch_size=1, seq_length=1, train_dir=str(tmp_path))
    try:
        batch = _rollout_batch(agent)
        assert "rnn_masks" not in batch
        agent.prepare_dataset(batch)
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]
        agent.train_actor_critic(agent.dataset[0])
        after = list(agent.model.parameters())
        assert all(torch.isfinite(parameter).all() for parameter in after)
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
    finally:
        agent.writer.close()
