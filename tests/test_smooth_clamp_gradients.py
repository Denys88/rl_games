"""Smoothed PPO clipping must retain finite gradients in its lower tail."""

import pytest
import torch

from rl_games.common.common_losses import smooth_clamp, smoothed_actor_loss
from tests.test_ppo_masking import make_ppo_agent, _rollout_batch
from tests.test_sac_correctness import SameStepFakeVecEnv


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("clip", [0.01, 0.05, 0.2])
def test_smooth_clamp_forward_and_backward_match_stable_reference(dtype, clip):
    values = torch.tensor([0.0, 0.1, 0.4, 0.8, 1.0, 1.2, 2.0],
                          dtype=dtype, requires_grad=True)
    lower, upper = 1.0 - clip, 1.0 + clip
    reference = values.detach().double().requires_grad_()
    expected = torch.sigmoid(((reference - lower) / (upper - lower) - 0.5) * 4)
    expected = expected * (upper - lower) + lower
    actual = smooth_clamp(values, lower, upper)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(values.grad).all()
    tolerance = 0.1 if dtype == torch.bfloat16 else 0.02 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(values.grad.double(), reference.grad,
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("clip", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_smoothed_actor_loss_has_finite_gradient(dtype, clip, advantage):
    old = torch.zeros(3, dtype=dtype)
    new = torch.tensor([0.0, 1.0, 4.0], dtype=dtype, requires_grad=True)
    advantages = torch.full_like(old, advantage)
    loss = smoothed_actor_loss(old, new, advantages, True, clip)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(new.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_non_ppo_actor_loss_is_unchanged(dtype):
    old = torch.zeros(3, dtype=dtype)
    new = torch.tensor([-0.5, 1.0, 2.0], dtype=dtype, requires_grad=True)
    advantages = torch.tensor([-1.0, 2.0, -0.5], dtype=dtype)
    loss = smoothed_actor_loss(old, new, advantages, False, 0.01)
    torch.testing.assert_close(loss, new * advantages)
    loss.sum().backward()
    torch.testing.assert_close(new.grad, advantages)


def test_smooth_clipping_keeps_actual_ppo_update_finite(tmp_path):
    agent, _ = make_ppo_agent(
        env_cls=SameStepFakeVecEnv, num_envs=2, horizon_length=4,
        minibatch_size=8, seq_length=1, use_smooth_clamp=True,
        e_clip=0.01, normalize_advantage=False, train_dir=str(tmp_path))
    try:
        batch = _rollout_batch(agent)
        assert "rnn_masks" not in batch
        # A finite log-ratio of -1 already overflows the old sigmoid exponent.
        batch["neglogpacs"] = batch["neglogpacs"] - 1.0
        batch["returns"] = batch["values"] + torch.tensor(
            [-2.0, -1.0, 0.5, 2.0, -2.0, -1.0, 0.5, 2.0]).unsqueeze(-1)
        agent.prepare_dataset(batch)
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]
        agent.train_actor_critic(agent.dataset[0])
        after = list(agent.model.parameters())
        assert all(torch.isfinite(parameter).all() for parameter in after)
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
    finally:
        agent.writer.close()
