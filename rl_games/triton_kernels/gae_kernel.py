"""GAE (Generalized Advantage Estimation) with optional Triton acceleration.

Provides compute_gae() which automatically dispatches to a Triton kernel
when available and enabled, otherwise falls back to the PyTorch loop.

Enable Triton: export RL_GAMES_USE_TRITON=1
"""

import torch
from rl_games.triton_config import USE_TRITON


def _pytorch_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau):
    """GAE via PyTorch — reverse Python loop over horizon timesteps."""
    horizon_length = mb_rewards.shape[0]
    mb_advs = torch.zeros_like(mb_rewards)
    lastgaelam = 0

    for t in reversed(range(horizon_length)):
        if t == horizon_length - 1:
            nextnonterminal = 1.0 - last_dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[t + 1]
            nextvalues = mb_values[t + 1]
        nextnonterminal = nextnonterminal.unsqueeze(1)

        delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam
    return mb_advs


def _triton_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau):
    """GAE via Triton — single kernel launch, one program per env.

    Each Triton program scans backwards over the full horizon in-kernel,
    eliminating the per-timestep kernel launch overhead of the PyTorch loop.

    Tensor layout:
        mb_rewards:  [horizon_length, num_envs, value_size]
        mb_values:   [horizon_length, num_envs, value_size]
        mb_dones:    [horizon_length, num_envs]
        last_values: [num_envs, value_size]
        last_dones:  [num_envs]
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _gae_kernel(
        rewards_ptr, values_ptr, dones_ptr,
        last_values_ptr, last_dones_ptr, advs_ptr,
        gamma: tl.constexpr, lam: tl.constexpr,
        horizon: tl.constexpr, num_envs: tl.constexpr,
        value_size: tl.constexpr,
        rewards_stride_t, rewards_stride_env,
        values_stride_t, values_stride_env,
        dones_stride_t, dones_stride_env,
        advs_stride_t, advs_stride_env,
    ):
        pid = tl.program_id(0)
        env_idx = pid // value_size
        val_idx = pid % value_size

        last_val = tl.load(last_values_ptr + env_idx * value_size + val_idx)
        last_done = tl.load(last_dones_ptr + env_idx)
        lastgaelam = 0.0

        for t_fwd in range(horizon):
            t = horizon - 1 - t_fwd

            rv_offset = t * rewards_stride_t + env_idx * rewards_stride_env + val_idx
            d_offset = t * dones_stride_t + env_idx * dones_stride_env

            reward_t = tl.load(rewards_ptr + rv_offset)
            value_t = tl.load(values_ptr + rv_offset)

            if t == horizon - 1:
                nextvalue = last_val
                nextnonterminal = 1.0 - last_done
            else:
                next_rv_offset = (t + 1) * values_stride_t + env_idx * values_stride_env + val_idx
                next_d_offset = (t + 1) * dones_stride_t + env_idx * dones_stride_env
                nextvalue = tl.load(values_ptr + next_rv_offset)
                nextnonterminal = 1.0 - tl.load(dones_ptr + next_d_offset)

            delta = reward_t + gamma * nextvalue * nextnonterminal - value_t
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

            adv_offset = t * advs_stride_t + env_idx * advs_stride_env + val_idx
            tl.store(advs_ptr + adv_offset, lastgaelam)

    horizon_length, num_envs, value_size = mb_rewards.shape
    mb_advs = torch.empty_like(mb_rewards)

    if mb_dones.dtype != torch.float32:
        mb_dones = mb_dones.float()
    if last_dones.dtype != torch.float32:
        last_dones = last_dones.float()

    grid = (num_envs * value_size,)

    _gae_kernel[grid](
        mb_rewards, mb_values, mb_dones, last_values, last_dones, mb_advs,
        gamma, tau, horizon_length, num_envs, value_size,
        mb_rewards.stride(0), mb_rewards.stride(1),
        mb_values.stride(0), mb_values.stride(1),
        mb_dones.stride(0), mb_dones.stride(1),
        mb_advs.stride(0), mb_advs.stride(1),
    )

    return mb_advs


def compute_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau):
    """Compute GAE advantages. Uses Triton kernel when enabled and on CUDA.

    Args:
        mb_rewards:  [horizon_length, num_envs, value_size]
        mb_values:   [horizon_length, num_envs, value_size]
        mb_dones:    [horizon_length, num_envs]
        last_values: [num_envs, value_size]
        last_dones:  [num_envs]
        gamma:       discount factor
        tau:         GAE lambda

    Returns:
        mb_advs: [horizon_length, num_envs, value_size]
    """
    if USE_TRITON and mb_rewards.is_cuda:
        return _triton_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau)
    return _pytorch_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau)
