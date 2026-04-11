"""Triton kernels for rl_games.

GAE (Generalized Advantage Estimation) kernel that replaces the Python
for-loop in discount_values(). Each Triton program handles one env,
scanning backwards over the horizon in-kernel (no Python loop, no
kernel launch per timestep).

Tensor layout (matches a2c_common.py):
    mb_rewards:  [horizon_length, num_envs, value_size]
    mb_values:   [horizon_length, num_envs, value_size]
    mb_dones:    [horizon_length, num_envs]
    mb_advs:     [horizon_length, num_envs, value_size]  (output)
    last_values: [num_envs, value_size]
    last_dones:  [num_envs]

For value_size=1 (the common case), the kernel processes one env per program.
For value_size>1, each program handles one (env, value_idx) pair.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gae_kernel(
    # Pointers
    rewards_ptr,
    values_ptr,
    dones_ptr,
    last_values_ptr,
    last_dones_ptr,
    advs_ptr,
    # Scalars
    gamma: tl.constexpr,
    lam: tl.constexpr,       # tau in rl_games
    horizon: tl.constexpr,
    num_envs: tl.constexpr,
    value_size: tl.constexpr,
    # Strides
    rewards_stride_t,   # stride along horizon dim
    rewards_stride_env, # stride along env dim
    values_stride_t,
    values_stride_env,
    dones_stride_t,
    dones_stride_env,
    advs_stride_t,
    advs_stride_env,
):
    """Compute GAE for one (env, value_idx) pair.

    Grid: (num_envs * value_size,)
    Each program scans backwards over horizon_length timesteps.
    """
    pid = tl.program_id(0)
    env_idx = pid // value_size
    val_idx = pid % value_size

    # Compute base offsets
    last_val_offset = env_idx * value_size + val_idx
    last_done_offset = env_idx

    # Load last values and dones (for t == horizon-1)
    last_val = tl.load(last_values_ptr + last_val_offset)
    last_done = tl.load(last_dones_ptr + last_done_offset)

    lastgaelam = 0.0

    # Reverse scan over horizon
    for t_fwd in range(horizon):
        t = horizon - 1 - t_fwd

        # Offsets for timestep t
        rv_offset = t * rewards_stride_t + env_idx * rewards_stride_env + val_idx
        d_offset = t * dones_stride_t + env_idx * dones_stride_env

        reward_t = tl.load(rewards_ptr + rv_offset)
        value_t = tl.load(values_ptr + rv_offset)
        done_t_raw = tl.load(dones_ptr + d_offset)

        # Next timestep values
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

        # Store advantage
        adv_offset = t * advs_stride_t + env_idx * advs_stride_env + val_idx
        tl.store(advs_ptr + adv_offset, lastgaelam)


def triton_gae(
    mb_rewards: torch.Tensor,
    mb_values: torch.Tensor,
    mb_dones: torch.Tensor,
    last_values: torch.Tensor,
    last_dones: torch.Tensor,
    gamma: float,
    tau: float,
) -> torch.Tensor:
    """Compute GAE advantages using Triton kernel.

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
    horizon_length, num_envs, value_size = mb_rewards.shape
    mb_advs = torch.empty_like(mb_rewards)

    # Convert dones to float if needed
    if mb_dones.dtype != torch.float32:
        mb_dones = mb_dones.float()
    if last_dones.dtype != torch.float32:
        last_dones = last_dones.float()

    grid = (num_envs * value_size,)

    _gae_kernel[grid](
        mb_rewards, mb_values, mb_dones, last_values, last_dones, mb_advs,
        gamma, tau, horizon_length, num_envs, value_size,
        # Strides for [horizon, env, value_size] tensors
        mb_rewards.stride(0), mb_rewards.stride(1),
        mb_values.stride(0), mb_values.stride(1),
        mb_dones.stride(0), mb_dones.stride(1),
        mb_advs.stride(0), mb_advs.stride(1),
    )

    return mb_advs
