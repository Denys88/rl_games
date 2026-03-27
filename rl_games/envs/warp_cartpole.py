"""GPU-accelerated CartPole environment using NVIDIA Warp.

Implements the classic CartPole-v1 dynamics entirely on GPU using Warp kernels,
supporting thousands of parallel environments for fast RL training.

Supports both discrete (push left/right) and continuous (force magnitude) actions.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import warp as wp
except ImportError:
    raise ImportError("NVIDIA Warp is required: pip install warp-lang")


# CartPole constants
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # half pole length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # timestep
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * 2.0 * np.pi / 360.0  # ~0.2095 rad
MAX_STEPS = 500


@wp.kernel
def cartpole_step_discrete(
    states: wp.array(dtype=wp.float32, ndim=2),       # (N, 4)
    actions: wp.array(dtype=wp.int32),                  # (N,)
    next_states: wp.array(dtype=wp.float32, ndim=2),   # (N, 4)
    rewards: wp.array(dtype=wp.float32),                # (N,)
    terminated: wp.array(dtype=wp.int32),               # (N,)
    step_counts: wp.array(dtype=wp.int32),              # (N,)
    truncated: wp.array(dtype=wp.int32),                # (N,)
):
    i = wp.tid()

    x = states[i, 0]
    x_dot = states[i, 1]
    theta = states[i, 2]
    theta_dot = states[i, 3]

    # Discrete action: 0 = left (-10N), 1 = right (+10N)
    force = wp.float32(0.0)
    if actions[i] == 1:
        force = wp.float32(FORCE_MAG)
    else:
        force = wp.float32(-FORCE_MAG)

    costheta = wp.cos(theta)
    sintheta = wp.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Euler integration
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc

    next_states[i, 0] = x
    next_states[i, 1] = x_dot
    next_states[i, 2] = theta
    next_states[i, 3] = theta_dot

    step_counts[i] = step_counts[i] + 1

    # Check termination
    done = 0
    if x < -X_THRESHOLD or x > X_THRESHOLD:
        done = 1
    if theta < -THETA_THRESHOLD or theta > THETA_THRESHOLD:
        done = 1
    terminated[i] = done

    # Check truncation (max steps)
    trunc = 0
    if step_counts[i] >= MAX_STEPS and done == 0:
        trunc = 1
    truncated[i] = trunc

    # Reward: 1.0 per step (including terminal)
    rewards[i] = wp.float32(1.0)


@wp.kernel
def cartpole_step_continuous(
    states: wp.array(dtype=wp.float32, ndim=2),       # (N, 4)
    actions: wp.array(dtype=wp.float32),                # (N,)
    next_states: wp.array(dtype=wp.float32, ndim=2),   # (N, 4)
    rewards: wp.array(dtype=wp.float32),                # (N,)
    terminated: wp.array(dtype=wp.int32),               # (N,)
    step_counts: wp.array(dtype=wp.int32),              # (N,)
    truncated: wp.array(dtype=wp.int32),                # (N,)
):
    i = wp.tid()

    x = states[i, 0]
    x_dot = states[i, 1]
    theta = states[i, 2]
    theta_dot = states[i, 3]

    # Continuous action: force in [-FORCE_MAG, FORCE_MAG]
    force = wp.clamp(actions[i], -FORCE_MAG, FORCE_MAG)

    costheta = wp.cos(theta)
    sintheta = wp.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Euler integration
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc

    next_states[i, 0] = x
    next_states[i, 1] = x_dot
    next_states[i, 2] = theta
    next_states[i, 3] = theta_dot

    step_counts[i] = step_counts[i] + 1

    # Check termination
    done = 0
    if x < -X_THRESHOLD or x > X_THRESHOLD:
        done = 1
    if theta < -THETA_THRESHOLD or theta > THETA_THRESHOLD:
        done = 1
    terminated[i] = done

    # Check truncation
    trunc = 0
    if step_counts[i] >= MAX_STEPS and done == 0:
        trunc = 1
    truncated[i] = trunc

    rewards[i] = wp.float32(1.0)


@wp.kernel
def cartpole_reset(
    states: wp.array(dtype=wp.float32, ndim=2),
    step_counts: wp.array(dtype=wp.int32),
    seed: int,
):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    states[i, 0] = wp.randf(state, -0.05, 0.05)
    states[i, 1] = wp.randf(state, -0.05, 0.05)
    states[i, 2] = wp.randf(state, -0.05, 0.05)
    states[i, 3] = wp.randf(state, -0.05, 0.05)
    step_counts[i] = 0


@wp.kernel
def cartpole_auto_reset(
    states: wp.array(dtype=wp.float32, ndim=2),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    step_counts: wp.array(dtype=wp.int32),
    seed: int,
    epoch: int,
):
    i = wp.tid()
    if terminated[i] == 1 or truncated[i] == 1:
        state = wp.rand_init(seed + epoch * 100000, i)
        states[i, 0] = wp.randf(state, -0.05, 0.05)
        states[i, 1] = wp.randf(state, -0.05, 0.05)
        states[i, 2] = wp.randf(state, -0.05, 0.05)
        states[i, 3] = wp.randf(state, -0.05, 0.05)
        step_counts[i] = 0


class WarpCartPole(gym.Env):
    """GPU-accelerated CartPole using NVIDIA Warp.

    Runs N parallel environments on GPU. Compatible with rl_games via
    the WarpVecEnv adapter.

    Args:
        continuous: If True, actions are continuous forces in [-10, 10].
                   If False (default), actions are discrete (0=left, 1=right).
        count_env: Number of parallel environments.
        device: Warp device string ('cuda:0', 'cpu').
        seed: Random seed.
    """

    def __init__(self, continuous=False, count_env=1, device='cuda:0', **kwargs):
        self.continuous = continuous
        self.count_env = count_env
        self.device = device
        self.seed_value = kwargs.get('seed', 42)
        self.epoch = 0

        # Observation: [x, x_dot, theta, theta_dot]
        high = np.array([4.8, np.finfo(np.float32).max, 0.4189, np.finfo(np.float32).max],
                       dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(
                low=-FORCE_MAG, high=FORCE_MAG, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(2)

        # Allocate Warp arrays
        self.wp_device = device
        self.states = wp.zeros((count_env, 4), dtype=wp.float32, device=self.wp_device)
        self.next_states = wp.zeros((count_env, 4), dtype=wp.float32, device=self.wp_device)
        self.rewards = wp.zeros(count_env, dtype=wp.float32, device=self.wp_device)
        self.terminated = wp.zeros(count_env, dtype=wp.int32, device=self.wp_device)
        self.truncated = wp.zeros(count_env, dtype=wp.int32, device=self.wp_device)
        self.step_counts = wp.zeros(count_env, dtype=wp.int32, device=self.wp_device)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed_value = seed

        wp.launch(
            cartpole_reset,
            dim=self.count_env,
            inputs=[self.states, self.step_counts, self.seed_value],
            device=self.wp_device,
        )

        return self.states

    def step(self, actions):
        self.epoch += 1

        if self.continuous:
            if not isinstance(actions, wp.array):
                import torch
                if isinstance(actions, torch.Tensor):
                    actions = wp.from_torch(actions.float().contiguous())
                else:
                    actions = wp.array(np.asarray(actions, dtype=np.float32),
                                      device=self.wp_device)

            wp.launch(
                cartpole_step_continuous,
                dim=self.count_env,
                inputs=[self.states, actions, self.next_states,
                       self.rewards, self.terminated, self.step_counts, self.truncated],
                device=self.wp_device,
            )
        else:
            if not isinstance(actions, wp.array):
                import torch
                if isinstance(actions, torch.Tensor):
                    actions = wp.from_torch(actions.int().contiguous())
                else:
                    actions = wp.array(np.asarray(actions, dtype=np.int32),
                                      device=self.wp_device)

            wp.launch(
                cartpole_step_discrete,
                dim=self.count_env,
                inputs=[self.states, actions, self.next_states,
                       self.rewards, self.terminated, self.step_counts, self.truncated],
                device=self.wp_device,
            )

        # Auto-reset done environments
        wp.launch(
            cartpole_auto_reset,
            dim=self.count_env,
            inputs=[self.next_states, self.terminated, self.truncated,
                   self.step_counts, self.seed_value, self.epoch],
            device=self.wp_device,
        )

        # Swap state buffers
        self.states, self.next_states = self.next_states, self.states

        return (
            self.states,
            self.rewards,
            self.terminated,
            {'time_outs': self.truncated},
        )

    def get_number_of_agents(self):
        return self.count_env

    def get_env_info(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }

    def close(self):
        pass
