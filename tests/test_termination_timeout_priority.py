"""A true termination must override timeout bootstrapping."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from test_sac_correctness import make_fake_env_sac_agent, _completed_training_buffer


class EndingEnv(gym.Env):
    metadata = {}

    def __init__(self, terminated=False, truncated=False, agents=1, provide_timeout=False, ep_len=1):
        self.terminated = terminated
        self.truncated = truncated
        self.agents = agents
        self.provide_timeout = provide_timeout
        self.ep_len = ep_len
        self.t = 0
        self.observation_space = gym.spaces.Box(-100, 100, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def get_number_of_agents(self):
        return self.agents

    def observation(self):
        obs = np.array([0, self.t, 0, 0], dtype=np.float32)
        return obs if self.agents == 1 else np.repeat(obs[None], self.agents, axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation(), {}

    def step(self, action):
        self.t += 1
        at_end = self.t >= self.ep_len
        terminated = self.terminated and at_end
        truncated = self.truncated and at_end
        reward = 1.0
        if self.agents > 1:
            terminated = np.full(self.agents, terminated)
            truncated = np.full(self.agents, truncated)
            reward = np.ones(self.agents, dtype=np.float32)
        info = {"time_outs": truncated} if self.provide_timeout else {}
        return self.observation(), reward, terminated, truncated, info


ENV_ID = "RLGamesTerminationPriority-v0"
if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=EndingEnv, disable_env_checker=True)


def close_env(env):
    if env._use_native_vec:
        env.env.close()
    else:
        for inner in env.envs:
            inner.close()


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("count", [1, 3])
def test_native_timeout_flags_respect_termination(terminated, truncated, count):
    env = GymnasiumVecEnv(ENV_ID, count, use_async=False,
                          terminated=terminated, truncated=truncated)
    try:
        env.reset()
        _, _, dones, info = env.step(np.zeros((count, 2), dtype=np.float32))
        np.testing.assert_array_equal(dones, np.full(count, terminated or truncated))
        np.testing.assert_array_equal(info["time_outs"], np.full(count, truncated and not terminated))
    finally:
        close_env(env)


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("agents", [1, 2])
@pytest.mark.parametrize("provide_timeout", [False, True])
def test_manual_timeout_flags_respect_termination(terminated, truncated, agents, provide_timeout):
    env = GymnasiumVecEnv("manual", 3, env_creator=lambda: EndingEnv(
        terminated, truncated, agents, provide_timeout))
    try:
        env.reset()
        _, _, dones, info = env.step(np.zeros((3 * agents, 2), dtype=np.float32))
        np.testing.assert_array_equal(dones, np.full(3 * agents, terminated or truncated))
        np.testing.assert_array_equal(info["time_outs"],
                                      np.full(3 * agents, truncated and not terminated))
    finally:
        close_env(env)


def test_async_timeout_flags_respect_simultaneous_termination():
    # Native ids exist in forkserver/spawn workers without copying Gym's registry.
    class SimultaneousEnding(gym.Wrapper):
        def step(self, action):
            obs, reward, _, _, info = self.env.step(action)
            return obs, reward, True, True, info

    env = GymnasiumVecEnv("CartPole-v1", 2, use_async=True,
                          wrap_env=lambda env: SimultaneousEnding(env))
    try:
        env.reset()
        _, _, dones, info = env.step(np.zeros(2, dtype=np.int64))
        assert dones.all()
        assert not info["time_outs"].any()
    finally:
        close_env(env)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_sac_replay_does_not_bootstrap_true_terminal_transitions(tmp_path, terminated, truncated):
    def make_env(count, unused_episode_length):
        return GymnasiumVecEnv("manual", count, env_creator=lambda: EndingEnv(
            terminated, truncated, ep_len=3))
    agent, env = make_fake_env_sac_agent(
        num_envs=3, env_cls=make_env, num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=6, value_bootstrap=True, train_dir=str(tmp_path))
    try:
        observations, _, _, _, dones, timeouts = _completed_training_buffer(agent)
        final_rows = observations[:, 1] == 2
        assert final_rows.sum() == 6
        torch.testing.assert_close(dones[final_rows], torch.full((6, 1), terminated))
        torch.testing.assert_close(timeouts[final_rows], torch.full((6, 1), truncated and not terminated))
    finally:
        agent.writer.close()
        close_env(env)
