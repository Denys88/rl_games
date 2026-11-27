"""Manual autoreset must retain terminal observations for bootstrapping."""

import copy
import itertools

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
from tests.test_sac_correctness import make_fake_env_sac_agent, _completed_training_buffer


class TerminalEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self, env_id=0, agents=1, layout="array", ending="truncated",
                 episode_length=None):
        self.env_id = env_id
        self.agents = agents
        self.layout = layout
        self.ending = ending
        self.episode_length = env_id + 1 if episode_length is None else episode_length
        self.use_central_value = layout == "central"
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        shape = (agents, 4) if agents > 1 else (4,)
        self.buffer = np.zeros(shape, dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        actor_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.observation_space = actor_space
        if layout == "dict":
            self.observation_space = gym.spaces.Dict({
                "obs": actor_space,
                "nested": gym.spaces.Dict({"velocity": actor_space}),
            })

    def get_number_of_agents(self):
        return self.agents

    def _observation(self):
        if self.layout == "dict":
            return {"obs": self.buffer, "nested": {"velocity": self.buffer}}
        if self.layout == "central":
            return {"obs": self.buffer, "state": self.state}
        return self.buffer

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        # Deliberately reuse storage, as many simulator wrappers do.
        self.buffer.fill(0)
        self.buffer[..., 0] = self.env_id
        self.state[:] = [self.env_id, 0]
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        self.buffer[..., 1] = self.t
        self.state[1] = self.t
        done = self.t == self.episode_length
        terminated = done and self.ending == "terminated"
        truncated = done and self.ending == "truncated"
        reward = 1.0
        if self.agents > 1:
            terminated = np.full(self.agents, terminated)
            truncated = np.full(self.agents, truncated)
            reward = np.ones(self.agents, dtype=np.float32)
        return self._observation(), reward, terminated, truncated, {}


def _manual(count, **kwargs):
    counter = itertools.count()
    return GymnasiumVecEnv("terminal", count, env_creator=lambda: TerminalEnv(
        env_id=next(counter), **kwargs))


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("agents", [1, 2])
@pytest.mark.parametrize("layout", ["array", "dict", "central"])
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_final_observation_survives_reset_with_mixed_live_rows(count, agents, layout, ending):
    env = _manual(count, agents=agents, layout=layout, ending=ending)
    env.reset()
    obs, _, dones, info = env.step(np.zeros((count * agents, 2)))
    expected_mask = np.repeat(np.arange(count) == 0, agents)
    np.testing.assert_array_equal(dones, expected_mask)
    np.testing.assert_array_equal(info["_final_observation"], expected_mask)
    final = info["final_observation"]
    actual_actor = obs if layout == "array" else obs["obs"]
    final_actor = final if layout == "array" else final["obs"]
    np.testing.assert_array_equal(final_actor[:, 1], np.ones(count * agents))
    np.testing.assert_array_equal(actual_actor[:agents, 1], np.zeros(agents))
    if layout == "dict":
        np.testing.assert_array_equal(final["nested"]["velocity"], final_actor)
    if layout == "central":
        np.testing.assert_array_equal(final["states"][:, 1], np.ones(count))
        assert obs["states"][0, 1] == 0
    saved = copy.deepcopy(final_actor)
    env.step(np.zeros((count * agents, 2)))
    np.testing.assert_array_equal(final_actor, saved)


def test_live_only_step_does_not_allocate_final_observations():
    env = _manual(2, episode_length=3)
    env.reset()
    _, _, dones, info = env.step(np.zeros((2, 2)))
    assert not dones.any()
    assert "final_observation" not in info


def test_sac_replay_bootstraps_from_actual_manual_terminal_state(tmp_path):
    def create_vec(count, unused_episode_length):
        return _manual(count, episode_length=3)

    agent, env = make_fake_env_sac_agent(
        num_envs=3, env_cls=create_vec,
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=6, value_bootstrap=True,
        train_dir=str(tmp_path))
    try:
        obses, _, _, next_obses, dones, truncated = _completed_training_buffer(agent)
        rows = truncated[:, 0].nonzero(as_tuple=True)[0]
        assert rows.numel() == 6
        torch.testing.assert_close(next_obses[rows, 1], torch.full((6,), 3.0))
        torch.testing.assert_close(obses[rows, 1], torch.full((6,), 2.0))
        torch.testing.assert_close(next_obses[rows, 0], obses[rows, 0])
        assert not dones[rows].any()
    finally:
        agent.writer.close()
        for inner in env.envs:
            inner.close()
