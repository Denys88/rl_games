"""Balanced evaluation averages must include every reset in one run."""

import re

import gymnasium as gym
import numpy as np
import pytest

from rl_games.algos_torch.players import PpoPlayerDiscrete


class EpisodeStatsEnv:
    def __init__(self, num_actors, gymnasium_api, with_scores):
        self.num_actors = num_actors
        self.gymnasium_api = gymnasium_api
        self.with_scores = with_scores
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode = -1
        self.steps = 0
        self.total_rewards = np.array([[2, 8], [8, 4], [20, 6]], dtype=np.float32)
        self.lengths = [1, 2, 3] if num_actors == 1 else [1, 1, 1]

    def get_env_info(self):
        return {"observation_space": self.observation_space,
                "action_space": self.action_space, "agents": 1}

    def observation(self):
        shape = (4,) if self.num_actors == 1 else (self.num_actors, 4)
        return np.zeros(shape, dtype=np.float32)

    def reset(self):
        self.episode += 1
        self.steps = 0
        obs = self.observation()
        return (obs, {}) if self.gymnasium_api else obs

    def step(self, actions):
        self.steps += 1
        length = self.lengths[self.episode]
        rewards = self.total_rewards[self.episode, :self.num_actors] / length
        done = np.full(self.num_actors, self.steps == length)
        info = {"scores": float(self.episode == 0)} if self.with_scores else {}
        if self.num_actors == 1:
            rewards, done = rewards[0], bool(done[0])
        if self.gymnasium_api:
            return self.observation(), rewards, done, np.zeros_like(done), info
        return self.observation(), rewards, done, info


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("num_actors", [1, 2])
@pytest.mark.parametrize("gymnasium_api", [False, True])
@pytest.mark.parametrize("with_scores", [False, True])
def test_evaluation_preserves_statistics_across_resets(
        balanced, num_actors, gymnasium_api, with_scores, capsys):
    env = EpisodeStatsEnv(num_actors, gymnasium_api, with_scores)
    params = {
        "model": {"name": "discrete_a2c"},
        "network": {
            "name": "actor_critic", "separate": False, "space": {"discrete": {}},
            "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
        },
        "config": {
            "env_name": "evaluation-statistics", "env_info": env.get_env_info(),
            "vec_env": env, "device_name": "cpu", "normalize_input": False,
            "normalize_value": False, "num_actors": num_actors,
            "player": {"games_num": 3 * num_actors, "print_stats": False,
                       "max_steps": 4 if num_actors == 1 else 1,
                       "balance_env_rewards": balanced},
        },
    }
    player = PpoPlayerDiscrete(params)
    player.run()
    assert env.episode == 2
    output = capsys.readouterr().out
    summary = re.search(r"av reward: (\S+) av steps: (\S+)(?: winrate: (\S+))?", output)
    assert summary, output
    expected_reward = float(env.total_rewards[:, :num_actors].mean())
    expected_steps = float(np.mean(env.lengths))
    assert float(summary[1]) == pytest.approx(expected_reward)
    assert float(summary[2]) == pytest.approx(expected_steps)
    if with_scores:
        expected_winrate = 1 / (3 if balanced else 3 * num_actors)
        assert float(summary[3]) == pytest.approx(expected_winrate)
    else:
        assert summary[3] is None
