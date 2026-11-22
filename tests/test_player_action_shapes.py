"""Discrete evaluation keeps vector axes and multi-discrete action axes."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete
from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv


class ActionShapeEnv:
    def __init__(self, actors, heads, masked):
        self.actors = actors
        self.heads = heads
        self.masked = masked
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = (gym.spaces.Discrete(3) if heads is None else
                             gym.spaces.Tuple([gym.spaces.Discrete(n) for n in heads]))
        self.seen = []

    def get_env_info(self):
        return {"observation_space": self.observation_space,
                "action_space": self.action_space, "agents": 1}

    def reset(self):
        return np.zeros((self.actors, 4) if self.actors else (4,), dtype=np.float32)

    def step(self, actions):
        expected = (() if self.actors == 0 else (self.actors,))
        if self.heads is not None:
            expected += (len(self.heads),)
        assert actions.shape == expected
        if self.masked:
            assert np.all(actions == 1)
        self.seen.append(actions.copy())
        if self.actors == 0:
            return self.reset(), 1.0, True, {}
        return self.reset(), np.ones(self.actors, dtype=np.float32), np.ones(self.actors, dtype=bool), {}

    def has_action_mask(self):
        return self.masked

    def get_action_mask(self):
        sizes = self.heads or [3]
        mask = np.zeros((self.actors or 1, sum(sizes)), dtype=bool)
        offset = 0
        for size in sizes:
            mask[:, offset + 1] = True
            offset += size
        return mask


def player_for(env, actors, deterministic):
    multi = type(env.action_space).__name__ == "Tuple"
    space = "multi_discrete" if multi else "discrete"
    return PpoPlayerDiscrete({
        "model": {"name": "multi_discrete_a2c" if multi else "discrete_a2c"},
        "network": {
            "name": "actor_critic", "separate": False, "space": {space: {}},
            "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
        },
        "config": {
            "env_name": "action-shape-test", "env_info": env.get_env_info(), "vec_env": env,
            "device_name": "cpu", "normalize_input": False, "normalize_value": False,
            "num_actors": actors or 1,
            "player": {"games_num": actors or 1, "max_steps": 2, "print_stats": False,
                       "deterministic": deterministic},
        },
    })


@pytest.mark.parametrize("actors", [0, 1, 3])
@pytest.mark.parametrize("heads", [None, [3], [3, 5]])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_discrete_player_preserves_action_shape(actors, heads, masked, deterministic):
    env = ActionShapeEnv(actors, heads, masked)
    player = player_for(env, actors, deterministic)
    player.run()
    assert len(env.seen) == 1


@pytest.mark.parametrize("deterministic", [False, True])
def test_single_native_gymnasium_actor_accepts_player_actions(deterministic):
    env = GymnasiumVecEnv("CartPole-v1", 1, use_async=False, max_episode_steps=1)
    try:
        player_for(env, 1, deterministic).run()
    finally:
        env.env.close()
