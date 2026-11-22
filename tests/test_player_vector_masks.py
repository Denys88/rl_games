"""Evaluation must use the action-mask protocol exposed by vector envs."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete
from rl_games.common.ivecenv import IVecEnv


class EvaluationEnv:
    def __init__(self, multi_discrete=False, masked=True):
        self.multi_discrete = multi_discrete
        self.masked = masked
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(3), gym.spaces.Discrete(5)]) if multi_discrete else gym.spaces.Discrete(3)
        self.seen = []

    def get_env_info(self):
        return {"observation_space": self.observation_space,
                "action_space": self.action_space, "agents": 1}

    def reset(self):
        return np.zeros((2, 4), dtype=np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        expected = 1 if self.masked else 0
        assert np.all(actions == expected), (actions, expected)
        self.seen.append(actions.copy())
        return (np.zeros((2, 4), dtype=np.float32),
                np.ones(2, dtype=np.float32), np.ones(2, dtype=bool), {})

    def masks(self):
        mask = np.zeros((2, 8 if self.multi_discrete else 3), dtype=bool)
        mask[:, 1] = True
        if self.multi_discrete:
            mask[:, 4] = True
        return mask


class VectorMaskEnv(EvaluationEnv, IVecEnv):
    def has_action_masks(self):
        return True

    def get_action_masks(self):
        return self.masks()


class LegacyMaskEnv(EvaluationEnv):
    def has_action_mask(self):
        return True

    def get_action_mask(self):
        return self.masks()


class LegacyInterfaceEnv(LegacyMaskEnv, IVecEnv):
    pass


@pytest.mark.parametrize("protocol", ["vector", "legacy", "legacy_interface", "none"])
@pytest.mark.parametrize("multi_discrete", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_player_run_respects_environment_masks(protocol, multi_discrete, deterministic):
    env_type = {
        "vector": VectorMaskEnv, "legacy": LegacyMaskEnv,
        "legacy_interface": LegacyInterfaceEnv, "none": EvaluationEnv,
    }[protocol]
    env = env_type(multi_discrete, masked=protocol != "none")
    space = "multi_discrete" if multi_discrete else "discrete"
    params = {
        "model": {"name": "multi_discrete_a2c" if multi_discrete else "discrete_a2c"},
        "network": {
            "name": "actor_critic", "separate": False, "space": {space: {}},
            "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
        },
        "config": {
            "env_name": "mask-evaluation", "env_info": env.get_env_info(),
            "vec_env": env, "device_name": "cpu", "normalize_input": False,
            "normalize_value": False, "num_actors": 2,
            "player": {"games_num": 2, "max_steps": 2, "print_stats": False,
                       "deterministic": deterministic},
        },
    }
    player = PpoPlayerDiscrete(params)
    with torch.no_grad():
        for parameter in player.model.parameters():
            parameter.zero_()
        heads = player.model.a2c_network.logits
        for head in heads if multi_discrete else [heads]:
            head.bias[0] = 1000.0  # action zero is invalid in every masked case
    player.run()
    assert len(env.seen) == 1
