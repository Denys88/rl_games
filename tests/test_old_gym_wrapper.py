"""OldGymWrapper adapts either gym API generation to gymnasium tuples."""

import numpy as np
import pytest
from gymnasium import spaces

from rl_games.common.wrappers import OldGymWrapper


class _OldApiEnv:
    observation_space = spaces.Box(-1, 1, (3,), np.float32)
    action_space = spaces.Box(-1, 1, (2,), np.float32)

    def reset(self):
        return np.zeros(3, np.float32)          # bare obs

    def step(self, action):
        return np.ones(3, np.float32), 1.0, True, {'k': 1}   # 4-tuple


class _NewApiEnv:
    observation_space = spaces.Box(-1, 1, (3,), np.float32)
    action_space = spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, **kwargs):
        return np.zeros(3, np.float32), {'seeded': True}

    def step(self, action):
        return np.ones(3, np.float32), 1.0, False, True, {}   # 5-tuple


@pytest.mark.parametrize('inner', [_OldApiEnv, _NewApiEnv])
def test_emits_gymnasium_tuples(inner):
    env = OldGymWrapper(inner())
    obs, info = env.reset()
    assert obs.shape == (3,) and isinstance(info, dict)
    obs, rew, terminated, truncated, info = env.step(np.zeros(2, np.float32))
    assert obs.shape == (3,)
    assert isinstance(terminated, (bool, np.bool_)) and isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


def test_old_api_done_maps_to_terminated():
    env = OldGymWrapper(_OldApiEnv())
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
    assert terminated and not truncated
