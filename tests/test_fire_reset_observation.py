"""Reset observations must describe the state after all startup actions."""

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.wrappers import FireResetEnv


class FireCounter(gym.Env):
    def __init__(self, ending_actions, truncated):
        self.observation_space = gym.spaces.Box(0, 1000, shape=(2,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(3)
        self.ending_actions = ending_actions
        self.truncated = truncated
        self.reset_count = 0
        self.actions = []
        self.current_observation = None

    def get_action_meanings(self):
        return ['NOOP', 'FIRE', 'RIGHT']

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self.current_observation = np.array([self.reset_count, 0])
        return self.current_observation.copy(), {}

    def step(self, action):
        self.actions.append(action)
        self.current_observation = np.array([self.reset_count, action + 1])
        done = action in self.ending_actions
        return (
            self.current_observation.copy(),
            0.0,
            done and not self.truncated,
            done and self.truncated,
            {},
        )


@pytest.mark.parametrize('ending_actions', [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize('truncated', [False, True])
def test_reset_returns_observation_after_last_startup_reset(ending_actions, truncated):
    base_env = FireCounter(ending_actions, truncated)
    env = FireResetEnv(base_env)

    result = env.reset(seed=17)
    observation = result[0] if isinstance(result, tuple) else result

    assert base_env.actions == [1, 2]
    assert base_env.reset_count == 1 + len(ending_actions)
    expected_action = 0 if 2 in ending_actions else 3
    np.testing.assert_array_equal(
        observation, [base_env.reset_count, expected_action]
    )
    np.testing.assert_array_equal(observation, base_env.current_observation)
