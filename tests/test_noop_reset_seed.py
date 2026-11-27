"""No-op reset randomization belongs to the environment's seeded stream."""

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.wrappers import NoopResetEnv


class NoopCounter(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 1000, shape=(1,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(1)
        self.steps = 0

    def get_action_meanings(self):
        return ['NOOP']

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0]), {}

    def step(self, action):
        assert action == 0
        self.steps += 1
        return np.array([self.steps]), 0.0, False, False, {}


@pytest.fixture(autouse=True)
def preserve_global_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize('seed', [0, 17, 42])
@pytest.mark.parametrize('global_seed', [7, 11])
def test_seeded_noops_follow_the_environment_stream(seed, global_seed):
    env = NoopResetEnv(NoopCounter(), noop_max=30)
    reference = np.random.default_rng(seed)
    expected = reference.integers(1, 31)
    np.random.seed(global_seed)

    result = env.reset(seed=seed)
    observation = result[0] if isinstance(result, tuple) else result

    np.testing.assert_array_equal(observation, [expected])
    assert env.unwrapped.steps == expected
    assert env.unwrapped.np_random.random() == reference.random()


def test_unseeded_resets_continue_the_environment_stream():
    env = NoopResetEnv(NoopCounter(), noop_max=30)
    reference = np.random.default_rng(123)
    for index in range(6):
        np.random.random(20 + index)
        env.reset(seed=123 if index == 0 else None)
        assert env.unwrapped.steps == reference.integers(1, 31)


@pytest.mark.parametrize('global_seed', [7, 11])
def test_noop_reset_does_not_advance_global_numpy_rng(global_seed):
    env = NoopResetEnv(NoopCounter(), noop_max=30)
    expected = np.random.RandomState(global_seed).random_sample(4)
    np.random.seed(global_seed)

    env.reset(seed=42)

    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize('seed', [0, 42])
def test_explicit_noop_count_does_not_draw_from_environment_rng(seed):
    env = NoopResetEnv(NoopCounter(), noop_max=30)
    env.override_num_noops = 3
    env.reset(seed=seed)
    assert env.unwrapped.steps == 3
    assert env.unwrapped.np_random.random() == np.random.default_rng(seed).random()


def test_single_noop_control():
    env = NoopResetEnv(NoopCounter(), noop_max=1)
    for seed in [0, 17, 42]:
        env.reset(seed=seed)
        assert env.unwrapped.steps == 1
