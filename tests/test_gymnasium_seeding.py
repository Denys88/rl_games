"""Gymnasium vector seeds must reach the next requested reset."""

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv


def _close(env):
    if env._use_native_vec:
        env.env.close()
    else:
        for inner in env.envs:
            inner.close()


def _make(mode, count, **kwargs):
    if mode == "manual":
        kwargs["env_creator"] = lambda: gym.make("CartPole-v1")
    return GymnasiumVecEnv("CartPole-v1", count,
                           use_async=mode == "async", **kwargs)


@pytest.mark.parametrize("mode", ["sync", "manual"])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_seed_method_replays_next_reset(mode, count, seed):
    env = _make(mode, count)
    try:
        env.seed(seed)
        first = env.reset().copy()
        env.step(np.zeros(count, dtype=np.int64))
        env.seed(seed)
        replay = env.reset()
        np.testing.assert_array_equal(replay, first)
        if count > 1:
            assert not np.array_equal(first[0], first[1])
    finally:
        _close(env)


def test_async_seed_method_replays_next_reset():
    env = _make("async", 2)
    try:
        env.seed(19)
        first = env.reset().copy()
        env.step(np.zeros(2, dtype=np.int64))
        env.seed(19)
        np.testing.assert_array_equal(env.reset(), first)
    finally:
        _close(env)


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_manual_constructor_seed_matches_individual_envs(count, seed):
    env = _make("manual", count, seed=seed)
    references = [gym.make("CartPole-v1") for _ in range(count)]
    try:
        expected = np.stack([inner.reset(seed=seed + i)[0]
                             for i, inner in enumerate(references)])
        np.testing.assert_array_equal(env.reset(), expected)
    finally:
        _close(env)
        for inner in references:
            inner.close()


def test_pending_seed_is_consumed_once():
    class RecordingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            self.seeds = []

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.seeds.append(seed)
            return self.np_random.uniform(-1, 1, 1).astype(np.float32), {}

    env = GymnasiumVecEnv("recording", 2, env_creator=RecordingEnv, seed=11)
    try:
        env.reset()
        env.reset()
        env.seed(29)
        env.reset()
        env.reset()
        assert [inner.seeds for inner in env.envs] == [
            [11, None, 29, None], [12, None, 30, None]]
    finally:
        _close(env)
