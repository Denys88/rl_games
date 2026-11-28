"""Early episode endings must pool the frames that were actually observed."""

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.wrappers import MaxAndSkipEnv


class ImageSequence(gym.Env):
    def __init__(self, images, stop_at=None, truncated=False):
        self.images = images
        self.stop_at = stop_at
        self.truncated = truncated
        self.observation_space = gym.spaces.Box(0, 255, shape=(2, 2), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros((2, 2), dtype=np.uint8), {}

    def step(self, action):
        image = self.images[self.steps].copy()
        self.steps += 1
        done = self.steps == self.stop_at
        return image, 1.0, done and not self.truncated, done and self.truncated, {'step': self.steps}


def frames():
    return np.array([
        [[10, 70], [30, 50]],
        [[60, 20], [80, 40]],
        [[5, 90], [15, 25]],
        [[65, 10], [75, 35]],
    ], dtype=np.uint8)


@pytest.mark.parametrize('stop_at', [1, 2, 3, 4])
@pytest.mark.parametrize('truncated', [False, True])
@pytest.mark.parametrize('warm_buffer', [False, True])
def test_pool_final_observations_after_early_end(stop_at, truncated, warm_buffer):
    current = frames()
    prefix = np.full((4, 2, 2), 240, dtype=np.uint8) if warm_buffer else current[:0]
    base = ImageSequence(np.concatenate([prefix, current]), len(prefix) + stop_at, truncated)
    env = MaxAndSkipEnv(base, skip=4)
    env.reset()
    if warm_buffer:
        env.step(0)

    result = env.step(0)

    expected = np.maximum.reduce(current[max(0, stop_at - 2):stop_at])
    np.testing.assert_array_equal(result[0], expected)
    assert result[0].dtype == np.uint8
    assert result[1] == stop_at
    assert result[-1]['step'] == len(prefix) + stop_at
    assert base.steps == len(prefix) + stop_at


@pytest.mark.parametrize('skip', [1, 4])
def test_complete_skip_controls(skip):
    images = np.concatenate([frames(), frames()[::-1]])
    env = MaxAndSkipEnv(ImageSequence(images), skip=skip)
    env.reset()
    previous = None
    for start in range(0, len(images), skip):
        actual = env.step(0)[0]
        if previous is not None:
            np.testing.assert_array_equal(previous[0], previous[1])
        expected = np.maximum.reduce(images[start + max(0, skip - 2):start + skip])
        np.testing.assert_array_equal(actual, expected)
        previous = actual, actual.copy()


@pytest.mark.parametrize('stop_at', [1, 2, 3, 4])
def test_no_pooling_control(stop_at):
    images = frames()
    env = MaxAndSkipEnv(ImageSequence(images, stop_at, True), skip=4, use_max=False)
    env.reset()
    np.testing.assert_array_equal(env.step(0)[0], images[stop_at - 1])
