"""Priority sampling includes every occupied replay-buffer slot."""

import random

import numpy as np
import pytest
from gymnasium import spaces

from rl_games.common.experience import PrioritizedReplayBuffer


def _buffer(capacity, count, alpha=1.0):
    buffer = PrioritizedReplayBuffer(
        capacity, alpha, spaces.Box(-1, 1, shape=(1,), dtype=np.float32))
    for index in range(count):
        buffer.add(np.array([index], dtype=np.float32), index, float(index),
                   np.array([index + 1], dtype=np.float32), False)
    return buffer


@pytest.mark.parametrize('capacity,count', [(1, 1), (4, 1), (4, 3), (4, 4), (5, 5)])
def test_uniform_priorities_cover_every_occupied_slot(capacity, count):
    buffer = _buffer(capacity, count)

    indices = buffer._sample_proportional(count)

    assert indices == list(range(count))


@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_proportional_sampling_matches_all_updated_priorities(monkeypatch, alpha):
    buffer = _buffer(4, 3, alpha)
    priorities = np.array([1.0, 3.0, 6.0])
    buffer.update_priorities([0, 1, 2], priorities)
    monkeypatch.setattr(random, 'random', random.Random(31).random)

    indices = buffer._sample_proportional(1000)

    probabilities = priorities ** alpha
    probabilities /= probabilities.sum()
    actual = np.bincount(indices, minlength=3) / 1000
    np.testing.assert_allclose(actual, probabilities, atol=2 / 1000, rtol=0)


def test_proportional_sampling_retains_full_range_after_wraparound():
    buffer = _buffer(3, 5)

    indices = buffer._sample_proportional(3)

    assert indices == [0, 1, 2]
    np.testing.assert_array_equal(buffer._actions[indices], [3, 4, 2])
