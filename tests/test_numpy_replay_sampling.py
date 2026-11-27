"""Replay sampling accepts scalar storage values under NumPy 2."""

import numpy as np
import pytest
from gymnasium import spaces

from rl_games.common.experience import PrioritizedReplayBuffer, ReplayBuffer


@pytest.mark.parametrize('prioritized', [False, True])
@pytest.mark.parametrize('shape', [(), (2,)])
@pytest.mark.parametrize('dtype', [np.float32, np.uint8])
def test_replay_samples_preserve_transition_fields(prioritized, shape, dtype):
    observation_space = spaces.Box(0, 255, shape=shape, dtype=dtype)
    if prioritized:
        buffer = PrioritizedReplayBuffer(3, 0.6, observation_space)
    else:
        buffer = ReplayBuffer(3, observation_space)
    for index in range(5):
        buffer.add(np.full(shape, index, dtype=dtype), index, index + 0.25,
                   np.full(shape, index + 10, dtype=dtype), index % 2 == 0)

    result = buffer.sample(8, beta=0.5) if prioritized else buffer.sample(8)
    observations, actions, rewards, next_observations, dones = result[:5]

    assert observations.shape == (8,) + shape
    assert next_observations.shape == observations.shape
    assert observations.dtype == next_observations.dtype == dtype
    assert actions.dtype == np.int32
    assert dones.dtype == np.bool_
    assert set(actions).issubset({2, 3, 4})
    expected_observations = actions.reshape((8,) + (1,) * len(shape))
    expected_observations = np.broadcast_to(expected_observations, observations.shape)
    np.testing.assert_array_equal(observations, expected_observations)
    np.testing.assert_array_equal(next_observations, expected_observations + 10)
    np.testing.assert_allclose(rewards, actions + 0.25)
    np.testing.assert_array_equal(dones, actions % 2 == 0)
    observations[...] = 200
    assert np.max(buffer._obses) == 4
    if prioritized:
        weights, indices = result[5:]
        assert weights.shape == (8,)
        assert np.isfinite(weights).all()
        np.testing.assert_array_equal(actions, buffer._actions[indices])
