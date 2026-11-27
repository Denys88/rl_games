"""History must survive an environment reusing its NumPy output buffers."""

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.wrappers import FrameStack, BatchedFrameStack, BatchedFrameStackWithStates


VARIANTS = [
    'vector', 'flat', 'image', 'batch_stack', 'batch_transpose', 'batch_flat',
    'state_stack', 'state_transpose', 'state_flat',
]


class ReusableFrames(gym.Env):
    def __init__(self, variant, reuse):
        self.variant = variant
        self.reuse = reuse
        self.batched = variant.startswith(('batch_', 'state_'))
        self.with_states = variant.startswith('state_')
        self.agents = 1 if variant.endswith('_flat') else 3
        shape = (2, 2, 1) if variant == 'image' else (2,)
        self.observation_space = gym.spaces.Box(-1000, 1000, shape=shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state_space = gym.spaces.Box(-1000, 1000, shape=(3,), dtype=np.float32)
        self.obs_buffer = np.empty((self.agents, 2) if self.batched else shape, dtype=np.float32)
        self.state_buffer = np.empty((self.agents, 3), dtype=np.float32)
        self.episode = -1
        self.time = 0

    def values(self, time, state=False):
        shape = self.state_buffer.shape if state else self.obs_buffer.shape
        return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                + 10 * time + 100 * self.episode + (500 if state else 0))

    def observation(self):
        self.obs_buffer[:] = self.values(self.time)
        self.state_buffer[:] = self.values(self.time, state=True)
        obs = self.obs_buffer if self.reuse else self.obs_buffer.copy()
        state = self.state_buffer if self.reuse else self.state_buffer.copy()
        return {'obs': obs, 'state': state} if self.with_states else obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.time = 0
        return self.observation(), {}

    def step(self, action):
        self.time += 1
        return self.observation(), 0.0, False, False, {}


def reference(base, step, frames, state=False):
    times = [max(0, step - frames + 1 + i) for i in range(frames)]
    sequence = [base.values(time, state=state) for time in times]
    if not base.batched:
        if base.variant == 'flat':
            return np.concatenate(sequence)
        if base.variant == 'image':
            return np.concatenate(sequence, axis=-1)
        return np.stack(sequence)
    rows = []
    for agent in range(base.agents):
        history = np.stack([frame[agent] for frame in sequence])
        if base.variant.endswith('_transpose'):
            history = history.T
        elif base.variant.endswith('_flat'):
            history = np.concatenate([frame[agent] for frame in sequence])
        rows.append(history)
    return np.stack(rows)


def check_history(variant, reuse, frames):
    base = ReusableFrames(variant, reuse)
    if base.with_states:
        wrapper = BatchedFrameStackWithStates
    elif base.batched:
        wrapper = BatchedFrameStack
    else:
        wrapper = FrameStack
    options = ({'transpose': variant.endswith('_transpose'),
                'flatten': variant.endswith('_flat')} if base.batched
               else {'flat': variant == 'flat'})
    env = wrapper(base, frames, **options)
    for _ in range(2):
        reset = env.reset()
        actual = reset[0] if isinstance(reset, tuple) else reset
        for step in range(7):
            if base.with_states:
                np.testing.assert_array_equal(actual['obs'], reference(base, step, frames))
                np.testing.assert_array_equal(actual['state'], reference(base, step, frames, True))
            else:
                np.testing.assert_array_equal(actual, reference(base, step, frames))
            if step < 6:
                actual = env.step(0)[0]
    env.close()


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('reuse', [False, True])
def test_frame_history_owns_observations(variant, reuse):
    check_history(variant, reuse, 3)


@pytest.mark.parametrize('variant', ['vector', 'batch_stack', 'state_stack'])
def test_single_frame_control(variant):
    check_history(variant, True, 1)
