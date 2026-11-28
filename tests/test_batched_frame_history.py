"""Flattening frame history must keep every agent's observations together."""

from collections import deque

import gymnasium as gym
import numpy as np
import pytest

from rl_games.common.wrappers import BatchedFrameStack, BatchedFrameStackWithStates


class AgentFrames(gym.Env):
    def __init__(self, agents, with_states):
        self.agents = agents
        self.with_states = with_states
        self.observation_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self.state_space = gym.spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.time = 0

    def values(self, width):
        values = 100 * self.time + 10 * np.arange(self.agents)[:, None] + np.arange(width)
        return values.astype(np.float32) / 1000

    def observation(self):
        obs = self.values(2)
        return {'obs': obs, 'state': self.values(3)} if self.with_states else obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        return self.observation(), {}

    def step(self, action):
        self.time += 1
        return self.observation(), 0.0, False, False, {}


def expected_history(history, agents, mode):
    # Build each agent's temporal sequence independently of the wrapper's layout.
    rows = []
    for agent in range(agents):
        sequence = np.array([frame[agent] for frame in history])
        if mode == 'flatten':
            sequence = np.concatenate([frame[agent] for frame in history])
        elif mode == 'transpose':
            sequence = sequence.T
        rows.append(sequence)
    return np.array(rows)


@pytest.mark.parametrize('with_states', [False, True])
@pytest.mark.parametrize('agents', [1, 3])
@pytest.mark.parametrize('frames', [2, 4])
def test_flattened_history_keeps_agents_separate(with_states, agents, frames):
    check_history(with_states, agents, frames, 'flatten')


@pytest.mark.parametrize('with_states', [False, True])
@pytest.mark.parametrize('mode', ['stack', 'transpose'])
def test_unflattened_history_control(with_states, mode):
    check_history(with_states, 3, 2, mode)


def check_history(with_states, agents, frames, mode):
    base = AgentFrames(agents, with_states)
    wrapper_type = BatchedFrameStackWithStates if with_states else BatchedFrameStack
    env = wrapper_type(base, frames, transpose=mode == 'transpose', flatten=mode == 'flatten')
    histories = {'obs': deque([base.values(2)] * frames, maxlen=frames)}
    if with_states:
        histories['state'] = deque([base.values(3)] * frames, maxlen=frames)

    reset = env.reset()
    observation = reset[0] if isinstance(reset, tuple) else reset
    for step in range(7):
        actual = observation if with_states else {'obs': observation}
        for key, history in histories.items():
            np.testing.assert_array_equal(actual[key], expected_history(history, agents, mode))
            assert actual[key].dtype == np.float32
        if step < 6:
            observation = env.step(0)[0]
            histories['obs'].append(base.values(2))
            if with_states:
                histories['state'].append(base.values(3))
    env.close()
