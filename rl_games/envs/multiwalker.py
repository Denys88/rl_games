import gym
import numpy as np
from pettingzoo.sisl import multiwalker_v6
import yaml
from rl_games.torch_runner import Runner
import os
from collections import deque
import rl_games.envs.connect4_network

class MultiWalker(gym.Env):
    def __init__(self, name="multiwalker",  **kwargs):
        gym.Env.__init__(self)
        self.name = name
        self.env = multiwalker_v6.parallel_env()
        self.use_central_value = kwargs.pop('central_value', False)
        self.use_prev_actions = kwargs.pop('use_prev_actions', False)
        self.add_timeouts = kwargs.pop('add_timeouts', False)
        self.action_space = self.env.action_spaces['walker_0']
        self.steps_count = 0
        obs_len = self.env.observation_spaces['walker_0'].shape[0]
        if self.use_prev_actions:
            obs_len += self.action_space.shape[0]
        self.observation_space = gym.spaces.Box(-1, 1, shape =(obs_len,))
        if self.use_central_value:
            self.state_space = gym.spaces.Box(-1, 1, shape =(obs_len*3,))

    def step(self, action):
        self.steps_count += 1
        actions = {'walker_0' : action[0], 'walker_1' : action[1], 'walker_2' : action[2],}
        obs, reward, done, info = self.env.step(actions)
        if self.use_prev_actions:
            obs = {
                k: np.concatenate([v, actions[k]]) for k,v in obs.items()
            }
        obses = np.stack([obs['walker_0'], obs['walker_1'], obs['walker_2']])
        rewards = np.stack([reward['walker_0'], reward['walker_1'], reward['walker_2']])
        dones = np.stack([done['walker_0'], done['walker_1'], done['walker_2']])
        if self.use_central_value:
            states = np.concatenate([obs['walker_0'], obs['walker_1'], obs['walker_2']])
            obses = {
                'obs' : obses,
                'state': states
            }
        return obses, rewards, dones, info

    def reset(self):
        obs = self.env.reset()
        self.steps_count = 0
        if self.use_prev_actions:
            zero_actions = np.zeros(self.action_space.shape[0])
            obs = {
                k: np.concatenate([v, zero_actions]) for k,v in obs.items()
            }
        obses = np.stack([obs['walker_0'], obs['walker_1'], obs['walker_2']])
        if self.use_central_value:
            states = np.concatenate([obs['walker_0'], obs['walker_1'], obs['walker_2']])
            obses = {
                'obs' : obses,
                'state': states
            }
        return obses

    def render(self, mode='ansi'):
        self.env.render(mode)

    def get_number_of_agents(self):
        return 3


    def has_action_mask(self):
        return False 