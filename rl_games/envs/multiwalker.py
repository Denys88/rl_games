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
        self.action_space = self.env.action_spaces['walker_0']
        self.observation_space = self.env.observation_spaces['walker_0']

    def step(self, action):
        actions = {'walker_0' : action[0], 'walker_1' : action[1], 'walker_2' : action[2],}
        obs, reward, done, info = self.env.step(actions)
        obses = np.stack([obs['walker_0'], obs['walker_1'], obs['walker_2']])
        rewards = np.stack([reward['walker_0'], reward['walker_1'], reward['walker_2']])
        dones = np.stack([done['walker_0'], done['walker_1'], done['walker_2']])

        return obses, rewards, dones, info

    def reset(self):
        obs = self.env.reset()
        obses = np.stack([obs['walker_0'], obs['walker_1'], obs['walker_2']])
        return obses

    def render(self, mode='ansi'):
        self.env.render(mode)

    def get_number_of_agents(self):
        return 3


    def has_action_mask(self):
        return False 