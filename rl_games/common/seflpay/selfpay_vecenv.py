import gym
import numpy as np
import yaml
import os
import torch
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext

import rl_games
from rl_games.torch_runner import Runner


class NetworksFolder():
    def __init__(self, path='nn/'):
        self.path = path

    def get_file_list(self):
        files = os.listdir(self.path)
        return files

    def sample_networks(self, count):
        files = self.get_file_list()

        if len(files) < count:
            sampled = random.choices(files, k=count)
        else:
            sampled = random.sample(files, count)

        return sampled

class SelfPlayEnv(vecenv.IVecEnv):
    def __init__(self, vec_env, **kwargs):
        vecenv.IVecEnv.__init__(self)
        self.vec_env = vec_env
        self.networks_rate = kwargs.pop('networks_rate', 2)
        self.current_game = 0
        self.net_path = kwargs.pop('net_path', 'nn')
        self.networks = NetworksFolder(self.net_path)

    def _get_obs(self, obs, last_actions=None):
        if self.eval_mode:
            return obs
        else:
            self.op_obs = obs[self.n_agents // 2:]
            return obs[:self.n_agents // 2]


    def step(self, action):
        if self.eval_mode:
            obs, reward, dones, info = self.env.step(action)
            obs = self._get_obs(obs, action)
            return obs, reward, dones, info

        if self.op_bot == 'random':
            op_action = self.random_step(self.op_obs)
        else:
            op_action = self.agent_step(self.op_obs)

        action = np.concatenate([action, op_action], axis=0)

        obs, reward, dones, info = self.env.step(action)
        obs = self._get_obs(obs, action)

        my_reward = reward[:self.n_agents // 2]

        if self.use_reward_diff:
            my_reward = reward[:self.n_agents // 2] - reward[self.n_agents // 2:]

        return obs, my_reward, dones, info

    def reset(self):
        if self.eval_mode:
            obs = self.env.reset()
            return self._get_obs(obs)

        if not self.agents:
            self.create_agents(self.config_path)
            for agent in self.agents:
                agent.batch_size = (self.n_agents // (2 * self.agents_num))
        [agent.reset() for agent in self.agents]
        if self.current_game % self.networks_rate == 0:
            self.update_networks()
        self.current_game += 1
        obs = self.env.reset()
        return self._get_obs(obs)

    def update_networks(self):
        net_names = self.networks.sample_networks(self.agents_num)
        print('sampling new opponent networks:', net_names)
        for agent, curr_path in zip(self.agents, net_names):
            agent.restore(self.net_path + curr_path)

    def create_agents(self, config):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)
            runner = Runner()
            config["params"]["config"]['env_info'] = self.get_env_info()
            runner.load(config)

        self.agents = []
        for _ in range(self.agents_num):
            agent = runner.create_player()
            agent.model.eval()
            if self.op_bot != 'random':
                agent.restore(self.op_bot)
            self.agents.append(agent)

    @torch.no_grad()
    def agent_step(self, obs):
        op_obs = self.agents[0].obs_to_torch(obs)
        batch_size = op_obs.size()[0]
        op_actions = []
        for i in range(self.agents_num):
            start = i * (batch_size // self.agents_num)
            end = (i + 1) * (batch_size // self.agents_num)
            opponent_action = self.agents[i].get_action(op_obs[start:end], self.is_determenistic)
            op_actions.append(opponent_action)

        op_actions = torch.cat(op_actions, axis=0)
        return self.cast_actions(op_actions)

    def cast_actions(self, actions):
        if self.numpy_env:
            actions = actions.cpu().numpy()
        return actions

    def random_step(self, obs):
        op_action = [self.env.action_space.sample() for _ in range(self.n_agents)]
        return op_action

    def get_number_of_agents(self):
        return self.vec_env.get_number_of_agents() // 2

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        info['value_size'] = 1
        info['agents'] = self.get_number_of_agents()
        if self.use_global_obs:
            info['state_space'] = self.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

    def render(self, mode='human'):
        self.vec_env.render(mode="human")

    def set_weights(self, indices, weights):
        for i in indices:
            self.agents[i % self.agents_num].set_weights(weights)

    def has_action_mask(self):
        return self.vec_env.has_action_mask()
