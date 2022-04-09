from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np
import torch.utils.dlpack as tpack

class Envpool(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        self.batch_size = num_actors
        env_name=kwargs.pop('env_name')
        self.env = envpool.make( env_name,
                                 env_type=kwargs.pop('env_type', 'gym'),
                                 num_envs=num_actors,
                                 batch_size=self.batch_size,
                                 episodic_life=kwargs.pop('episodic_life', True),
                                 reward_clip=kwargs.pop('reward_clip', False) # thread_affinity=False,
                                )

        self.observation_space = self.env.observation_space
        self.ids = np.arange(0, num_actors)
        self.action_space = self.env.action_space
        #self.scores = np.zeros(num_actors)

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action , self.ids)
        info['time_outs'] = info['TimeLimit.truncated']
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset(self.ids)
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_envpool(**kwargs):
    return Envpool("", kwargs.pop('num_actors', 16), **kwargs)