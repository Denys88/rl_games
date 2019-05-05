import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import networks 
import wrappers
import tr_helpers
import ray
import numpy as np

def create_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    #env = wrappers.MaxAndSkipEnv(env, skip=2)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    return env
    
a2c_configurations = {
    'CartPole-v1' : {
        'NETWORK' : networks.CartPoleA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
        'ENV_CREATOR' : lambda : gym.make('CartPole-v1')
    },
    'Acrobot-v1' : {
        'NETWORK' : networks.CartPoleA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
        'ENV_CREATOR' : lambda : gym.make('Acrobot-v1')
    },
    'LunarLander-v2' : {
        'NETWORK' : networks.CartPoleA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0/100.0),
        'ENV_CREATOR' : lambda : gym.make('LunarLander-v2')
    },
    'PongNoFrameskip-v4' : {
        'NETWORK' : networks.AtariA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
        'ENV_CREATOR' : lambda :  wrappers.make_atari_deepmind('PongNoFrameskip-v4', skip=4)
    },
    'CarRacing-v0' : {
        'NETWORK' : networks.AtariA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
        'ENV_CREATOR' : lambda :  wrappers.make_atari_deepmind('CarRacing-v0', skip=4)
    },
    'RoboschoolAnt-v1' : {
        'NETWORK' : networks.CartPoleA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0/15.0),
        'ENV_CREATOR' : lambda : gym.make('RoboschoolAnt-v1')
    },
    'SuperMarioBros-v1' : {
        'NETWORK' : networks.AtariA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0/100.0),
        'ENV_CREATOR' : lambda :  create_super_mario_env()
    },

}

class Worker:
    def __init__(self, config_name):
        self.env = a2c_configurations[config_name]['ENV_CREATOR']()
        self.obs = self.env.reset()
    
    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if is_done:
            next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs


class VecEnv:
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(Worker)
        self.workers = [self.remote_worker.remote(self.config_name) for i in range(self.num_actors)]

    def step(self, actions):
        newobs, newrewards, newdones, newinfos = [], [], [], []
        res_obs = []
        for (action, worker) in zip(actions, self.workers):
            res_obs.append(worker.step.remote(action))
        for res in res_obs:
            cobs, crewards, cdones, cinfos = ray.get(res)
            newobs.append(cobs)
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        return np.asarray(newobs), np.asarray(newrewards), np.asarray(newdones, dtype=np.bool), np.asarray(newinfos)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        return np.asarray(ray.get(obs), dtype=np.float32)

    

    