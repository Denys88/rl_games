import ray
from env_configurations import configurations
import numpy as np

class IVecEnv(object):
    def step(self, actions):
        raise NotImplementedError 

    def reset(self):
        raise NotImplementedError 



class IsaacEnv(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.env = configurations[config_name]['ENV_CREATOR']()
        self.obs = self.env.reset()
    
    def step(self, action): 
        _, reward, is_done, info = self.env.step(action)
        reward -= is_done * 100.0
        next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs


class RayWorker:
    def __init__(self, config_name):
        self.env = configurations[config_name]['ENV_CREATOR']()
        self.obs = self.env.reset()
    
    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if is_done:
            next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs


class RayVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(RayWorker)
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

    

def create_vec_env(config_name, num_actors):
    if configurations[config_name]['VECENV_TYPE'] == 'RAY':
        return RayVecEnv(config_name, num_actors)
    if configurations[config_name]['VECENV_TYPE'] == 'ISAAC':
        return IsaacEnv(config_name, num_actors)