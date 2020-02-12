import ray
from env_configurations import configurations
import numpy as np

class IVecEnv(object):
    def step(self, actions):
        raise NotImplementedError 

    def reset(self):
        raise NotImplementedError    

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return 1


class IsaacEnv(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.env = configurations[config_name]['env_creator']()
        self.obs = self.env.reset()
    
    def step(self, action): 
        next_state, reward, is_done, info = self.env.step(action)
        #reward -= is_done * 100.0
        next_state = self.reset() 
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs


class RayWorker:
    def __init__(self, config_name):
        self.env = configurations[config_name]['env_creator']()
        self.obs = self.env.reset()
    
    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if is_done:
            next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def get_action_mask(self):
        return self.env.get_action_mask()


    def get_number_of_agents(self):
        return self.env.get_number_of_agents()


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

        return np.asarray(newobs, dtype=cobs.dtype), np.asarray(newrewards), np.asarray(newdones, dtype=np.bool), np.asarray(newinfos)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        return np.asarray(ray.get(mask), dtype=np.int32)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        return np.asarray(ray.get(obs))


class RayWorkerSelfPlay:
    def __init__(self, config_name):
        self.env = configurations[config_name]['env_creator']()
        self.obs = self.env.reset()
    
    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        if is_done:
            next_state = self.reset()
        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def set_weights(self, weights):
        self.env.update_weights(weights)

    def get_action_mask(self):
        return self.env.get_action_mask()

class RayVecEnvSelfPlay(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(RayWorkerSelfPlay)
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

        return np.asarray(newobs, dtype=cobs.dtype), np.asarray(newrewards), np.asarray(newdones, dtype=np.bool), np.asarray(newinfos)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        return np.asarray(ray.get(obs))

    def set_weights(self, indices, weights):
        res = []
        for ind in indices:
            res.append(self.workers[ind].set_weights.remote(weights))
        
        ray.get(res)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        return np.asarray(ray.get(mask), dtype=np.int32)

class RayVecSMACEnv(IVecEnv):
    def __init__(self, config_name, num_actors):
        self.config_name = config_name
        self.num_actors = num_actors
        self.remote_worker = ray.remote(RayWorker)
        self.workers = [self.remote_worker.remote(self.config_name) for i in range(self.num_actors)]
        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = ray.get(res)

    def get_number_of_agents(self):
        return self.num_agents

    def step(self, actions):
        NUM_AGENTS = self.num_agents
        newobs, newrewards, newdones, newinfos = [], [], [], []
        res_obs = []
        for num, worker in enumerate(self.workers):
            res_obs.append(worker.step.remote(actions[NUM_AGENTS*num: NUM_AGENTS*num+NUM_AGENTS]))

        for res in res_obs:
            cobs, crewards, cdones, cinfos = ray.get(res)
            newobs.append(cobs)
            newrewards.append([crewards] * NUM_AGENTS)
            newdones.append([cdones] * NUM_AGENTS)
            newinfos.append(cinfos)
        return np.concatenate(newobs, axis=0), np.concatenate(newrewards, axis=0), np.concatenate(newdones, axis=0), newinfos

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self):
        obs = [worker.reset.remote() for worker in self.workers]
        newobs = ray.get(obs)
        return np.concatenate(newobs, axis=0)


def create_vec_env(config_name, num_actors):
    if configurations[config_name]['vecenv_type'] == 'RAY':
        return RayVecEnv(config_name, num_actors)
    if configurations[config_name]['vecenv_type'] == 'RAY_SMAC':
        return RayVecSMACEnv(config_name, num_actors)
    if configurations[config_name]['vecenv_type'] == 'RAY_SELFPLAY':
        return RayVecEnvSelfPlay(config_name, num_actors)
    if configurations[config_name]['vecenv_type'] == 'ISAAC':
        return IsaacEnv(config_name, num_actors)