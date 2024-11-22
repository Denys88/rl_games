from rl_games.common.ivecenv import IVecEnv
from rl_games.common.env_configurations import configurations
from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
import numpy as np
import gymnasium as gym
import random
from time import sleep
import torch

class RayWorker:
    """Wrapper around a third-party (gym for example) environment class that enables parallel training.

    The RayWorker class wraps around another environment class to enable the use of this 
    environment within an asynchronous parallel training setup

    """
    def __init__(self, config_name, config):
        """Initialise the class. Sets up the environment creator using the `rl_games.common.env_configurations.configuraitons` dict

        Args:
            config_name (:obj:`str`): Key of the environment to create.
            config: Misc. kwargs passed on to the environment creator function

        """
        self.env = configurations[config_name]['env_creator'](**config)

    def _obs_to_fp32(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        if dv.dtype == np.float64:
                            v[dk] = dv.astype(np.float32)
                else:
                    if v.dtype == np.float64:
                        obs[k] = v.astype(np.float32)
        else:
            if obs.dtype == np.float64:
                obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        """Step the environment and reset if done

        Args:
            action (type depends on env): Action to take.

        """
        next_state, reward, is_done, info = self.env.step(action)
        
        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        return next_state, reward, is_done, info

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
            
    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_to_fp32(obs)
        return obs

    def get_action_mask(self):
        return self.env.get_action_mask()

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        else:
            return 1

    def set_weights(self, weights):
        self.env.update_weights(weights)

    def can_concat_infos(self):
        if hasattr(self.env, 'concat_infos'):
            return self.env.concat_infos
        else:
            return False

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        #if isinstance(observation_space, gym.spaces.dict.Dict):
        #    observation_space = observation_space['observations']

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if hasattr(self.env, 'use_central_value'):
            info['use_global_observations'] = self.env.use_central_value
        if hasattr(self.env, 'value_size'):
            info['value_size'] = self.env.value_size
        if hasattr(self.env, 'state_space'):
            info['state_space'] = self.env.state_space
        return info

class RayWorkerGymnasium:
    def __init__(self, config_name, config):
        self.env = configurations[config_name]['env_creator'](**config)
        self.saved_seed = 0
    def _obs_to_fp32(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        if dv.dtype == np.float64:
                            v[dk] = dv.astype(np.float32)
                else:
                    if v.dtype == np.float64:
                        obs[k] = v.astype(np.float32)
        else:
            if obs.dtype == np.float64:
                obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        next_state, reward, is_terminated, is_truncated, info = self.env.step(action)
        is_done = is_terminated or is_truncated
        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        info['time_outs'] = is_truncated
        return next_state, reward, is_done, info

    def seed(self, seed):
        self.saved_seed = seed
        '''
        if hasattr(self.env, 'seed'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
        '''
            
    def render(self):
        pass

    def reset(self):
        obs, info = self.env.reset(seed=self.saved_seed) # ignoring info for now
        obs = self._obs_to_fp32(obs)
        return obs

    def get_action_mask(self):
        return self.env.get_action_mask()

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        else:
            return 1

    def set_weights(self, weights):
        self.env.update_weights(weights)

    def can_concat_infos(self):
        if hasattr(self.env, 'concat_infos'):
            return self.env.concat_infos
        else:
            return False

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        #if isinstance(observation_space, gym.spaces.dict.Dict):
        #    observation_space = observation_space['observations']

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if hasattr(self.env, 'use_central_value'):
            info['use_global_observations'] = self.env.use_central_value
        if hasattr(self.env, 'value_size'):
            info['value_size'] = self.env.value_size
        if hasattr(self.env, 'state_space'):
            info['state_space'] = self.env.state_space
        return info


class RayVecEnv(IVecEnv):
    """Main env class that manages several `rl_games.common.vecenv.Rayworker` objects for parallel training
    
    The RayVecEnv class manages a set of individual environments and wraps around the methods from RayWorker.
    Each worker is executed asynchronously.

    """
    import ray

    def __init__(self, config_name, num_actors, **kwargs):
        """Initialise the class. Sets up the config for the environment and creates individual workers to manage.

        Args:
            config_name (:obj:`str`): Key of the environment to create.
            num_actors (:obj:`int`): Number of environments (actors) to create
            **kwargs: Misc. kwargs passed on to the environment creator function within the RayWorker __init__

        """
        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)

        self.remote_worker = self.ray.remote(RayWorkerGymnasium)
        self.workers = [self.remote_worker.remote(self.config_name, kwargs) for i in range(self.num_actors)]

        if self.seed is not None:
            seeds = range(self.seed, self.seed + self.num_actors)
            seed_set = []
            for (seed, worker) in zip(seeds, self.workers):	        
                seed_set.append(worker.seed.remote(seed))
            self.ray.get(seed_set)

        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = self.ray.get(res)

        res = self.workers[0].get_env_info.remote()
        env_info = self.ray.get(res)
        res = self.workers[0].can_concat_infos.remote()
        can_concat_infos = self.ray.get(res)
        self.use_global_obs = env_info['use_global_observations']
        self.concat_infos = can_concat_infos
        self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
        self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
        if self.num_agents == 1:
            self.concat_func = np.stack
        else:
            self.concat_func = np.concatenate
    
    def step(self, actions):
        """Step all individual environments (using the created workers). 
        Returns a concatenated array of observations, rewards, done states, and infos if the env allows concatenation.
        Else returns a nested dict.

        Args:
            action (type depends on env): Action to take.

        """
        newobs, newstates, newrewards, newdones, newinfos = [], [], [], [], []
        res_obs = []
        if self.num_agents == 1:
            for (action, worker) in zip(actions, self.workers):	        
                res_obs.append(worker.step.remote(action))
        else:
            for num, worker in enumerate(self.workers):
                res_obs.append(worker.step.remote(actions[self.num_agents * num: self.num_agents * num + self.num_agents]))

        all_res = self.ray.get(res_obs)
        for res in all_res:
            cobs, crewards, cdones, cinfos = res
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        if self.concat_infos:
            newinfos = dicts_to_dict_with_arrays(newinfos, False)
        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

    def get_env_info(self):
        res = self.workers[0].get_env_info.remote()
        return self.ray.get(res)

    def set_weights(self, indices, weights):
        res = []
        for ind in indices:
            res.append(self.workers[ind].set_weights.remote(weights))
        self.ray.get(res)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = self.ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self):
        res_obs = [worker.reset.remote() for worker in self.workers]
        newobs, newstates = [],[]
        for res in res_obs:
            cobs = self.ray.get(res)
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        return ret_obs

vecenv_config = {}

def register(config_name, func):
    """Add an environment type (for example RayVecEnv) to the list of available types `rl_games.common.vecenv.vecenv_config`
    Args:
        config_name (:obj:`str`): Key of the environment to create.
        func (:obj:`func`): Function that creates the environment 

    """
    vecenv_config[config_name] = func

def create_vec_env(config_name, num_actors, **kwargs):
    vec_env_name = configurations[config_name]['vecenv_type']
    return vecenv_config[vec_env_name](config_name, num_actors, **kwargs)

register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))

from rl_games.envs.brax import BraxEnv
register('BRAX', lambda config_name, num_actors, **kwargs: BraxEnv(config_name, num_actors, **kwargs))

from rl_games.envs.envpool import Envpool
register('ENVPOOL', lambda config_name, num_actors, **kwargs: Envpool(config_name, num_actors, **kwargs))

from rl_games.envs.cule import CuleEnv
register('CULE', lambda config_name, num_actors, **kwargs: CuleEnv(config_name, num_actors, **kwargs))