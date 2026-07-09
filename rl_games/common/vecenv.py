from rl_games.common.ivecenv import IVecEnv
from rl_games.common.env_configurations import configurations
from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
import numpy as np
import random
from time import sleep
import torch


def can_concat_infos(env):
    """Return the env's concat_infos declaration.

    Checks the env itself first (so wrapper-level declarations like
    wrappers.TimeLimit.concat_infos are visible), then falls back to the
    unwrapped env.
    """
    if getattr(env, 'concat_infos', False):
        return True
    unwrapped = getattr(env, 'unwrapped', env)
    return bool(getattr(unwrapped, 'concat_infos', False))


# Episode-result keys consumed by DefaultAlgoObserver.process_infos. Its dict
# path reads infos[key][ind // num_agents] for each done index, guarded by
# `len(game_res) > ind // num_agents`, so each key must merge to a per-worker,
# index-aligned array.
_OBSERVER_RESULT_KEYS = ('scores', 'battle_won')


def _merge_ray_infos(infos_list):
    """Merge per-worker ray info dicts into a single dict.

    'time_outs': each worker carries a scalar or per-agent array; missing
    entries default to False (mirrors gymnasium_vecenv).

    'scores'/'battle_won' (emitted on episode end, consumed by
    DefaultAlgoObserver): merged to a per-worker array aligned with worker
    index, truncated after the last worker that carries the key. The observer's
    `len(game_res) > ind // num_agents` guard skips trailing workers without it.
    Interior gaps are NaN-filled but never read, since envs emit these keys only
    on done steps and the observer indexes only done workers.

    The original per-worker list is preserved under 'worker_infos' so custom
    AlgoObservers can still read keys outside the merged set.
    """
    time_outs = []
    for info in infos_list:
        to = info.get('time_outs', False) if isinstance(info, dict) else False
        if np.isscalar(to):
            time_outs.append(to)
        else:
            time_outs.extend(to)
    merged = {'time_outs': np.array(time_outs)}

    for key in _OBSERVER_RESULT_KEYS:
        last = -1
        for i, info in enumerate(infos_list):
            if isinstance(info, dict) and key in info:
                last = i
        if last < 0:
            continue
        merged[key] = np.asarray([
            info[key] if isinstance(info, dict) and key in info else np.nan
            for info in infos_list[:last + 1]
        ])
    merged['worker_infos'] = infos_list
    return merged


class RayWorker:
    """Wraps a third-party (e.g. gym) environment so it can run as a Ray actor
    for asynchronous parallel training.
    """
    def __init__(self, config_name, config):
        """Create the wrapped env via the `rl_games.common.env_configurations.configurations` dict.

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
        next_state, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated if np.isscalar(terminated) else terminated | truncated
        if 'time_outs' not in info:
            info['time_outs'] = truncated

        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        return next_state, reward, is_done, info

    def seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def render(self):
        self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        obs = self._obs_to_fp32(obs)
        return obs

    def _unwrapped(self):
        """Get the unwrapped env, bypassing gymnasium wrappers that don't forward custom attributes."""
        return getattr(self.env, 'unwrapped', self.env)

    def get_action_mask(self):
        return self._unwrapped().get_action_mask()

    def get_number_of_agents(self):
        unwrapped = self._unwrapped()
        if hasattr(unwrapped, 'get_number_of_agents'):
            return unwrapped.get_number_of_agents()
        else:
            return 1

    def set_weights(self, weights):
        self._unwrapped().update_weights(weights)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def can_concat_infos(self):
        return can_concat_infos(self.env)

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        # Use unwrapped env to access custom attributes through gymnasium wrappers
        unwrapped = self._unwrapped()
        if hasattr(unwrapped, 'use_central_value'):
            info['use_global_observations'] = unwrapped.use_central_value
        if hasattr(unwrapped, 'value_size'):
            info['value_size'] = unwrapped.value_size
        if hasattr(unwrapped, 'state_space'):
            info['state_space'] = unwrapped.state_space
        return info


class RayVecEnv(IVecEnv):
    """Manages several RayWorker actors, each running one environment
    asynchronously, and aggregates their results for parallel training.
    """
    def __init__(self, config_name, num_actors, **kwargs):
        """Initialise the class. Sets up the config for the environment and creates individual workers to manage.

        Args:
            config_name (:obj:`str`): Key of the environment to create.
            num_actors (:obj:`int`): Number of environments (actors) to create
            **kwargs: Misc. kwargs passed on to the environment creator function within the RayWorker __init__

        """
        # Import and initialize Ray only when RayVecEnv is actually used
        try:
            import ray
            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayVecEnv. Please install it with: pip install ray")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            # Get Ray initialization parameters from config
            # Example usage in YAML under params.config.env_config:
            #   ray_config:
            #     num_cpus: 8                      # Number of CPUs Ray can use
            #     num_gpus: 1                      # Number of GPUs Ray can use  
            #     object_store_memory: 2000000000  # Object store memory in bytes (2GB)
            #     local_mode: True                 # Run Ray in local mode for debugging
            #     ignore_reinit_error: True        # Suppress errors if Ray is already initialized
            #     include_dashboard: False         # Disable Ray dashboard
            #     dashboard_host: '0.0.0.0'        # Ray dashboard host
            ray_config = kwargs.pop('ray_config', {})

            # Set default object store memory if not specified
            if 'object_store_memory' not in ray_config:
                ray_config['object_store_memory'] = 1024*1024*1000

            ray.init(**ray_config)

        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)

        self.remote_worker = self.ray.remote(RayWorker)
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
        obs_space = env_info.get('observation_space')
        state_space = env_info.get('state_space')
        self.obs_type_dict = obs_space is not None and type(obs_space).__name__ == 'Dict'
        self.state_type_dict = state_space is not None and type(state_space).__name__ == 'Dict'
        if self.num_agents == 1:
            self.concat_func = np.stack
        else:
            self.concat_func = np.concatenate

    def step(self, actions):
        """Step all worker environments in parallel.

        Returns concatenated observations, rewards, dones, and infos when the env
        supports concatenation; otherwise observations come back as a nested dict.

        Args:
            actions: Actions for all workers (type depends on env).

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
        else:
            # Without concat_infos a list of per-worker dicts was returned, so
            # consumers checking `'time_outs' in infos` never saw timeouts.
            # Deliver a merged dict instead (time_outs plus the scores/battle_won
            # keys DefaultAlgoObserver tracks).
            newinfos = _merge_ray_infos(newinfos)
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
    
    def close(self):
        """Close all workers and shutdown Ray if we initialized it."""
        # Close all worker environments
        for worker in self.workers:
            self.ray.get(worker.close.remote())
        
        # Shutdown Ray if we initialized it
        if hasattr(self, 'ray') and self.ray.is_initialized():
            # Only shutdown if no other processes are using Ray
            try:
                self.ray.shutdown()
            except:
                pass  # Ray might be used by other processes

vecenv_config = {}

def register(config_name, func):
    """Add an environment type (for example RayVecEnv) to the list of available types `rl_games.common.vecenv.vecenv_config`
    Args:
        config_name (:obj:`str`): Key of the environment to create.
        func (:obj:`func`): Function that creates the environment 

    """
    vecenv_config[config_name] = func

def create_vec_env(config_name, num_actors, **kwargs):
    config = configurations[config_name]
    vec_env_name = config['vecenv_type']
    # Merge default_env_config from configuration with user kwargs
    if 'default_env_config' in config:
        merged_kwargs = {**config['default_env_config'], **kwargs}
    else:
        merged_kwargs = kwargs
    # Pass env_creator through to GYMNASIUM vecenv for manual vectorization
    # Skip for 'gymnasium' config — GymnasiumVecEnv handles env_name from kwargs natively
    if vec_env_name == 'GYMNASIUM' and 'env_creator' in config and 'env_creator' not in merged_kwargs and config_name != 'gymnasium':
        merged_kwargs['env_creator'] = config['env_creator']
    return vecenv_config[vec_env_name](config_name, num_actors, **merged_kwargs)

register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))

from rl_games.envs.brax import BraxEnv
register('BRAX', lambda config_name, num_actors, **kwargs: BraxEnv(config_name, num_actors, **kwargs))

from rl_games.envs.maniskill import ManiskillEnv
register('MANISKILL', lambda config_name, num_actors, **kwargs: ManiskillEnv(config_name, num_actors, **kwargs))

from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv
register('GYMNASIUM', lambda config_name, num_actors, **kwargs: GymnasiumVecEnv(config_name, num_actors, **kwargs))

def _create_envpool(config_name, num_actors, **kwargs):
    from rl_games.envs.envpool import Envpool
    return Envpool(config_name, num_actors, **kwargs)
register('ENVPOOL', _create_envpool)

def _create_dmc_soccer_selfplay(config_name, num_actors, **kwargs):
    from rl_games.envs.dmc_soccer_selfplay import SoccerSelfPlay
    return SoccerSelfPlay(config_name, num_actors, **kwargs)
register('DMC_SOCCER_SELFPLAY', _create_dmc_soccer_selfplay)

def _create_pufferlib(config_name, num_actors, **kwargs):
    from rl_games.envs.pufferlib_vecenv import PufferLibVecEnv
    return PufferLibVecEnv(config_name, num_actors, **kwargs)
register('PUFFERLIB', _create_pufferlib)

def _create_mjlab(config_name, num_actors, **kwargs):
    from rl_games.envs.mjlab_vecenv import MjlabVecEnv
    return MjlabVecEnv(config_name, num_actors, **kwargs)
register('MJLAB', _create_mjlab)