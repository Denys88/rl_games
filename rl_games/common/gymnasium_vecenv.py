from rl_games.common.ivecenv import IVecEnv
import gymnasium
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np

# Register ALE (Atari) environments if available
try:
    import ale_py
    gymnasium.register_envs(ale_py)
except ImportError:
    pass  # ale_py not installed, Atari envs won't be available


def wrap_atari(env, frame_stack=4, noop_max=30, frame_skip=4,
               terminal_on_life_loss=True, grayscale=True, scale=False):
    """Apply standard Atari preprocessing wrappers.

    Args:
        env: Base Atari environment
        frame_stack: Number of frames to stack (0 to disable)
        noop_max: Max no-ops at reset
        frame_skip: Frames to skip between actions
        terminal_on_life_loss: End episode on life loss
        grayscale: Convert to grayscale
        scale: Scale observations to [0,1)
    """
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=84,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,  # Don't add channel dim - FrameStack will stack along new axis
        scale_obs=scale
    )
    if frame_stack > 0:
        env = FrameStackObservation(env, frame_stack)
    return env


def _dicts_to_dict_with_arrays(dicts, squeeze=False):
    """Convert a list of dicts to a dict of stacked arrays."""
    result = {}
    for key in dicts[0].keys():
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            result[key] = _dicts_to_dict_with_arrays(values, squeeze)
        else:
            result[key] = np.stack(values)
            if squeeze:
                result[key] = result[key].squeeze()
    return result


class GymnasiumVecEnv(IVecEnv):
    """Vectorized environment wrapper using gymnasium.vector.

    This replaces Ray-based parallelization for standard gymnasium environments,
    providing better performance and simpler setup.

    Args:
        config_name: Environment configuration name (used as env_name fallback)
        num_actors: Number of parallel environments
        env_name: Gymnasium environment ID (defaults to config_name)
        use_async: Use AsyncVectorEnv instead of SyncVectorEnv
        seed: Random seed for environments
        wrap_env: Optional function to wrap each env before vectorization
        env_creator: Optional callable to create custom envs (bypasses gymnasium.make)
        **kwargs: Additional kwargs passed to gymnasium.make() or env_creator()
    """

    def __init__(self, config_name, num_actors, **kwargs):
        self.batch_size = num_actors
        # Use config_name as env_name if not explicitly provided
        env_name = kwargs.pop('env_name', config_name)
        use_async = kwargs.pop('use_async', True)
        self.seed_value = kwargs.pop('seed', None)
        wrap_env = kwargs.pop('wrap_env', None)
        env_creator = kwargs.pop('env_creator', None)

        # Store remaining kwargs for env creation
        self.env_kwargs = kwargs

        if env_creator is not None:
            # Custom env creator - use manual vectorization for multi-agent support
            self._use_native_vec = False
            self.envs = [env_creator(**dict(self.env_kwargs)) for _ in range(num_actors)]
            sample_env = self.envs[0]
            self.observation_space = sample_env.observation_space
            self.action_space = sample_env.action_space
        else:
            # Standard gymnasium env - use native vectorized env
            self._use_native_vec = True

            def make_env(idx):
                def _init():
                    env = gymnasium.make(env_name, **self.env_kwargs)
                    if wrap_env is not None:
                        env = wrap_env(env)
                    if self.seed_value is not None:
                        env.reset(seed=self.seed_value + idx)
                    return env
                return _init

            VecEnvClass = AsyncVectorEnv if use_async else SyncVectorEnv
            self.env = VecEnvClass([make_env(i) for i in range(num_actors)])

            single_obs_space = self.env.single_observation_space
            single_action_space = self.env.single_action_space
            self.observation_space = single_obs_space
            self.action_space = single_action_space

    def step(self, action):
        if self._use_native_vec:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            is_done = terminated | truncated
            info['time_outs'] = truncated
            return next_obs, reward, is_done, info
        else:
            return self._step_manual(action)

    def _step_manual(self, actions):
        """Step all envs manually (for multi-agent / custom envs)."""
        sample_env = self.envs[0]
        num_agents = self._get_num_agents(sample_env)

        all_obs, all_rewards, all_dones, all_infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if num_agents == 1:
                act = actions[i]
            else:
                act = actions[i * num_agents:(i + 1) * num_agents]
            obs, reward, terminated, truncated, info = env.step(act)
            is_done = terminated | truncated if np.isscalar(terminated) else terminated | truncated
            if 'time_outs' not in info:
                info['time_outs'] = truncated

            if np.isscalar(is_done):
                episode_done = is_done
            else:
                episode_done = is_done.all()
            if episode_done:
                obs, _ = env.reset()

            all_obs.append(obs)
            all_rewards.append(reward)
            all_dones.append(is_done)
            all_infos.append(info)

        concat_func = np.concatenate if num_agents > 1 else np.stack

        # Check if obs are dicts (central_value)
        use_global_obs = isinstance(all_obs[0], dict) and 'obs' in all_obs[0]
        if use_global_obs:
            newobs = [o['obs'] for o in all_obs]
            newstates = [o['state'] for o in all_obs]
            ret_obs = {
                'obs': concat_func(newobs),
                'states': np.stack(newstates),
            }
        else:
            ret_obs = concat_func(all_obs)

        rewards = concat_func(all_rewards)
        dones = concat_func(all_dones)

        # Merge time_outs from infos
        time_outs = []
        for info in all_infos:
            to = info.get('time_outs', False)
            if np.isscalar(to):
                time_outs.append(to)
            else:
                time_outs.extend(to)
        merged_info = {'time_outs': np.array(time_outs)}
        return ret_obs, rewards, dones, merged_info

    def reset(self):
        if self._use_native_vec:
            obs, info = self.env.reset()
            return obs
        else:
            return self._reset_manual()

    def _reset_manual(self):
        """Reset all envs manually (for multi-agent / custom envs)."""
        sample_env = self.envs[0]
        num_agents = self._get_num_agents(sample_env)
        concat_func = np.concatenate if num_agents > 1 else np.stack

        all_obs = []
        for env in self.envs:
            obs, info = env.reset()
            all_obs.append(obs)

        use_global_obs = isinstance(all_obs[0], dict) and 'obs' in all_obs[0]
        if use_global_obs:
            newobs = [o['obs'] for o in all_obs]
            newstates = [o['state'] for o in all_obs]
            ret_obs = {
                'obs': concat_func(newobs),
                'states': np.stack(newstates),
            }
        else:
            ret_obs = concat_func(all_obs)

        return ret_obs

    def _get_num_agents(self, env):
        if hasattr(env, 'get_number_of_agents'):
            return env.get_number_of_agents()
        return 1

    def seed(self, seed):
        self.seed_value = seed

    def get_number_of_agents(self):
        if not self._use_native_vec and self.envs:
            return self._get_num_agents(self.envs[0])
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if not self._use_native_vec and self.envs:
            sample_env = self.envs[0]
            if hasattr(sample_env, 'use_central_value'):
                info['use_global_observations'] = sample_env.use_central_value
            if hasattr(sample_env, 'value_size'):
                info['value_size'] = sample_env.value_size
            if hasattr(sample_env, 'state_space'):
                info['state_space'] = sample_env.state_space
        return info


def create_gymnasium_env(**kwargs):
    return GymnasiumVecEnv("", kwargs.pop('num_actors', 16), **kwargs)
