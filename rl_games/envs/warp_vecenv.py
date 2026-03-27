"""NVIDIA Warp vectorized environment wrapper for rl_games.

Warp environments are GPU-accelerated and can run many parallel instances.
This wrapper adapts Warp-based gymnasium envs to rl_games' IVecEnv interface,
handling tensor conversion between Warp arrays and PyTorch tensors.

Usage:
    1. Create a gymnasium-compatible env that uses Warp for simulation
    2. Register it with rl_games:
        env_configurations.register('my_env', {
            'env_creator': lambda **kwargs: MyWarpEnv(**kwargs),
            'vecenv_type': 'WARP'
        })
    3. Use env_name: 'my_env' in your config
"""

import torch
import numpy as np
from rl_games.common.ivecenv import IVecEnv
from rl_games.common import env_configurations


class WarpVecEnv(IVecEnv):
    """Wraps Warp-based GPU environments for rl_games.

    Handles conversion between Warp arrays and PyTorch tensors using
    warp.to_torch() for zero-copy GPU interop. Supports both single
    and multi-instance environments.

    The wrapped env should:
    - Accept count_env (or num_envs) in its constructor
    - Return batched observations/rewards/dones from step() and reset()
    - Support Warp arrays or numpy arrays as outputs
    """

    def __init__(self, config_name, num_actors, **kwargs):
        self.num_actors = num_actors

        env_name = kwargs.pop('env_name', config_name)
        self.seed_value = kwargs.pop('seed', None)
        self.device = kwargs.pop('device', 'cuda:0')

        # Remove rl_games-specific keys
        for key in ['full_experiment_name', 'name']:
            kwargs.pop(key, None)

        env_creator = env_configurations.configurations[config_name]['env_creator']
        self.env = env_creator(count_env=self.num_actors, **kwargs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._warp_available = False
        try:
            import warp as wp
            self._warp_available = True
            self._wp = wp
        except ImportError:
            pass

    def _to_torch(self, x):
        """Convert Warp array or numpy array to PyTorch tensor on the correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)

        if self._warp_available:
            wp = self._wp
            if isinstance(x, wp.array):
                return wp.to_torch(x).to(self.device)

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)

        return torch.tensor(x, device=self.device)

    def _from_torch(self, x):
        """Convert PyTorch tensor to Warp array if Warp is available, otherwise numpy."""
        if self._warp_available:
            return self._wp.from_torch(x.contiguous())
        return x.cpu().numpy()

    def step(self, actions):
        # Convert actions from torch to whatever the env expects
        if self._warp_available:
            warp_actions = self._from_torch(actions.reshape(-1) if actions.dim() > 1 else actions)
        else:
            warp_actions = actions

        result = self.env.step(warp_actions)

        # Handle both 4-tuple and 5-tuple returns
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # Combine terminated and truncated into done
            done = self._to_torch(terminated).float()
            trunc = self._to_torch(truncated)
            if isinstance(trunc, torch.Tensor) and trunc.dtype == torch.bool:
                trunc = trunc.float()
            done = ((done + trunc) > 0).float()
            if not isinstance(info, dict):
                info = {}
            info['time_outs'] = self._to_torch(truncated)
        elif len(result) == 4:
            obs, reward, done, info = result
            done = self._to_torch(done).float()
            # Extract time_outs if present
            if isinstance(info, dict) and 'time_outs' in info:
                info['time_outs'] = self._to_torch(info['time_outs'])
        else:
            raise ValueError(f"env.step() returned {len(result)} values, expected 4 or 5")

        obs = self._to_torch(obs)
        reward = self._to_torch(reward).float()

        # Ensure correct shapes for rl_games
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)

        return obs, reward, done, info

    def reset(self):
        result = self.env.reset()

        # Handle both single return and tuple return
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result

        return self._to_torch(obs)

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        return 1

    def get_env_info(self):
        if hasattr(self.env, 'get_env_info'):
            return self.env.get_env_info()
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }

    def seed(self, seed):
        self.seed_value = seed
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
