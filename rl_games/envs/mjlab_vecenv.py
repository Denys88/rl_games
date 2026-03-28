"""MJLab (MuJoCo Lab) vectorized environment wrapper for rl_games.

MJLab provides Isaac Lab's manager-based API powered by MuJoCo Warp,
running GPU-accelerated environments without Isaac Sim/Omniverse.

Available tasks: Mjlab-Velocity-Flat-Unitree-G1, Mjlab-Velocity-Rough-Unitree-G1,
Mjlab-Velocity-Flat-Unitree-Go1, Mjlab-Tracking-Flat-Unitree-G1, etc.
"""

import torch
from rl_games.common.ivecenv import IVecEnv


class MjlabVecEnv(IVecEnv):
    """Wraps MJLab environments for rl_games.

    MJLab envs return dict observations with 'actor' and 'critic' keys.
    This wrapper converts to rl_games format.
    """

    def __init__(self, config_name, num_actors, **kwargs):
        import warp as wp
        wp.init()

        from mjlab.tasks.registry import load_env_cfg
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

        task_name = kwargs.pop('task_name', config_name)
        self.device = kwargs.pop('device', 'cuda')

        cfg = load_env_cfg(task_name)
        cfg.scene.num_envs = num_actors

        self.env = ManagerBasedRlEnv(cfg, device=self.device)

        obs, _ = self.env.reset()

        # Extract spaces from first obs
        self.num_envs = obs['actor'].shape[0]
        self.obs_dim = obs['actor'].shape[-1]
        self.has_critic_obs = 'critic' in obs

        from gymnasium import spaces
        import numpy as np

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.env.action_space.shape[-1],), dtype=np.float32
        )

        if self.has_critic_obs:
            critic_dim = obs['critic'].shape[-1]
            self.state_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(critic_dim,), dtype=np.float32
            )

    def step(self, actions):
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)

        done = terminated | truncated
        info['time_outs'] = truncated

        if self.has_critic_obs:
            obs = {'obs': obs_dict['actor'], 'states': obs_dict['critic']}
        else:
            obs = obs_dict['actor']

        return obs, reward, done.float(), info

    def reset(self):
        obs_dict, _ = self.env.reset()

        if self.has_critic_obs:
            return {'obs': obs_dict['actor'], 'states': obs_dict['critic']}
        return obs_dict['actor']

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }
        if self.has_critic_obs:
            info['state_space'] = self.state_space
            info['use_global_observations'] = True
        return info

    def seed(self, seed):
        pass

    def close(self):
        self.env.close()
