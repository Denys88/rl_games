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

        # optional override of a step-scheduled command curriculum (velocity
        # tasks): stage switch points in env steps, e.g. [0, 60000, 120000].
        # The reference schedule stays the default; this is a training-protocol
        # knob for our own runs, not an env modification.
        stage_steps = kwargs.pop('velocity_stage_steps', None)
        if stage_steps is not None:
            term = cfg.curriculum['command_vel']
            stages = term.params['velocity_stages']
            if len(stage_steps) != len(stages):
                raise ValueError(
                    f"velocity_stage_steps has {len(stage_steps)} entries, "
                    f"task schedule has {len(stages)} stages")
            for stage, step in zip(stages, stage_steps):
                stage['step'] = int(step)

        self.env = ManagerBasedRlEnv(cfg, device=self.device)

        obs, _ = self.env.reset()

        # mjlab's own tasks name the policy obs group 'actor'; Isaac Lab-style
        # task plugins (e.g. wuji-mjlab) use 'policy' — accept both
        self.actor_key = 'actor' if 'actor' in obs else 'policy'
        self.num_envs = obs[self.actor_key].shape[0]
        self.obs_dim = obs[self.actor_key].shape[-1]
        self.has_critic_obs = 'critic' in obs

        from gymnasium import spaces
        import numpy as np

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
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
        # surface the env's episode metrics (success rates, per-term rewards)
        # so the observer logs them — reward totals alone hide task failure.
        # mjlab reuses its extras dict across steps and emits an empty 'log'
        # on non-reset steps: refresh on every burst and copy (never alias)
        if info.get('log'):
            info['episode'] = dict(info['log'])
        else:
            info.pop('episode', None)

        if self.has_critic_obs:
            # asymmetric actor-critic: privileged obs feed the central value net
            obs = {'obs': obs_dict[self.actor_key], 'states': obs_dict['critic']}
        else:
            obs = obs_dict[self.actor_key]
        return obs, reward, done.float(), info

    def reset(self):
        obs_dict, _ = self.env.reset()
        if self.has_critic_obs:
            return {'obs': obs_dict[self.actor_key], 'states': obs_dict['critic']}
        return obs_dict[self.actor_key]

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
