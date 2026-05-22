"""IsaacLab vectorized environment wrapper for rl_games.

IsaacLab environments are GPU-accelerated and already vectorized.
They return torch tensors with batch dimension, similar to Brax.
This wrapper adapts the IsaacLab gymnasium-like API to rl_games' IVecEnv interface.

Supports both symmetric and asymmetric (separate policy/critic obs) actor-critic.
"""

import torch
from gymnasium import spaces
from rl_games.common.ivecenv import IVecEnv


def _remove_batch_dim(space):
    """Remove the first (batch) dimension from a gymnasium space."""
    if isinstance(space, spaces.Box):
        low = space.low[0]
        high = space.high[0]
        return spaces.Box(low=low, high=high, dtype=space.dtype)
    elif isinstance(space, spaces.Discrete):
        return space
    elif isinstance(space, spaces.MultiDiscrete):
        nvec = space.nvec[0]
        return spaces.MultiDiscrete(nvec)
    elif isinstance(space, spaces.Dict):
        return spaces.Dict({k: _remove_batch_dim(s) for k, s in space.spaces.items()})
    else:
        return space


class IsaacLabEnv(IVecEnv):
    """Wraps IsaacLab environments (ManagerBasedRLEnv / DirectRLEnv) for rl_games.

    IsaacLab envs are already vectorized on GPU. This wrapper:
    - Converts terminated/truncated to single done flag
    - Passes truncated as info['time_outs'] for value bootstrapping
    - Handles dict observations with 'policy' and 'critic' groups
    - Clips observations and actions within configurable bounds
    - Keeps tensors on GPU (no numpy conversion)
    """

    def __init__(self, config_name, num_actors, **kwargs):
        import gymnasium

        self.batch_size = num_actors
        env_name = kwargs.pop('env_name', config_name)
        self.clip_obs = kwargs.pop('clip_observations', 0.0)
        self.clip_actions = kwargs.pop('clip_actions', 0.0)
        self.device = kwargs.pop('device', 'cuda:0')
        self.seed = kwargs.pop('seed', 0)

        # Remove rl_games-specific keys that shouldn't go to the env
        for key in ['full_experiment_name', 'name']:
            kwargs.pop(key, None)

        self.env = gymnasium.make(
            env_name,
            num_envs=num_actors,
            device=self.device,
            seed=self.seed,
            **kwargs,
        )

        # Detect asymmetric actor-critic (policy + critic observation groups)
        self.has_critic_obs = False
        raw_obs_space = self.env.observation_space
        if isinstance(raw_obs_space, spaces.Dict):
            if 'policy' in raw_obs_space.spaces:
                self.observation_space = _remove_batch_dim(raw_obs_space['policy'])
                if 'critic' in raw_obs_space.spaces:
                    self.has_critic_obs = True
                    self.state_space = _remove_batch_dim(raw_obs_space['critic'])
            else:
                self.observation_space = _remove_batch_dim(raw_obs_space)
        else:
            self.observation_space = _remove_batch_dim(raw_obs_space)

        raw_act_space = self.env.action_space
        self.action_space = _remove_batch_dim(raw_act_space)

    def _process_obs(self, obs):
        """Extract and optionally clip observations."""
        if isinstance(obs, dict):
            policy_obs = obs.get('policy', obs)
            if isinstance(policy_obs, dict):
                # Nested dict obs — flatten or pass through
                policy_obs = policy_obs
            if self.clip_obs > 0:
                if isinstance(policy_obs, torch.Tensor):
                    policy_obs = torch.clamp(policy_obs, -self.clip_obs, self.clip_obs)

            if self.has_critic_obs and 'critic' in obs:
                critic_obs = obs['critic']
                if self.clip_obs > 0 and isinstance(critic_obs, torch.Tensor):
                    critic_obs = torch.clamp(critic_obs, -self.clip_obs, self.clip_obs)
                return {'obs': policy_obs, 'states': critic_obs}

            return policy_obs
        else:
            if self.clip_obs > 0 and isinstance(obs, torch.Tensor):
                obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
            return obs

    def step(self, actions):
        if self.clip_actions > 0 and isinstance(actions, torch.Tensor):
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs, reward, terminated, truncated, info = self.env.step(actions)
        is_done = terminated | truncated
        info['time_outs'] = truncated

        obs = self._process_obs(obs)
        return obs, reward, is_done, info

    def reset(self):
        obs, _ = self.env.reset()
        return self._process_obs(obs)

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
        self.seed = seed

    def close(self):
        self.env.close()


def create_isaaclab_env(**kwargs):
    return IsaacLabEnv(
        kwargs.pop('full_experiment_name', ''),
        kwargs.pop('num_actors', 4096),
        **kwargs,
    )
