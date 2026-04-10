"""EnvPool vectorized environment wrapper for rl_games.

Uses envpool's native gymnasium API (envpool >= 1.1.1).

Supported features that can be passed via env_config:
    env_name (str): Required. envpool task ID (e.g. "Ant-v4", "Pong-v5",
        "HumanoidWalk-v1").
    has_lives (bool): Atari only. Tracks lives so episode score is reported on
        game-over rather than per-life.
    use_dict_obs_space (bool): Atari only. Wraps observations into a Dict
        space containing the previous reward and last action (asymmetric
        actor-critic setups).
    flatten_obs (bool): Flatten Dict observations to a single Box. Required
        for DeepMind Control envs (which return Dict obs).
    frame_stack (int): Native envpool frame stacking. Available for MuJoCo
        and Atari envs (envpool >= 1.1.1).
    from_pixels (bool): MuJoCo only. Use rendered pixel observations instead
        of state. Available in envpool >= 1.1.1.
    Any other kwargs are passed through to envpool.make_gymnasium().
"""
from rl_games.common.ivecenv import IVecEnv
import gymnasium
import numpy as np


def _flatten_dict_obs(obs):
    """Flatten a dict of batched arrays to a single (batch, total_dim) array."""
    parts = [v.reshape(v.shape[0], -1) for v in obs.values()]
    return np.column_stack(parts)


class Envpool(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        self.batch_size = num_actors
        env_name = kwargs.pop('env_name')
        self.has_lives = kwargs.pop('has_lives', False)
        self.use_dict_obs_space = kwargs.pop('use_dict_obs_space', False)
        self.flatten_obs = kwargs.pop('flatten_obs', False)

        self.env = envpool.make_gymnasium(
            env_name,
            num_envs=num_actors,
            batch_size=self.batch_size,
            **kwargs,
        )

        if self.use_dict_obs_space:
            self.observation_space = gymnasium.spaces.Dict({
                'observation': self.env.observation_space,
                'reward': gymnasium.spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
                'last_action': gymnasium.spaces.Box(
                    low=0, high=self.env.action_space.n, shape=(), dtype=int),
            })
        else:
            self.observation_space = self.env.observation_space

        if self.flatten_obs:
            self.orig_observation_space = self.observation_space
            self.observation_space = gymnasium.spaces.flatten_space(self.observation_space)

        self.action_space = self.env.action_space
        self.ids = np.arange(0, num_actors)
        self.scores = np.zeros(num_actors)
        self.returned_scores = np.zeros(num_actors)

    def _set_scores(self, infos, dones):
        """Track raw (unclipped) episode rewards for Atari benchmarking.

        envpool clips Atari rewards to [-1, 1] for training and exposes the
        raw reward via info['reward']. We accumulate the raw reward into
        self.scores and expose the totals as info['scores'] for tensorboard
        reporting. With episodic_life=True we only reset the score on real
        game-over (lives == 0) instead of per-life.
        Adapted from CleanRL's ppo_atari_envpool.
        """
        if 'reward' not in infos:
            return
        self.scores += infos['reward']
        self.returned_scores[:] = self.scores
        infos['scores'] = self.returned_scores

        if self.has_lives:
            all_lives_exhausted = infos['lives'] == 0
            self.scores *= 1 - all_lives_exhausted
        else:
            if 'lives' in infos:
                del infos['lives']
            self.scores *= 1 - dones

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action, self.ids)
        is_done = terminated | truncated
        info['time_outs'] = truncated

        self._set_scores(info, is_done)

        if self.flatten_obs:
            next_obs = _flatten_dict_obs(next_obs)
        if self.use_dict_obs_space:
            next_obs = {
                'observation': next_obs,
                'reward': np.clip(reward, -1, 1),
                'last_action': action,
            }
        return next_obs, reward, is_done, info

    def reset(self):
        obs, info = self.env.reset(self.ids)

        if self.flatten_obs:
            obs = _flatten_dict_obs(obs)
        if self.use_dict_obs_space:
            obs = {
                'observation': obs,
                'reward': np.zeros(obs.shape[0]),
                'last_action': np.zeros(obs.shape[0]),
            }
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }


def create_envpool(**kwargs):
    return Envpool('', kwargs.pop('num_actors', 16), **kwargs)
