"""BasePlayer must accept both the classic 4-tuple and gymnasium 5-tuple env
contracts: the myo_gym path returns gymnasium-native envs since MyoSuite
dropped OldGymWrapper."""
import numpy as np
import torch

from rl_games.common.player import BasePlayer


class _Shim:
    """Just enough BasePlayer surface to run env_step/env_reset unbound."""
    is_tensor_obses = False
    value_size = 1

    def obs_to_torch(self, obs):
        return obs


class _GymnasiumEnv:
    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, actions):
        return np.zeros(3, dtype=np.float32), 1.0, False, True, {}


class _ClassicEnv:
    def reset(self):
        return np.zeros(3, dtype=np.float32)

    def step(self, actions):
        return np.zeros(3, dtype=np.float32), 1.0, True, {}


def test_env_step_gymnasium_5tuple():
    obs, rew, done, info = BasePlayer.env_step(_Shim(), _GymnasiumEnv(), torch.zeros(1))
    assert bool(done[0])   # terminated OR truncated
    assert float(rew[0]) == 1.0


def test_env_step_classic_4tuple():
    obs, rew, done, info = BasePlayer.env_step(_Shim(), _ClassicEnv(), torch.zeros(1))
    assert bool(done[0])


def test_env_reset_gymnasium_tuple_unwrapped():
    obs = BasePlayer.env_reset(_Shim(), _GymnasiumEnv())
    assert isinstance(obs, np.ndarray) and obs.shape == (3,)


def test_env_reset_classic_passthrough():
    obs = BasePlayer.env_reset(_Shim(), _ClassicEnv())
    assert isinstance(obs, np.ndarray) and obs.shape == (3,)
