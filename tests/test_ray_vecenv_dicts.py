"""RayVecEnv call sites batch shipped asymmetric observations without Ray in CI."""
import itertools
import sys
import types

import numpy as np
import pytest

from rl_games.common import env_configurations
from rl_games.common.vecenv import RayVecEnv


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _Actor:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return _Remote(getattr(self._obj, name))


@pytest.fixture
def inline_ray(monkeypatch):
    ray = types.ModuleType('ray')
    ray.is_initialized = lambda: True
    ray.get = lambda value: value
    ray.remote = lambda cls: _Remote(lambda *args, **kwargs: _Actor(cls(*args, **kwargs)))
    monkeypatch.setitem(sys.modules, 'ray', ray)

    def concat_infos_env(**kwargs):
        env = env_configurations.create_test_env('TestRnnEnv-v0', **kwargs)
        env.unwrapped.concat_infos = True
        return env

    monkeypatch.setitem(env_configurations.configurations, 'test_env_concat_infos',
                        {'env_creator': concat_infos_env, 'vecenv_type': 'RAY'})


@pytest.mark.parametrize('num_actors', [1, 2])
def test_ray_vecenv_batches_dicts_for_any_worker_count(inline_ray, num_actors):
    env = RayVecEnv('test_env_concat_infos', num_actors, multi_obs_space=True,
                    use_central_value=True, disable_env_checker=True)
    obs = env.reset()
    assert obs['obs']['pos'].shape == (num_actors, 2)
    assert obs['states']['info'].shape == (num_actors, 4)
    obs, _, _, infos = env.step(np.zeros(num_actors, dtype=np.int64))
    assert obs['obs']['pos'].shape == (num_actors, 2)
    assert obs['states']['info'].shape == (num_actors, 4)
    assert infos['time_outs'].shape == (num_actors,)

@pytest.mark.parametrize('concat_infos', [True, False])
def test_ray_timeout_keeps_final_actor_and_critic_observations(inline_ray, monkeypatch, concat_infos):
    from tests.test_gymnasium_terminal_obs import TerminalEnv

    count = itertools.count()

    def create():
        env = TerminalEnv(env_id=next(count), agents=2, layout='central')
        env.concat_infos = concat_infos
        return env

    monkeypatch.setitem(env_configurations.configurations, 'ray_timeout', {'env_creator': create})
    env = RayVecEnv('ray_timeout', 2)
    env.reset()
    obs, _, _, info = env.step(np.zeros((4, 2)))
    np.testing.assert_array_equal(obs['obs'][:, 1], [0, 0, 1, 1])
    np.testing.assert_array_equal(info['final_observation']['obs'][:, 1], [1, 1, 1, 1])
    np.testing.assert_array_equal(info['final_observation']['states'][:, 1], [1, 1])
    np.testing.assert_array_equal(info['_final_observation'], [True, True, False, False])
