"""BasePlayer must forward env_config['seed'] to vecenv and regular envs without mutating config."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer


class _DummyPlayer(BasePlayer):
    def load_networks(self, params):
        self.model = None
        self.is_rnn = False
        self.normalize_input = False


def _base_params(env_config, use_vecenv, env_name='dummy_seed_env', player_extra=None):
    player_cfg = {'use_vecenv': use_vecenv}
    if player_extra:
        player_cfg.update(player_extra)
    return {
        'config': {
            'env_name': env_name,
            'num_actors': 2,
            'device_name': 'cpu',
            'env_config': env_config,
            'player': player_cfg,
        }
    }


def test_base_player_forwards_seed_to_vecenv_without_mutating_config(monkeypatch):
    captured = {}

    class _FakeVecEnv:
        def get_env_info(self):
            return {
                'action_space': gym.spaces.Box(-1.0, 1.0, shape=(2,)),
                'observation_space': gym.spaces.Box(-1.0, 1.0, shape=(3,)),
            }

    def fake_create_vec_env(env_name, num_actors, **kwargs):
        captured['env_name'] = env_name
        captured['num_actors'] = num_actors
        captured['kwargs'] = kwargs
        return _FakeVecEnv()

    monkeypatch.setattr(vecenv, 'create_vec_env', fake_create_vec_env)

    env_cfg = {'seed': 42, 'foo': 'bar'}
    player = _DummyPlayer(_base_params(env_cfg, use_vecenv=True))

    assert player.seed == 42
    assert captured['kwargs'] == {'seed': 42, 'foo': 'bar'}
    assert env_cfg == {'seed': 42, 'foo': 'bar'}


def test_base_player_handles_none_env_config_and_player_env_config_fallback(monkeypatch):
    captured = {}

    class _FakeVecEnv:
        def get_env_info(self):
            return {
                'action_space': gym.spaces.Box(-1.0, 1.0, shape=(2,)),
                'observation_space': gym.spaces.Box(-1.0, 1.0, shape=(3,)),
            }

    def fake_create_vec_env(env_name, num_actors, **kwargs):
        captured['kwargs'] = kwargs
        return _FakeVecEnv()

    monkeypatch.setattr(vecenv, 'create_vec_env', fake_create_vec_env)

    p_none = _DummyPlayer(_base_params(None, use_vecenv=True))
    assert p_none.seed is None

    p_fallback = _DummyPlayer(
        _base_params(
            {'seed': 77},
            use_vecenv=True,
            player_extra={'env_config': {'custom': 1}},
        )
    )
    assert p_fallback.seed == 77
    assert captured['kwargs'] == {'custom': 1, 'seed': 77}


def test_base_player_seeds_regular_env_and_resets_once_with_seed(monkeypatch):
    reset_calls = []
    seed_calls = []

    class _FakeGymnasiumEnv:
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))

        def seed(self, value):
            seed_calls.append(value)

        def reset(self, seed=None):
            reset_calls.append(seed)
            return np.zeros(3, dtype=np.float32), {}

    monkeypatch.setitem(
        env_configurations.configurations,
        'dummy_seed_env',
        {'env_creator': lambda **kwargs: _FakeGymnasiumEnv()},
    )
    monkeypatch.setattr(
        env_configurations,
        'get_env_info',
        lambda env: {
            'action_space': env.action_space,
            'observation_space': env.observation_space,
        },
    )

    env_cfg = {'seed': 123}
    player = _DummyPlayer(_base_params(env_cfg, use_vecenv=False))

    assert seed_calls == [123]
    assert env_cfg == {'seed': 123}

    player.env_reset(player.env)
    player.env_reset(player.env)
    assert reset_calls == [123, None]


def test_base_player_env_reset_falls_back_for_seedless_reset():
    calls = []

    class _NoSeedResetEnv:
        def reset(self):
            calls.append('no_seed')
            return np.zeros(3, dtype=np.float32)

    shim = SimpleNamespace(
        seed=99,
        _seed_pending=True,
        is_tensor_obses=False,
        obs_to_torch=lambda obs: obs,
    )
    obs = BasePlayer.env_reset(shim, _NoSeedResetEnv())
    assert calls == ['no_seed']
    assert shim._seed_pending is False
    assert isinstance(obs, np.ndarray)


def test_e2e_cartpole_determinism_vecenv_and_regular():
    # 1. Native GymnasiumVecEnv (use_vecenv=True)
    cfg_vec_42 = _base_params(
        {'env_name': 'CartPole-v1', 'seed': 42, 'use_async': False},
        use_vecenv=True,
        env_name='gymnasium',
    )
    cfg_vec_43 = _base_params(
        {'env_name': 'CartPole-v1', 'seed': 43, 'use_async': False},
        use_vecenv=True,
        env_name='gymnasium',
    )
    p1 = _DummyPlayer(cfg_vec_42)
    p2 = _DummyPlayer(cfg_vec_42)
    p3 = _DummyPlayer(cfg_vec_43)

    obs1 = p1.env_reset(p1.env)
    obs2 = p2.env_reset(p2.env)
    obs3 = p3.env_reset(p3.env)
    assert torch.equal(obs1, obs2)
    assert not torch.equal(obs1, obs3)

    # 2. Custom env_creator in GymnasiumVecEnv (via #408 pending seed)
    def creator(**kw):
        return gym.make('CartPole-v1')

    c1 = _DummyPlayer(
        _base_params(
            {'env_creator': creator, 'seed': 42},
            use_vecenv=True,
            env_name='gymnasium',
        )
    )
    c2 = _DummyPlayer(
        _base_params(
            {'env_creator': creator, 'seed': 42},
            use_vecenv=True,
            env_name='gymnasium',
        )
    )
    assert torch.equal(c1.env_reset(c1.env), c2.env_reset(c2.env))


@pytest.mark.parametrize("override, legacy, expected", [
    ({}, {}, "cpu"),
    ({}, {"device_name": "cuda:1"}, "cuda:1"),
    ({"device": "cpu"}, {"device_name": "cuda:1"}, "cpu"),
])
def test_player_device_precedence(override, legacy, expected):
    params = _base_params({}, True, player_extra=override)
    cfg = params['config']
    cfg.pop('device_name')
    cfg.update(device='cpu', env_info={
        'action_space': gym.spaces.Discrete(2),
        'observation_space': gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
    }, **legacy)
    assert _DummyPlayer(params).device == torch.device(expected)
