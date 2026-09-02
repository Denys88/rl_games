"""Regression tests for the release-preparation correctness fixes: the
central value normalizer prior (default from the CV's own geometry; explicit
count from central_value_config, else the top-level key), the player's vecenv
fallback for creator-less registrations, and discrete PPO honoring
multi_gpu_scheduler_kl: 'local' through _kl_for_lr_schedule."""

import pytest


class _Captured(Exception):
    pass


class _CaptureNetwork:
    """Stub network whose build() records the state config and aborts."""

    def __init__(self):
        self.state_config = None

    def build(self, cfg):
        self.state_config = cfg
        raise _Captured


def _make_cv_train(network, horizon=8, num_actors=4, cv_mini_epochs=2,
                   state_shape=(4,), explicit_count=None, **config_extra):
    """CentralValueTrain on CPU around `network`, minimal config."""
    from rl_games.algos_torch.central_value import CentralValueTrain

    config = {
        'mini_epochs': cv_mini_epochs,
        'normalize_input': True,
        'learning_rate': 1e-3,
        'clip_value': False,
        'mixed_precision': False,
        'lr_schedule': None,
        'schedule_type': 'standard',
        'kl_threshold': 0.01,
        'grad_norm': 1.0,
        'truncate_grads': False,
        'minibatch_size': horizon * num_actors,
        **config_extra,
    }
    return CentralValueTrain(
        state_shape=state_shape, value_size=1, ppo_device='cpu', num_agents=1,
        horizon_length=horizon, num_actors=num_actors, num_actions=3,
        seq_length=4, normalize_value=False, network=network, config=config,
        writter=None, max_epochs=1, multi_gpu=False, zero_rnn_on_done=True,
        normalize_input_init_count=explicit_count)


_ABSENT = object()


def _capture_cv_init_count(cv_mini_epochs, horizon, num_actors, explicit=None,
                           cv_key=_ABSENT):
    """`explicit` plays the agent's raw top-level key; `cv_key` the
    central_value_config key."""
    net = _CaptureNetwork()
    extra = {} if cv_key is _ABSENT else {'normalize_input_init_count': cv_key}
    with pytest.raises(_Captured):
        _make_cv_train(net, horizon, num_actors, cv_mini_epochs,
                       explicit_count=explicit, **extra)
    return net.state_config['normalize_input_init_count']


def _test_network_cv_stack():
    """The shipped central-value TestNet inside the real ModelCentralValue
    wrapper and CentralValueTrain, as A2CBase.load_networks builds it."""
    import rl_games.envs  # noqa: F401  (registers 'testnet')
    from rl_games.algos_torch import model_builder

    network = model_builder.ModelBuilder().load(
        {'model': {'name': 'central_value'},
         'network': {'name': 'testnet', 'central_value': True}})
    return _make_cv_train(network, state_shape={'pos': (2,), 'info': (2,)},
                          normalize_input=False)


def test_cv_normalizer_prior_uses_cv_geometry():
    # default: cv_mini_epochs * horizon * num_actors -- NOT the actor's count
    assert _capture_cv_init_count(2, 8, 4) == 2 * 8 * 4
    # SMAC-shaped: on the stock 27-agent config the actor count was 27x this
    assert _capture_cv_init_count(4, 128, 8) == 4 * 128 * 8


def test_cv_normalizer_prior_explicit_value_preserved():
    assert _capture_cv_init_count(4, 128, 8, explicit=123) == 123


def test_player_defaults_to_vecenv_without_env_creator():
    from rl_games.common.player import BasePlayer
    from rl_games.common import env_configurations

    # envpool registers only a vecenv_type -- --play used to raise
    # KeyError: 'env_creator' here (regression from the UV migration)
    assert 'env_creator' not in env_configurations.configurations['envpool']
    assert BasePlayer._default_use_vecenv('envpool') is True

    # registrations that ship a creator keep the classic path
    with_creator = [k for k, v in env_configurations.configurations.items()
                    if 'env_creator' in v]
    assert with_creator, 'expected at least one creator-based registration'
    assert BasePlayer._default_use_vecenv(with_creator[0]) is False

    # unregistered names keep the classic path: a downstream subclass
    # overriding create_env() under its own name must get its override
    # consulted, not a KeyError from vecenv.create_vec_env
    assert 'no-such-env' not in env_configurations.configurations
    assert BasePlayer._default_use_vecenv('no-such-env') is False


def test_player_config_registered_vecenv_type_env_defaults_to_vecenv():
    """Configs that carry a top-level vecenv_type (the mjlab ones) register
    their env_name at Runner.load time, before the player is built, so the
    default still routes them through vecenv."""
    from rl_games.common.player import BasePlayer
    from rl_games.common import env_configurations
    from rl_games.torch_runner import Runner

    name = 'pytest-vecenv-only-env'
    assert name not in env_configurations.configurations
    try:
        Runner().load({'params': {
            'seed': 0, 'algo': {'name': 'a2c_continuous'},
            'config': {'env_name': name, 'vecenv_type': 'MJLAB',
                       'reward_shaper': {'scale_value': 1.0}}}})
        assert env_configurations.configurations[name] == {'vecenv_type': 'MJLAB'}
        assert BasePlayer._default_use_vecenv(name) is True
    finally:
        env_configurations.configurations.pop(name, None)


def _downstream_player_params(env_name, **player):
    return {
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {
            'name': 'actor_critic', 'separate': False,
            'space': {'continuous': {
                'mu_activation': 'None', 'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'const_initializer', 'val': 0.0},
                'fixed_sigma': True}},
            'mlp': {'units': [8], 'activation': 'elu',
                    'initializer': {'name': 'default'}},
        },
        'config': {'env_name': env_name, 'num_actors': 1,
                   'device_name': 'cpu', 'player': player},
    }


def test_player_unregistered_env_reaches_create_env_override():
    """Denys88's downstream case: a BasePlayer subclass overriding
    create_env() under a name absent from env_configurations. The default
    must take the classic path (override consulted); an explicit
    player.use_vecenv still wins in both directions."""
    import numpy as np
    import gymnasium as gym
    from rl_games.common.player import BasePlayer
    from rl_games.common import env_configurations

    class _Env:
        observation_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    class _Player(BasePlayer):
        def __init__(self, params):
            self.create_env_calls = 0
            super().__init__(params)

        def create_env(self):
            self.create_env_calls += 1
            return _Env()

    name = 'pytest-downstream-env'
    assert name not in env_configurations.configurations
    player = _Player(_downstream_player_params(name))
    assert player.create_env_calls == 1
    assert player.env_info['observation_space'].shape == (3,)

    # explicit use_vecenv: True wins: vecenv path, whose registry lookup
    # raises on the unregistered name
    with pytest.raises(KeyError, match=name):
        _Player(_downstream_player_params(name, use_vecenv=True))

    # explicit use_vecenv: False wins for a creator-less registration too
    player = _Player(_downstream_player_params('envpool', use_vecenv=False))
    assert player.create_env_calls == 1


def test_cv_test_network_strict_loads_pre_2_0_checkpoint():
    """Pre-2.0 TestNet always built the actor head, so checkpoints of the
    asymmetric test config carry model.a2c_network.mean_linear.* under
    assymetric_vf_nets; the central-value TestNet no longer has that head and
    set_full_state_weights restores the CV strictly."""
    import torch

    cv = _test_network_cv_stack()
    keys = set(cv.state_dict().keys())
    assert not any('mean_linear' in k for k in keys)

    legacy = {k: torch.full_like(v, 0.5) for k, v in cv.state_dict().items()}
    legacy['model.a2c_network.mean_linear.weight'] = torch.zeros(3, 64)
    legacy['model.a2c_network.mean_linear.bias'] = torch.zeros(3)
    cv.load_state_dict(legacy)
    assert set(cv.state_dict().keys()) == keys
    assert all(torch.all(v == 0.5) for v in cv.state_dict().values())

    # the filter is scoped to the dead head: any other stray key still raises
    legacy['model.a2c_network.bogus.weight'] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='bogus'):
        cv.load_state_dict(legacy)

    # actor-mode TestNet keeps its head and loads it as before
    from rl_games.envs.test_network import TestNet
    actor = TestNet({}, actions_num=3, input_shape={'pos': (2,), 'info': (2,)})
    sd = {k: torch.full_like(v, 0.25) for k, v in actor.state_dict().items()}
    assert 'mean_linear.weight' in sd
    actor.load_state_dict(sd)
    assert torch.all(actor.mean_linear.weight == 0.25)


def test_cv_init_count_explicit_fallback_order():
    """Explicit counts: central_value_config key > top-level key > the CV's
    own geometry default; resolved inside CentralValueTrain."""
    # top-level explicit value still reaches the CV (backward compat)
    assert _capture_cv_init_count(4, 128, 8, explicit=81920) == 81920
    # the CV-scoped key wins over the top-level one
    assert _capture_cv_init_count(4, 128, 8, explicit=81920, cv_key=123) == 123
    # nothing explicit -> the CV derives its own default
    assert _capture_cv_init_count(4, 128, 8) == 4 * 128 * 8


def test_remove_batch_dim_rebuilds_maniskill_spaces():
    """ManiSkill vector envs expose batched Box / Dict-of-Box spaces;
    remove_batch_dim rebuilds them per-env (shape, bounds, dtype) on its own,
    so ManiskillEnv needs no second rebuild through convert_space."""
    import numpy as np
    from gymnasium import spaces
    from rl_games.envs.maniskill import remove_batch_dim

    box = spaces.Box(low=-np.ones((16, 7), np.float32), high=np.ones((16, 7), np.float32))
    rgb = spaces.Box(low=0, high=255, shape=(16, 64, 64, 3), dtype=np.uint8)
    out = remove_batch_dim(spaces.Dict({'state': box, 'rgb': rgb}))
    assert isinstance(out, spaces.Dict)
    assert out['state'] == spaces.Box(low=-np.ones(7, np.float32), high=np.ones(7, np.float32))
    assert out['rgb'] == spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    assert remove_batch_dim(box).dtype == np.float32


def test_discrete_scheduler_kl_respects_local_mode():
    """multi_gpu_scheduler_kl='local' must skip the collective: the shared
    _kl_for_lr_schedule returns the local estimate untouched (no dist calls),
    which the discrete train_epoch now consumes."""
    import types
    import torch
    from rl_games.common.a2c_common import A2CBase

    stub = types.SimpleNamespace(multi_gpu=True, multi_gpu_scheduler_kl='local',
                                 world_size=2)
    kl = torch.tensor(0.017)
    out = A2CBase._kl_for_lr_schedule(stub, kl)
    assert out is kl and float(out) == pytest.approx(0.017)

    # single-GPU: 'global' must be a no-op too (no dist initialized here --
    # a collective would raise)
    stub = types.SimpleNamespace(multi_gpu=False, multi_gpu_scheduler_kl='global',
                                 world_size=1)
    assert float(A2CBase._kl_for_lr_schedule(stub, torch.tensor(0.03))) == pytest.approx(0.03)
