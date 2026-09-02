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


def _capture_cv_init_count(cv_mini_epochs, horizon, num_actors, explicit=None):
    net = _CaptureNetwork()
    with pytest.raises(_Captured):
        _make_cv_train(net, horizon, num_actors, cv_mini_epochs,
                       explicit_count=explicit)
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

    # unknown names fall through to vecenv, whose own error is actionable
    assert BasePlayer._default_use_vecenv('no-such-env') is True


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
    """Explicit counts: central_value_config key > top-level key > None."""
    from rl_games.algos_torch.central_value import resolve_cv_init_count

    # top-level explicit value still reaches the CV (backward compat)
    assert resolve_cv_init_count({'normalize_input_init_count': 81920}, {}) == 81920
    # the CV-scoped key wins over the top-level one
    assert resolve_cv_init_count(
        {'normalize_input_init_count': 81920},
        {'normalize_input_init_count': 123}) == 123
    # nothing explicit -> None -> CentralValueTrain derives its own default
    assert resolve_cv_init_count({}, {}) is None


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
