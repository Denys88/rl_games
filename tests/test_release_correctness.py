"""Regression tests for the release-preparation correctness fixes:
central-value normalizer prior geometry and the player's vecenv fallback
for creator-less registrations."""

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


def _capture_cv_init_count(cv_mini_epochs, horizon, num_actors, explicit=None):
    from rl_games.algos_torch.central_value import CentralValueTrain

    net = _CaptureNetwork()
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
        'multi_gpu_grad_sync': 'allreduce',
    }
    with pytest.raises(_Captured):
        CentralValueTrain(
            state_shape=(4,), value_size=1, ppo_device='cpu', num_agents=1,
            horizon_length=horizon, num_actors=num_actors, num_actions=2,
            seq_length=4, normalize_value=False, network=net, config=config,
            writter=None, max_epochs=1, multi_gpu=False, zero_rnn_on_done=True,
            normalize_input_init_count=explicit)
    return net.state_config['normalize_input_init_count']


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
