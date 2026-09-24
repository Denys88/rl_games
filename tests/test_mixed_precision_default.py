"""mixed_precision defaults to False even on GPUs with native bf16.

bf16 rounding of the policy mean is a KL of 0.01-0.03 per update at small
sigma, independent of the learning rate, so the adaptive schedule collapses
the rate to min_lr. The tests pretend a bf16-capable GPU is present, because
on CPU the removed bf16 default was already False.
"""
import pytest
import torch

from test_ppo_masking import make_ppo_agent
from test_sac_correctness import make_fake_env_sac_agent


@pytest.fixture
def bf16_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda *a, **k: True)


def test_ppo_default_is_off(bf16_gpu):
    agent, _ = make_ppo_agent(mixed_precision=None)
    assert agent.mixed_precision is False


def test_ppo_explicit_true_is_kept(bf16_gpu):
    agent, _ = make_ppo_agent(mixed_precision=True)
    assert agent.mixed_precision is True


def test_sac_default_is_off(bf16_gpu):
    agent, _ = make_fake_env_sac_agent(mixed_precision=None)
    assert agent.enable_mixed_precision is False


def test_central_value_default_is_off(bf16_gpu):
    import rl_games.envs  # noqa: F401  (registers 'testnet')
    from rl_games.algos_torch import model_builder
    from rl_games.algos_torch.central_value import CentralValueTrain

    network = model_builder.ModelBuilder().load(
        {'model': {'name': 'central_value'},
         'network': {'name': 'testnet', 'central_value': True}})
    config = {
        'mini_epochs': 1, 'normalize_input': False, 'learning_rate': 1e-3,
        'clip_value': False, 'lr_schedule': None, 'schedule_type': 'standard',
        'kl_threshold': 0.01, 'grad_norm': 1.0, 'truncate_grads': False,
        'minibatch_size': 32,
    }
    cv = CentralValueTrain(
        state_shape={'pos': (2,), 'info': (2,)}, value_size=1, ppo_device='cpu',
        num_agents=1, horizon_length=8, num_actors=4, num_actions=3,
        seq_length=4, normalize_value=False, network=network, config=config,
        writter=None, max_epochs=1, multi_gpu=False, zero_rnn_on_done=True)
    assert cv.mixed_precision is False
