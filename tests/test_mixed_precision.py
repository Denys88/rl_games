"""mixed_precision: TF32 by default, fp16 with loss scaling, bf16 on request.

bf16 rounding of the policy mean adds a KL of 0.01-0.03 per update and noise
in the PPO ratio at small sigma, independent of the learning rate, so the
default is no autocast. The default tests pretend a bf16-capable GPU is
present, because on CPU the removed bf16 default was already off.
"""
import pytest
import torch

from rl_games.algos_torch import torch_ext
from tests.test_ppo_masking import make_ppo_agent
from tests.test_sac_correctness import make_fake_env_sac_agent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


class _OnCuda:
    """The PPO test env returns CPU tensors; a CUDA agent expects them on its device."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset().cuda()

    def step(self, actions):
        obs, rew, dones, infos = self.env.step(actions)
        return obs.cuda(), rew.cuda(), dones.cuda(), infos

    def set_train_info(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_cuda_ppo_agent(**config):
    agent, fake = make_ppo_agent(device='cuda:0', **config)
    agent.vec_env = _OnCuda(agent.vec_env)
    return agent, fake


@pytest.fixture
def bf16_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda *a, **k: True)


def test_ppo_default_is_off(bf16_gpu):
    agent, _ = make_ppo_agent(mixed_precision=None)
    assert agent.mixed_precision is False
    assert not agent.scaler.is_enabled()


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


@pytest.mark.parametrize('value, dtype', [
    (False, None), (None, None), ('tf32', None), ('off', None),
    ('fp16', torch.float16), ('FP16', torch.float16), ('half', torch.float16),
    ('bf16', torch.bfloat16), ('bfloat16', torch.bfloat16),
])
def test_resolve_values(value, dtype):
    assert torch_ext.resolve_mixed_precision(value, 'cuda:0') is dtype


def test_resolve_true_is_bf16_with_warning():
    with pytest.warns(UserWarning, match='fp16'):
        assert torch_ext.resolve_mixed_precision(True, 'cuda:0') is torch.bfloat16


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError):
        torch_ext.resolve_mixed_precision('fp8', 'cuda:0')


def test_resolve_half_on_cpu_falls_back():
    with pytest.warns(UserWarning, match='CUDA'):
        assert torch_ext.resolve_mixed_precision('fp16', 'cpu') is None


@cuda
def test_ppo_fp16_trains_with_loss_scaling():
    agent, _ = make_cuda_ppo_agent(mixed_precision='fp16', truncate_grads=True, max_epochs=4, mini_epochs=2)
    assert agent.amp_dtype is torch.float16 and agent.scaler.is_enabled()
    before = [p.detach().clone() for p in agent.model.parameters()]
    agent.train()
    after = list(agent.model.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(a, b) for a, b in zip(after, before))
    # the first steps overflow at the initial scale 2**16 and are skipped
    assert 0 < agent.scaler.get_scale() < 2.0 ** 16

    state = agent.get_full_state_weights()
    assert 'scaler' in state
    fresh, _ = make_cuda_ppo_agent(mixed_precision='fp16')
    fresh.set_full_state_weights(state)
    assert fresh.scaler.get_scale() == agent.scaler.get_scale()


@cuda
def test_skipped_fp16_step_does_not_raise_the_rate():
    agent, _ = make_cuda_ppo_agent(mixed_precision='fp16', max_epochs=1, lr_schedule='adaptive',
                                   schedule_type='per_minibatch', learning_rate=1e-4)
    agent.train()
    # one minibatch, skipped at scale 2**16: the rate must stay put
    assert agent.step_skipped
    assert agent.skipped_steps in (0, 1)  # reset when the epoch's stats are written
    assert agent.last_lr == pytest.approx(1e-4)


@cuda
@pytest.mark.parametrize('precision', ['fp16', 'bf16'])
def test_rollout_and_update_score_actions_alike(precision):
    # both passes run under the same autocast, so before any update the PPO
    # ratio is 1 up to kernel differences between batch shapes
    agent, _ = make_cuda_ppo_agent(mixed_precision=precision)
    obs = agent.obs_to_tensors(agent.env_reset())
    res = agent.get_action_values(obs)
    agent.model.train()
    with torch.no_grad(), torch_ext.autocast(agent.amp_dtype):
        train = agent.model({'is_train': True, 'prev_actions': res['actions'],
                             'obs': agent._preproc_obs(obs['obs'])})
    logratio = res['neglogpacs'] - train['prev_neglogp'].float()
    assert logratio.abs().max().item() < 1e-2


@cuda
def test_sac_fp16_trains_with_loss_scaling():
    agent, _ = make_fake_env_sac_agent(device='cuda:0', mixed_precision='fp16')
    assert agent.critic_scaler.is_enabled() and agent.actor_scaler.is_enabled()
    agent.train()
    assert all(torch.isfinite(p).all() for p in agent.model.parameters())
    state = agent.get_full_state_weights()
    assert 'critic_scaler' in state and 'actor_scaler' in state
