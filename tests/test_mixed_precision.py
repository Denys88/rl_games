"""mixed_precision: TF32 by default, fp16 with loss scaling, bf16 on request.

bf16 rounding of the policy mean adds a KL of 0.01-0.03 per update and noise
in the PPO ratio at small sigma, independent of the learning rate, so the
default is no autocast. The default tests pretend a bf16-capable GPU is
present, because on CPU the removed bf16 default was already off.
"""
import pytest
import torch

from rl_games.algos_torch import torch_ext
from tests.test_critical_fixes import make_cartpole_agent
from tests.test_ppo_masking import HORIZON, NUM_ENVS, make_ppo_agent, _rollout_batch
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


@pytest.mark.parametrize('value', [True, 1, 'true', 'True', '1'])
def test_resolve_true_is_fp16(value, recwarn):
    assert torch_ext.resolve_mixed_precision(value, 'cuda:0') is torch.float16
    assert not recwarn.list


@pytest.mark.parametrize('value', [0, '0', 'False', 'FALSE'])
def test_resolve_false_strings_and_zero(value):
    assert torch_ext.resolve_mixed_precision(value, 'cuda:0') is None


def test_resolve_rejects_unknown():
    for value in ('fp8', 2, 0.5):
        with pytest.raises(ValueError):
            torch_ext.resolve_mixed_precision(value, 'cuda:0')


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
    assert 0 < agent.scaler.get_scale() <= 2.0 ** 16

    state = agent.get_full_state_weights()
    assert 'scaler' in state
    fresh, _ = make_cuda_ppo_agent(mixed_precision='fp16')
    fresh.set_full_state_weights(state)
    assert fresh.scaler.get_scale() == agent.scaler.get_scale()


class _SkippingScaler:
    """GradScaler stand-in that runs on CPU and skips the listed optimizer steps."""

    def __init__(self, skip):
        self.skip, self.calls, self._scale, self._skipping = set(skip), 0, 2.0 ** 16, False

    def is_enabled(self):
        return True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self._skipping = self.calls in self.skip
        self.calls += 1
        if not self._skipping:
            optimizer.step()

    def update(self):
        if self._skipping:
            self._scale /= 2

    def get_scale(self):
        return self._scale


def _with_skipping_scaler(agent, skip):
    agent.scaler = _SkippingScaler(skip)
    agent._last_scale = agent.scaler.get_scale()
    agent.vec_env.set_train_info = lambda *args: None
    calls = []
    update = agent.scheduler.update

    def record(lr, entropy, epoch, frames, kl, **kwargs):
        calls.append(kl)
        return update(lr, entropy, epoch, frames, kl, **kwargs)

    agent.scheduler.update = record
    return calls


@pytest.mark.parametrize('schedule_type', ['per_minibatch', 'standard'])
def test_skipped_step_does_not_drive_the_continuous_rate(schedule_type):
    # one minibatch per mini-epoch: skipping step 0 leaves mini-epoch 0 without
    # a step, so it must not update the rate under either schedule
    agent, _ = make_ppo_agent(lr_schedule='adaptive', kl_threshold=0.008, max_lr=1e-3,
                              schedule_type=schedule_type, mini_epochs=3)
    _rollout_batch(agent)
    calls = _with_skipping_scaler(agent, skip={0})
    agent.train_epoch()
    assert agent.scaler.calls == 3
    assert len(calls) == 2
    assert agent.skipped_steps == 1


def test_standard_schedule_averages_only_the_steps_taken():
    # each minibatch's KL is measured before its own step: step 0 moves the
    # policy, so minibatch 1 reports a KL above zero; its step is skipped
    agent, _ = make_ppo_agent(lr_schedule='adaptive', kl_threshold=0.008, max_lr=1e-2,
                              learning_rate=1e-2, schedule_type='standard', mini_epochs=1,
                              minibatch_size=NUM_ENVS * HORIZON // 2)
    _rollout_batch(agent)
    calls = _with_skipping_scaler(agent, skip={1})
    kls = []
    train = agent.train_actor_critic

    def record_train(batch):
        result = train(batch)
        kls.append(result[3].item())
        return result

    agent.train_actor_critic = record_train
    agent.train_epoch()
    assert len(kls) == 2 and len(calls) == 1
    assert kls[1] > kls[0] + 1e-6
    assert calls[0] == pytest.approx(kls[0])


def test_skipped_step_does_not_drive_the_discrete_rate():
    agent = make_cartpole_agent(lr_schedule='adaptive', kl_threshold=0.008, max_lr=1e-3,
                                num_actors=2, horizon_length=8, minibatch_size=16, mini_epochs=3)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    calls = _with_skipping_scaler(agent, skip={0})
    agent.train_epoch()
    assert agent.scaler.calls == 3
    assert len(calls) == 2


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
