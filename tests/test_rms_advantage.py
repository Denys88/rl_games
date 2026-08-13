"""RMS advantage normalization (normalize_rms_advantage) revival tests.

The feature was dead since #315: the first statistics update assigned plain
tensors to nn.Parameter attributes (TypeError), the mask argument was
silently ignored, and PpoDiagnostics referenced attribute names from the
pre-#231 predecessor class. Three stacked bugs, each hiding the next.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.common.diagnostics import PpoDiagnostics


def test_first_update_does_not_crash_and_converges():
    torch.manual_seed(0)
    for impl in ('mean_std', 'mean_std_corr'):
        gms = GeneralizedMovingStats((1,), impl=impl, decay=0.9)
        gms.train()
        for _ in range(300):
            gms(torch.randn(256) * 2.0 + 3.0)     # the #315 crash site
        mean, std = gms.get_mean_std()
        assert abs(mean.item() - 3.0) < 0.2, (impl, mean)
        assert abs(std.item() - 2.0) < 0.2, (impl, std)


def test_eval_mode_normalizes_without_updating():
    gms = GeneralizedMovingStats((1,), decay=0.9)
    gms.train()
    gms(torch.randn(512) + 5.0)
    state = {k: v.clone() for k, v in gms.state_dict().items()}
    gms.eval()
    out = gms(torch.randn(64) + 5.0)
    assert torch.isfinite(out).all()
    for k, v in gms.state_dict().items():
        assert torch.equal(v, state[k]), k


def test_mask_excludes_rows_bit_identically():
    torch.manual_seed(1)
    valid = torch.randn(300) + 1.0
    poison = torch.full((100,), 1e6)
    x = torch.cat([valid, poison])
    mask = torch.cat([torch.ones(300), torch.zeros(100)])

    a = GeneralizedMovingStats((1,), decay=0.9)
    b = GeneralizedMovingStats((1,), decay=0.9)
    a.train(); b.train()
    a(valid)
    b(x, mask=mask)
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), k
    # mask=None behaves as all-valid
    c = GeneralizedMovingStats((1,), decay=0.9)
    c.train()
    c(x)
    assert not torch.equal(a.state_dict()['mean'], c.state_dict()['mean'])


def test_all_masked_batch_is_a_noop():
    gms = GeneralizedMovingStats((1,), decay=0.9)
    gms.train()
    gms(torch.randn(64))
    state = {k: v.clone() for k, v in gms.state_dict().items()}
    gms(torch.full((32,), 1e9), mask=torch.zeros(32))
    for k, v in gms.state_dict().items():
        assert torch.equal(v, state[k]), k


def test_state_dict_keys_stable_for_old_checkpoints():
    # stats moved from requires_grad=False Parameters to buffers; the keys
    # must not change or existing checkpoints break
    gms = GeneralizedMovingStats((1,))
    assert set(gms.state_dict().keys()) == {'step', 'mean', 'sqrs'}
    fresh = GeneralizedMovingStats((1,))
    fresh.load_state_dict(gms.state_dict())


def test_diagnostics_reads_live_attributes():
    class FakeAgent:
        normalize_rms_advantage = True
        normalize_value = False
        advantage_mean_std = GeneralizedMovingStats((1,), decay=0.9)
    FakeAgent.advantage_mean_std.train()
    FakeAgent.advantage_mean_std(torch.randn(128) + 2.0)
    d = PpoDiagnostics()
    d.exp_vars = [torch.tensor(0.5)]
    d.epoch(FakeAgent, current_epoch=1)
    m = d.diag_dict['diagnostics/rms_advantage/mean']
    v = d.diag_dict['diagnostics/rms_advantage/var']
    assert torch.isfinite(m).all() and torch.isfinite(v).all()
    assert v.item() > 0


def test_agent_level_rms_advantage_full_epoch():
    """The original crash fired inside prepare_dataset on the first epoch --
    run a real (tiny, CPU) PPO epoch with normalize_rms_advantage on."""
    from rl_games.torch_runner import Runner
    from tests.test_sac_correctness import FakeNextStepVecEnv, OBS_DIM

    NUM_ENVS, HORIZON = 4, 16

    class TorchEnvAdapter:
        def __init__(self, fake):
            self.fake = fake

        def reset(self):
            return torch.from_numpy(self.fake.reset())

        def step(self, actions):
            obs, rew, dones, infos = self.fake.step(actions.detach().cpu().numpy())
            return (torch.from_numpy(obs), torch.from_numpy(rew),
                    torch.from_numpy(dones), infos)

        def get_env_info(self):
            return self.fake.get_env_info()

        def get_env_state(self):
            return None

        def set_env_state(self, state):
            pass

        def set_train_info(self, frame, agent):
            pass

    torch.manual_seed(3)
    np.random.seed(3)
    fake = FakeNextStepVecEnv(NUM_ENVS)
    env = TorchEnvAdapter(fake)
    params = {
        'algo': {'name': 'a2c_continuous'},
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {
            'name': 'actor_critic', 'separate': False,
            'space': {'continuous': {
                'mu_activation': 'None', 'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'const_initializer', 'val': 0.0},
                'fixed_sigma': True}},
            'mlp': {'units': [16], 'activation': 'elu',
                    'initializer': {'name': 'default'}},
        },
        'config': {
            'name': 'rms_adv_epoch', 'env_name': 'unused',
            'reward_shaper': {'scale_value': 1.0},
            'device': 'cpu', 'multi_gpu': False, 'mixed_precision': False,
            'normalize_input': False, 'normalize_value': False,
            'normalize_advantage': True, 'normalize_rms_advantage': True,
            'adv_rms_momentum': 0.9, 'value_bootstrap': False,
            'num_actors': NUM_ENVS, 'horizon_length': HORIZON,
            'minibatch_size': NUM_ENVS * HORIZON, 'mini_epochs': 1,
            'learning_rate': 1e-4, 'lr_schedule': None, 'kl_threshold': 0.008,
            'gamma': 0.99, 'tau': 0.95, 'e_clip': 0.2, 'clip_value': False,
            'critic_coef': 1.0, 'entropy_coef': 0.0, 'truncate_grads': False,
            'grad_norm': 1.0, 'max_epochs': 1, 'save_frequency': 0,
            # master's bound_loss returns int 0 without the coef and the loss
            # assembly crashes on it (latent bug, fixed on the B1 branch);
            # set it like every shipped continuous config does
            'bounds_loss_coef': 0.0001,
            'save_best_after': 10_000, 'print_stats': False,
            'env_info': fake.get_env_info(),
        },
    }
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(runner.algo_name, base_name='rms_adv_epoch',
                                       params=runner.params)
    assert agent.normalize_rms_advantage
    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.train_epoch()                       # crashed with TypeError pre-fix
    assert agent.advantage_mean_std.step.item() > 1
    mean, std = agent.advantage_mean_std.get_mean_std()
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
