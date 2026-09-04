"""Regression tests for a batch of critical fixes: runner CLI on py3.12+,
max_frames/max_steps handling, LR scheduling, and checkpoint save/restore round-trips."""
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_help_works_without_distutils():
    # Regression: `from distutils.util import strtobool` crashes on py3.12+ (distutils removed).
    # '--help' exercises the import-time crash point (the old line-1 distutils import).
    res = subprocess.run([sys.executable, os.path.join(REPO, 'runner.py'), '--help'],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert 'distutils' not in res.stderr
    assert 'usage' in res.stdout.lower()


def test_strtobool_semantics():
    # Direct unit test: runner.py is import-safe (all side effects live under __main__).
    sys.path.insert(0, REPO)
    try:
        from runner import strtobool
    finally:
        sys.path.remove(REPO)
    for token in ('y', 'yes', 't', 'true', 'on', '1', 'YES', ' True '):
        assert strtobool(token) == 1
    for token in ('n', 'no', 'f', 'false', 'off', '0', 'OFF'):
        assert strtobool(token) == 0
    with pytest.raises(ValueError):
        strtobool('maybe')


CARTPOLE_YAML = os.path.join(REPO, 'rl_games', 'configs', 'ppo_cartpole.yaml')
SAC_YAML = os.path.join(REPO, 'rl_games', 'configs', 'mujoco', 'sac_halfcheetah.yaml')

# Agents create runs/<name>_<timestamp>/ at construction; keep test debris out of the real runs/.
TEST_TRAIN_DIR = tempfile.mkdtemp(prefix='rl_games_test_runs_')


def _load_params(path):
    with open(path) as f:
        return yaml.safe_load(f)['params']


def make_cartpole_runner(**config_overrides):
    """Tiny CPU CartPole PPO (discrete A2C) through the real Runner."""
    from rl_games.torch_runner import Runner
    params = _load_params(CARTPOLE_YAML)
    cfg = params['config']
    cfg.update({
        'device': 'cpu', 'device_name': 'cpu', 'multi_gpu': False,
        'num_actors': 2, 'horizon_length': 8, 'minibatch_size': 16,
        'mini_epochs': 1, 'max_epochs': 2, 'save_frequency': 0,
        'save_best_after': 10_000, 'torch_compile': False, 'print_stats': False,
        'train_dir': TEST_TRAIN_DIR, 'name': 'pytest_cartpole',
        'env_config': {'use_async': False},
    })
    cfg.update(config_overrides)
    params.setdefault('seed', 7)
    runner = Runner()
    runner.load({'params': params})
    return runner


def make_cartpole_agent(**config_overrides):
    runner = make_cartpole_runner(**config_overrides)
    return runner.algo_factory.create(runner.algo_name, base_name='test_run', params=runner.params)


def test_max_steps_config_usable():
    # Regression: np.max(a, b) passes b as the AXIS argument.
    # Pre-fix: any positive max_steps raises AxisError during agent __init__.
    agent = make_cartpole_agent(max_steps=500_000)
    assert agent.max_frames == 500_000


def test_max_frames_and_max_steps_take_elementwise_max():
    agent = make_cartpole_agent(max_frames=100, max_steps=200)
    assert agent.max_frames == 200


def test_linear_lr_schedule_anneals_with_max_frames():
    # Regression: all scheduler.update call sites hardcoded frames=0, so
    # lr_schedule: linear + max_frames never anneals (mul stays 1.0 forever).
    #
    # Setup notes:
    # - max_epochs must be -1: the LinearScheduler is only frames-based
    #   (use_epochs=False) when max_epochs == -1 (a2c_common.py scheduler setup).
    #   The run then terminates via the max_frames check instead.
    # - 2 actors * 8 horizon -> curr_frames=16 per epoch, so max_frames=2048
    #   means 128 tiny CPU epochs (seconds).
    #
    # Call-order: in the discrete train loop, scheduler.update runs INSIDE
    # train_epoch(), but self.frame += curr_frames happens in train() AFTER
    # train_epoch() returns. So the last scheduler call saw the frame count
    # from before the final epoch's increment: agent.frame - agent.curr_frames.
    runner = make_cartpole_runner(lr_schedule='linear', max_frames=2048,
                                  max_epochs=-1, learning_rate=3e-4)
    agent = runner.algo_factory.create(runner.algo_name, base_name='test_anneal', params=runner.params)
    agent.train()
    assert agent.frame > 0
    assert agent.last_lr < 3e-4, \
        f"lr did not anneal at all: last_lr={agent.last_lr}, frame={agent.frame}"
    frames_at_last_update = agent.frame - agent.curr_frames
    expected_mul = max(0, 2048 - frames_at_last_update) / 2048
    expected_lr = 1e-6 + (3e-4 - 1e-6) * expected_mul
    assert agent.last_lr == pytest.approx(expected_lr, rel=1e-4), \
        f"lr did not match linear formula: last_lr={agent.last_lr}, frame={agent.frame}"


def test_update_lr_sets_optimizer_and_last_lr_consistently():
    # Regression: train_actor_critic used to overwrite param_group['lr']
    # with rank-local stale self.last_lr after every minibatch; on non-zero ranks
    # that value is permanently stale (their scheduler is forced to Identity).
    # Contract now: update_lr is the single writer and keeps self.last_lr in sync.
    agent = make_cartpole_agent()
    agent.update_lr(1.23e-4)
    assert agent.last_lr == pytest.approx(1.23e-4)
    for group in agent.optimizer.param_groups:
        assert group['lr'] == pytest.approx(1.23e-4)
    # behavioral guard: train_actor_critic must not re-apply rank-stale last_lr
    # to the optimizer (the same bug, in any spelling or refactor)
    agent.last_lr = 999.0                  # poison the rank-stale value
    agent.calc_gradients = lambda d: None  # isolate the contract under test
    agent.train_result = (0, 0, 0, 0, agent.last_lr, 1.0)
    agent.train_actor_critic({})
    for group in agent.optimizer.param_groups:
        assert group['lr'] == pytest.approx(1.23e-4), \
            "train_actor_critic overwrote optimizer lr with rank-stale last_lr"


def test_rms_advantage_checkpoint_roundtrip():
    # Regression: set_stats_weights loads weights['advantage_mean_std']
    # unconditionally when normalize_rms_advantage is on, but get_stats_weights
    # never saved it -> restore crashes with KeyError.
    agent = make_cartpole_agent(normalize_advantage=True, normalize_rms_advantage=True)
    state = agent.get_full_state_weights()
    assert 'advantage_mean_std' in state
    agent.set_full_state_weights(state)  # pre-fix: KeyError 'advantage_mean_std'


def make_sac_pendulum_agent(**config_overrides):
    """Tiny CPU Pendulum SAC: halfcheetah config re-targeted at classic-control Pendulum (no mujoco dep)."""
    from rl_games.torch_runner import Runner
    params = _load_params(SAC_YAML)
    cfg = params['config']
    cfg.pop('env_config', None)  # drop HalfCheetah-specific env kwargs before re-targeting
    cfg.update({
        'device': 'cpu', 'device_name': 'cpu', 'multi_gpu': False,
        'env_name': 'Pendulum-v1', 'num_actors': 2, 'print_stats': False,
        'max_epochs': 2, 'save_frequency': 0, 'save_best_after': 10_000,
        'replay_buffer_size': 1000, 'batch_size': 32, 'num_warmup_frames': 1,
        'train_dir': TEST_TRAIN_DIR, 'name': 'pytest_sac_pendulum',
        'env_config': {'use_async': False},
    })
    cfg.update(config_overrides)
    runner = Runner()
    runner.load({'params': params})
    return runner.algo_factory.create(runner.algo_name, base_name='test_sac', params=runner.params)


def test_resume_preserves_best_reward_watermark():
    # get_full_state_weights saves last_mean_rewards precisely
    # "to prevent overriding the best ever checkpoint upon experiment restart",
    # but train() reset it to -1e9 right after restore.
    agent = make_cartpole_agent()
    state = agent.get_full_state_weights()
    state['last_mean_rewards'] = 1.0e6  # unbeatable by 2 epochs of cartpole
    agent.set_full_state_weights(state)
    agent.train()
    assert agent.last_mean_rewards == pytest.approx(1.0e6), \
        "train() reset the restored best-reward watermark"


def test_sac_checkpoint_contains_what_restore_reads():
    # SAC set_full_state_weights reads last_mean_rewards and
    # env_state, but get_full_state_weights never saved either.
    agent = make_sac_pendulum_agent()
    agent.last_mean_rewards = 42.0
    state = agent.get_full_state_weights()
    assert 'last_mean_rewards' in state and state['last_mean_rewards'] == pytest.approx(42.0)
    assert 'env_state' in state


def test_load_checkpoint_strips_compile_prefix_everywhere(tmp_path):
    # Regression: load_checkpoint stripped '_orig_mod.' only from
    # state['model']; compiled central-value nets save nested keys like
    # 'model._orig_mod.layer' under 'assymetric_vf_nets' and resume fails.
    from rl_games.algos_torch import torch_ext
    state = {
        'model': {'_orig_mod.actor.weight': torch.ones(2)},
        'assymetric_vf_nets': {'model._orig_mod.critic.weight': torch.ones(2),
                               'plain_key': torch.zeros(1)},
        'optimizer': {'state': {0: {'step': torch.tensor(1)}},
                      'param_groups': [{'lr': 1e-4, 'params': [0]}]},
        'frame': 128,
    }
    fn = str(tmp_path / 'ckpt')
    torch_ext.save_checkpoint(fn, state)
    loaded = torch_ext.load_checkpoint(fn + '.pth')
    assert 'actor.weight' in loaded['model']
    assert '_orig_mod.actor.weight' not in loaded['model']
    assert 'model.critic.weight' in loaded['assymetric_vf_nets']
    assert 'plain_key' in loaded['assymetric_vf_nets']
    assert loaded['optimizer']['state'][0]['step'] == torch.tensor(1)  # int keys survive
    assert loaded['frame'] == 128
