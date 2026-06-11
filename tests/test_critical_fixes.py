"""Regression tests for the 2026-06-10 review critical-bug batch (docs/reviews/2026-06-10/02-bugs.md)."""
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_help_works_without_distutils():
    # Finding P5: `from distutils.util import strtobool` crashes on py3.12+ (distutils removed).
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
    # Finding 6 (high): np.max(a, b) passes b as the AXIS argument.
    # Pre-fix: any positive max_steps raises AxisError during agent __init__.
    agent = make_cartpole_agent(max_steps=500_000)
    assert agent.max_frames == 500_000


def test_max_frames_and_max_steps_take_elementwise_max():
    agent = make_cartpole_agent(max_frames=100, max_steps=200)
    assert agent.max_frames == 200


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
