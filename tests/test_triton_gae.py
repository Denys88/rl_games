"""Correctness tests for the GAE implementations (PyTorch loop and Triton kernel).

The Triton tests require CUDA and an importable triton; they are skipped
otherwise (e.g. on CPU-only CI). The PyTorch-path tests always run.
"""

import os
import subprocess
import sys

import pytest
import torch

from rl_games.triton_config import TRITON_AVAILABLE
from rl_games.triton_kernels.gae_kernel import compute_gae, _pytorch_gae

TRITON_CUDA = TRITON_AVAILABLE and torch.cuda.is_available()


def reference_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau):
    """Independent scalar reference: per-(env, value) backward recursion."""
    horizon, num_envs, value_size = mb_rewards.shape
    advs = torch.zeros_like(mb_rewards, dtype=torch.float64)
    r = mb_rewards.double()
    v = mb_values.double()
    d = mb_dones.double()
    lv = last_values.double()
    ld = last_dones.double()
    for e in range(num_envs):
        for k in range(value_size):
            lastgaelam = 0.0
            for t in reversed(range(horizon)):
                if t == horizon - 1:
                    nextvalue = lv[e, k]
                    nextnonterminal = 1.0 - ld[e]
                else:
                    nextvalue = v[t + 1, e, k]
                    nextnonterminal = 1.0 - d[t + 1, e]
                delta = r[t, e, k] + gamma * nextvalue * nextnonterminal - v[t, e, k]
                lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam
                advs[t, e, k] = lastgaelam
    return advs.to(mb_rewards.dtype)


def make_inputs(horizon, num_envs, value_size, device='cpu', seed=0):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(horizon, num_envs, value_size, generator=g)
    values = torch.randn(horizon, num_envs, value_size, generator=g)
    dones = (torch.rand(horizon, num_envs, generator=g) < 0.15).float()
    last_values = torch.randn(num_envs, value_size, generator=g)
    last_dones = (torch.rand(num_envs, generator=g) < 0.15).float()
    return tuple(t.to(device) for t in (rewards, values, dones, last_values, last_dones))


@pytest.mark.parametrize('shape', [(8, 4, 1), (16, 6, 3)])
@pytest.mark.parametrize('gamma,tau', [(0.99, 0.95), (1.0, 1.0)])
def test_pytorch_gae_matches_reference(shape, gamma, tau):
    inputs = make_inputs(*shape)
    out = _pytorch_gae(*inputs, gamma, tau)
    ref = reference_gae(*inputs, gamma, tau)
    assert torch.allclose(out, ref, atol=1e-5)


def test_compute_gae_cpu_uses_pytorch_path():
    inputs = make_inputs(8, 4, 1)
    out = compute_gae(*inputs, 0.99, 0.95)
    ref = _pytorch_gae(*inputs, 0.99, 0.95)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not TRITON_CUDA, reason='requires CUDA and triton')
@pytest.mark.parametrize('shape', [(8, 16, 1), (36, 64, 3), (200, 128, 1)])
@pytest.mark.parametrize('gamma,tau', [(0.99, 0.95), (1.0, 1.0)])
def test_triton_matches_pytorch(shape, gamma, tau):
    from rl_games.triton_kernels.gae_kernel import _triton_gae
    inputs = make_inputs(*shape, device='cuda')
    out = _triton_gae(*inputs, gamma, tau)
    ref = _pytorch_gae(*inputs, gamma, tau)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()


@pytest.mark.skipif(not TRITON_CUDA, reason='requires CUDA and triton')
def test_triton_handles_view_inputs():
    """Non-contiguous inputs (transposes, slices) must not silently corrupt."""
    from rl_games.triton_kernels.gae_kernel import _triton_gae
    horizon, num_envs, value_size = 12, 8, 2
    rewards, values, dones, last_values, last_dones = make_inputs(
        horizon, num_envs, value_size + 1, device='cuda')
    # Last-dim slices and transposed views exercise every stride assumption.
    rewards = rewards[:, :, :value_size]
    values = values[:, :, :value_size]
    last_values = last_values[:, :value_size]
    dones_t = dones.t().contiguous().t()
    assert not rewards.is_contiguous()
    out = _triton_gae(rewards, values, dones_t, last_values, last_dones, 0.99, 0.95)
    ref = _pytorch_gae(rewards.contiguous(), values.contiguous(), dones,
                       last_values.contiguous(), last_dones, 0.99, 0.95)
    assert torch.allclose(out, ref, atol=1e-4)


@pytest.mark.skipif(not TRITON_CUDA, reason='requires CUDA and triton')
def test_triton_all_done_and_no_done_edges():
    from rl_games.triton_kernels.gae_kernel import _triton_gae
    rewards, values, dones, last_values, last_dones = make_inputs(10, 4, 1, device='cuda')
    for fill in (0.0, 1.0):
        d = torch.full_like(dones, fill)
        ld = torch.full_like(last_dones, fill)
        out = _triton_gae(rewards, values, d, last_values, ld, 0.99, 0.95)
        ref = _pytorch_gae(rewards, values, d, last_values, ld, 0.99, 0.95)
        assert torch.allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize('value', ['1', 'true', 'yes', 'True', ' 1 ', ''])
def test_rlg_no_triton_env_values_do_not_crash(value):
    """Any plausible RLG_NO_TRITON value must not break `import rl_games`."""
    env = dict(os.environ, RLG_NO_TRITON=value)
    result = subprocess.run(
        [sys.executable, '-c',
         'import rl_games.triton_config as tc; print(tc.USE_TRITON)'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if value.strip().lower() in ('1', 'true', 'yes'):
        assert result.stdout.strip() == 'False'
