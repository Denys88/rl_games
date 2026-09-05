"""Tests for the mujoco_playground vec env (JAX + Warp physics, built-in
batch renderer). Needs the 3.11 venv with playground, warp-lang 1.16 and,
under WSL2, the CUDA driver from /usr/lib/wsl/lib (preloaded by the env)."""

import numpy as np
import pytest
import torch


def _playground_ready():
    """playground importable AND Warp sees a CUDA device (under WSL2 that needs
    LD_LIBRARY_PATH=/usr/lib/wsl/lib so the driver's libcuda wins over the stub)."""
    try:
        import mujoco_playground, jax  # noqa: F401
        import warp as wp
        wp.init()
        return len(wp.get_cuda_devices()) > 0
    except Exception:
        return False


needs_playground = pytest.mark.skipif(not _playground_ready(),
                                      reason='mujoco_playground unavailable or Warp has no CUDA device')


def _make(n=8, **kw):
    from rl_games.envs.playground_vecenv import PlaygroundVecEnv
    cfg = dict(env_name='CartpoleBalance', obs='state', seed=1, device='cpu')
    cfg.update(kw)
    return PlaygroundVecEnv('playground', n, **cfg)


@needs_playground
def test_state_mode_shapes_and_time_outs():
    env = _make(8, config_overrides={'episode_length': 20})
    info = env.get_env_info()
    assert info['action_space'].shape == (1,) and info['agents'] == 1
    assert info['observation_space'].shape == info['state_space'].shape == (5,)
    obs = env.reset()
    assert torch.is_tensor(obs) and obs.shape == (8, 5) and obs.dtype == torch.float32
    saw_timeout = False
    for t in range(25):
        obs, r, d, inf = env.step(torch.zeros(8, 1))
        assert obs.shape == (8, 5) and r.shape == (8,) and r.dtype == torch.float32 and d.dtype == torch.bool
        assert inf['time_outs'].shape == (8,)
        if d.any():
            saw_timeout = True
            assert torch.equal(inf['time_outs'], d)          # balance task only ends by time limit
            assert t == 19
    assert saw_timeout


@needs_playground
def test_pixels_and_both_modes():
    env = _make(4, obs='pixels', cam_res=(32, 32))
    assert env.get_env_info()['observation_space'].shape == (32, 32, 3)
    obs = env.reset()
    assert obs.shape == (4, 32, 32, 3) and obs.dtype == torch.uint8 and int(obs.max()) > 0
    env = _make(4, obs='both', cam_res=(32, 32))
    info = env.get_env_info()
    assert info['observation_space'].shape == (32, 32, 3) and info['state_space'].shape == (5,)
    obs = env.reset()
    assert obs['obs'].shape == (4, 32, 32, 3) and obs['states'].shape == (4, 5)
    o, r, d, inf = env.step(torch.zeros(4, 1))
    assert o['obs'].shape == (4, 32, 32, 3) and o['states'].dtype == torch.float32


@needs_playground
def test_metrics_reported_on_done_and_observer():
    from rl_games.envs.playground_vecenv import PlaygroundObserver
    env = _make(4, config_overrides={'episode_length': 10})
    env.reset()
    got = None
    for _ in range(12):
        _, _, d, inf = env.step(torch.zeros(4, 1))
        if d.any():
            got = inf
            break
    assert got is not None and 'metrics' in got and 'done_mask' in got
    assert all(v.shape == (4,) for v in got['metrics'].values())

    class _Algo:
        writer = None
        num_agents = 1

    obs = PlaygroundObserver()
    obs.after_init(_Algo())
    obs.process_infos(got, torch.nonzero(d)[:, :1])
    assert len(obs.episodes) == int(d.sum())
    means = obs.means()
    assert set(means) == set(got['metrics'])
