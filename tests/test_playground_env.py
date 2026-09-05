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


@needs_playground
def test_continuous_distillation_end_to_end_cartpole(tmp_path):
    """Continuous-action distillation plumbing: untrained state teacher ->
    pixel student on CartpoleBalance (fast vision env), two PPO epochs."""
    import yaml
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch.model_builder import ModelBuilder
    tcfg = yaml.safe_load(open('rl_games/configs/playground/ppo_panda_pick_state.yaml'))
    tcfg['params']['config']['env_config'].update(env_name='CartpoleBalance')
    tpath = tmp_path / 'teacher_cfg.yaml'
    yaml.safe_dump(tcfg, open(tpath, 'w'))
    c = tcfg['params']['config']
    teacher = ModelBuilder().load(tcfg['params']).build(
        {'actions_num': 1, 'input_shape': (5,), 'num_seqs': 1, 'value_size': 1,
         'normalize_value': c['normalize_value'], 'normalize_input': c['normalize_input']})
    ck = tmp_path / 'teacher.pth'
    torch.save({'model': teacher.state_dict(), 'epoch': 0}, ck)
    scfg = yaml.safe_load(open('rl_games/configs/playground/ppo_panda_pick_pixels_distill.yaml'))
    sc = scfg['params']['config']
    sc.update(num_actors=16, horizon_length=8, minibatch_size=64, max_epochs=2, save_frequency=0,
              train_dir=str(tmp_path), name='pg_distill_smoke')
    sc['central_value_config']['minibatch_size'] = 64
    sc['distillation'].update(teacher_config=str(tpath), teacher_checkpoint=str(ck), beta=0.5)
    sc['env_config'].update(env_name='CartpoleBalance', cam_res=[32, 32], config_overrides={'episode_length': 20})
    runner = Runner()
    runner.load(scfg)
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    assert agent.distill is not None and not agent.distill.is_discrete
    agent.train()
    assert 'distill' in agent.aux_loss_dict
