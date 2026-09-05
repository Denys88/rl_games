"""Tests for the Craftax vec env (JAX, 3.11 venv) and the end-to-end
distillation plumbing through the PPO agent."""

import numpy as np
import pytest
import torch


def _has_craftax():
    try:
        import craftax, jax  # noqa: F401
        return True
    except Exception:
        return False


needs_craftax = pytest.mark.skipif(not _has_craftax(), reason='craftax/jax unavailable')


def _make(n=8, **kw):
    from rl_games.envs.craftax_vecenv import CraftaxVecEnv
    cfg = dict(game='Craftax-Classic-v1', obs='both', seed=1, device='cpu')
    cfg.update(kw)
    return CraftaxVecEnv('craftax', n, **cfg)


@needs_craftax
def test_both_mode_shapes_and_env_info():
    env = _make(8)
    info = env.get_env_info()
    assert info['observation_space'].shape == (63, 63, 3) and info['observation_space'].dtype == np.uint8
    assert info['state_space'].shape == (1345,)
    assert info['action_space'].n == 17 and info['agents'] == 1
    obs = env.reset()
    assert obs['obs'].shape == (8, 63, 63, 3) and obs['obs'].dtype == torch.uint8
    assert obs['states'].shape == (8, 1345) and obs['states'].dtype == torch.float32
    assert int(obs['obs'].max()) > 0
    o, r, d, inf = env.step(torch.randint(0, 17, (8,)))
    assert o['obs'].shape == (8, 63, 63, 3) and r.shape == (8,) and d.shape == (8,) and d.dtype == torch.bool
    assert r.dtype == torch.float32


@needs_craftax
def test_single_modes():
    env = _make(4, obs='symbolic')
    assert env.get_env_info()['observation_space'].shape == (1345,)
    o = env.reset()
    assert torch.is_tensor(o) and o.shape == (4, 1345)
    env = _make(4, obs='pixels')
    assert env.get_env_info()['observation_space'].shape == (63, 63, 3)
    o = env.reset()
    assert torch.is_tensor(o) and o.shape == (4, 63, 63, 3) and o.dtype == torch.uint8


@needs_craftax
def test_autoreset_and_achievements_info():
    env = _make(4, max_timesteps=30)          # short episodes -> dones within 30 steps
    env.reset()
    seen = False
    for t in range(40):
        o, r, d, info = env.step(torch.randint(0, 17, (4,)))
        assert o['obs'].shape == (4, 63, 63, 3)
        if d.any():
            seen = True
            assert info['achievements'].shape == (4, 22)
            assert torch.equal(info['done_mask'], d)
    assert seen


def test_craftax_observer_score():
    from rl_games.envs.craftax_vecenv import CraftaxObserver, crafter_score
    rates = np.zeros(22)
    rates[:2] = 1.0            # two achievements always, rest never
    expected = np.exp(np.mean(np.log(1 + rates * 100))) - 1
    assert abs(crafter_score(rates) - expected) < 1e-9

    class _Algo:
        writer = None
        num_agents = 1

    obs = CraftaxObserver()
    obs.after_init(_Algo())
    ach = torch.zeros(4, 22)
    ach[0, 3] = 1
    ach[1, 3] = 1
    obs.process_infos({'achievements': ach, 'done_mask': torch.tensor([True, True, False, False])},
                      torch.tensor([[0], [1]]))
    assert len(obs.episodes) == 2 and abs(obs.rates()[3] - 1.0) < 1e-9 and obs.rates()[0] == 0.0


@needs_craftax
def test_distillation_end_to_end_two_epochs(tmp_path):
    """Symbolic teacher (untrained, saved to disk) -> pixel student with the
    distillation block; two PPO epochs must run and log losses/distill."""
    import yaml
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch.model_builder import ModelBuilder
    tcfg = yaml.safe_load(open('rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml'))
    net = ModelBuilder().load(tcfg['params'])
    c = tcfg['params']['config']
    teacher = net.build({'actions_num': 17, 'input_shape': (1345,), 'num_seqs': 1, 'value_size': 1,
                         'normalize_value': c['normalize_value'], 'normalize_input': c['normalize_input']})
    ck = tmp_path / 'teacher.pth'
    torch.save({'model': teacher.state_dict(), 'epoch': 0}, ck)
    scfg = yaml.safe_load(open('rl_games/configs/craftax/ppo_craftax_classic_pixels_distill.yaml'))
    sc = scfg['params']['config']
    sc.update(num_actors=16, horizon_length=8, minibatch_size=64, max_epochs=2, save_frequency=0,
              train_dir=str(tmp_path), name='distill_smoke', device='cpu')
    sc['central_value_config']['minibatch_size'] = 64
    sc['distillation'].update(teacher_checkpoint=str(ck), beta=0.5)
    sc['env_config'].update(device='cpu')
    runner = Runner()
    runner.load(scfg)
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    assert agent.distill is not None and agent.store_states
    agent.train()
    assert 'distill' in agent.aux_loss_dict
