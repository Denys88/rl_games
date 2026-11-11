"""RunningMeanStdObs must survive torch.jit.script -- models.py wraps it
unconditionally for dict observation spaces (regression: upstream #315
scripted the container whose forward lacked the Dict annotation, so
dict-obs + normalize_input crashed at model construction)."""

import torch

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs


def _obs(n=64):
    return {'proprio': torch.randn(n, 4) + 3, 'camera': torch.randn(n, 8) * 2 - 1}


def test_scripted_construction_matches_models_py_usage():
    m = torch.jit.script(RunningMeanStdObs({'proprio': (4,), 'camera': (8,)}))
    out = m(_obs())
    assert set(out) == {'proprio', 'camera'}
    assert out['proprio'].shape == (64, 4) and out['camera'].shape == (64, 8)


def test_scripted_matches_eager_stats_and_output():
    torch.manual_seed(4)
    data = _obs(512)
    eager = RunningMeanStdObs({'proprio': (4,), 'camera': (8,)})
    scripted = torch.jit.script(RunningMeanStdObs({'proprio': (4,), 'camera': (8,)}))
    for m in (eager, scripted):
        m.train()
        m(data)
        m.eval()
    for k in data:
        e = getattr(eager.running_mean_std, k)
        s = getattr(scripted.running_mean_std, k)
        assert torch.allclose(e.running_mean, s.running_mean, atol=0, rtol=0)
        assert torch.allclose(e.running_var, s.running_var, atol=0, rtol=0)
        out_e, out_s = eager(data)[k], scripted(data)[k]
        assert torch.allclose(out_e, out_s, atol=1e-6)
    # normalization actually happened
    normed = eager(data)['proprio']
    assert abs(normed.mean().item()) < 0.2 and abs(normed.std().item() - 1.0) < 0.2


def test_scripted_denorm_roundtrip():
    torch.manual_seed(9)
    m = torch.jit.script(RunningMeanStdObs({'a': (3,)}))
    m.train()
    x = {'a': torch.randn(256, 3) * 5 + 2}
    m(x)
    m.eval()
    y = m(x)
    back = m(y, denorm=True)
    assert torch.allclose(back['a'], x['a'], atol=1e-4)


def test_key_mismatch_fails_loudly():
    import pytest
    eager = RunningMeanStdObs({'a': (3,)})
    scripted = torch.jit.script(RunningMeanStdObs({'a': (3,)}))
    good = {'a': torch.randn(8, 3)}
    extra = {'a': torch.randn(8, 3), 'stray': torch.randn(8, 2)}
    missing = {}
    for m in (eager, scripted):
        m(good)   # sanity: matching keys pass
        with pytest.raises(Exception, match='normalizer keys'):
            m(extra)
        with pytest.raises(Exception):
            m(missing)
