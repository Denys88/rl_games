"""torch_compile is off unless a config asks for it."""
import torch

from tests.test_critical_fixes import make_cartpole_runner


def _count_compiles(monkeypatch):
    # the runner compiles modules; example networks may decorate a function at import
    calls = []

    def fake_compile(model, *args, **kwargs):
        if isinstance(model, torch.nn.Module):
            calls.append(model)
        return model

    monkeypatch.setattr(torch, 'compile', fake_compile)
    return calls


def test_runner_does_not_compile_by_default(monkeypatch, capsys):
    calls = _count_compiles(monkeypatch)
    runner = make_cartpole_runner()
    runner.params['config'].pop('torch_compile')
    runner.run_train({'checkpoint': None})
    assert not calls
    assert 'torch.compile: Disabled' in capsys.readouterr().out


def test_runner_compiles_on_request(monkeypatch):
    calls = _count_compiles(monkeypatch)
    runner = make_cartpole_runner(torch_compile=True)
    runner.run_train({'checkpoint': None})
    assert calls
