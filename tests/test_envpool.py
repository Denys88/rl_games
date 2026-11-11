"""Envpool wrapper guard: MyoSuite tasks on envpool < 1.2.6 are refused
(deterministic resets, envpool#432) unless env_config sets
allow_deterministic_resets: true.

The stub-module tests run without envpool installed (CI has none); the
real-path test skips unless envpool is importable.
"""
import sys
import types
import warnings

import gymnasium
import numpy as np
import pytest

from rl_games.envs.envpool import Envpool, _version_tuple

MYO_ID = 'MyoSuite/myoElbowPose1D6MRandom-v0'


class _FakeEnv:
    observation_space = gymnasium.spaces.Box(-1, 1, (9,), np.float32)
    action_space = gymnasium.spaces.Box(-1, 1, (6,), np.float32)


def _stub_envpool(monkeypatch, version):
    """Install a stub envpool module: records make_gymnasium calls, returns a spaces-only env."""
    mod = types.ModuleType('envpool')
    mod.__version__ = version
    mod.calls = []

    def make_gymnasium(env_name, **kwargs):
        mod.calls.append((env_name, kwargs))
        return _FakeEnv()

    mod.make_gymnasium = make_gymnasium
    monkeypatch.setitem(sys.modules, 'envpool', mod)
    return mod


@pytest.mark.parametrize('env_name', [MYO_ID, 'myoElbowPose1D6MRandom-v0', 'MyoHandAirplaneFixed-v0'])
def test_myosuite_on_old_envpool_raises(monkeypatch, env_name):
    stub = _stub_envpool(monkeypatch, '1.2.5')
    with pytest.raises(RuntimeError, match=r'1\.2\.6'):
        Envpool('', 4, env_name=env_name)
    assert stub.calls == []  # refused before any env is built


def test_opt_out_warns_and_constructs(monkeypatch):
    stub = _stub_envpool(monkeypatch, '1.2.5')
    with pytest.warns(RuntimeWarning, match='deterministic'):
        env = Envpool('', 4, env_name=MYO_ID, allow_deterministic_resets=True)
    (name, kwargs), = stub.calls
    assert name == MYO_ID
    assert 'allow_deterministic_resets' not in kwargs  # consumed, not passed to envpool
    assert env.observation_space.shape == (9,)


@pytest.mark.parametrize('env_name', ['HalfCheetah-v4', 'Pong-v5', 'HumanoidWalk-v1'])
def test_non_myosuite_on_old_envpool_unaffected(monkeypatch, env_name):
    stub = _stub_envpool(monkeypatch, '1.2.5')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Envpool('', 4, env_name=env_name)
    assert len(stub.calls) == 1


@pytest.mark.parametrize('version', ['1.2.6', '1.10.0', '2.0.0'])
def test_myosuite_on_fixed_envpool_unaffected(monkeypatch, version):
    stub = _stub_envpool(monkeypatch, version)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Envpool('', 4, env_name=MYO_ID)
    assert len(stub.calls) == 1


def test_version_tuple_tolerates_suffixes():
    assert _version_tuple('1.2.5') == (1, 2, 5)
    assert _version_tuple('1.2.6rc1') == (1, 2, 6)
    assert _version_tuple('1.2.5.post1') == (1, 2, 5)
    assert _version_tuple('1.10.0') > (1, 2, 6)  # numeric, not string, comparison


def test_real_envpool_myosuite_guard():
    envpool = pytest.importorskip('envpool')
    if MYO_ID not in envpool.list_all_envs():
        pytest.skip(f'{MYO_ID} not registered in envpool {envpool.__version__}')
    if _version_tuple(envpool.__version__) < (1, 2, 6):
        with pytest.raises(RuntimeError, match=r'1\.2\.6'):
            Envpool('', 2, env_name=MYO_ID)
        with pytest.warns(RuntimeWarning, match='deterministic'):
            env = Envpool('', 2, env_name=MYO_ID, allow_deterministic_resets=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            env = Envpool('', 2, env_name=MYO_ID)
    assert env.reset().shape[0] == 2
    Envpool('', 2, env_name='HalfCheetah-v4')  # non-MyoSuite: no guard on any version
