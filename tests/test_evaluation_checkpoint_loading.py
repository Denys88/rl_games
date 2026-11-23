"""Evaluation checkpoint probes must handle full states and partial writes."""

import pickle
import threading
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer


@dataclass
class _SimulatorState:
    step: int


def _player(path, *, trusted=False):
    player = BasePlayer.__new__(BasePlayer)
    player.checkpoint_mutex = threading.Lock()
    player.checkpoint_to_load = str(path) if path is not None else None
    player.loaded = []
    player.restore = lambda filename: player.loaded.append(torch_ext.load_checkpoint(filename, trusted=trusted))
    return player


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_preexisting_numpy_reward_checkpoint_loads_by_default(tmp_path, dtype):
    path = tmp_path / 'best.pth'
    torch.save({'model': {'weight': torch.ones(1)}, 'last_mean_rewards': dtype(3.5)}, path)

    checkpoint = torch_ext.load_checkpoint(path, map_location='cpu')
    assert checkpoint['last_mean_rewards'] == 3.5
    assert type(checkpoint['last_mean_rewards']) is dtype
    torch.testing.assert_close(checkpoint['model']['weight'], torch.ones(1))


def test_restricted_load_preserves_caller_registered_globals(tmp_path):
    path = tmp_path / 'best.pth'
    torch.save({'last_mean_rewards': np.float32(3.5)}, path)
    with torch.serialization.safe_globals([np.dtype]):
        previous = set(torch.serialization.get_safe_globals())
        torch_ext.load_checkpoint(path)
        assert set(torch.serialization.get_safe_globals()) == previous


@pytest.mark.parametrize("with_environment_state", [False, True])
def test_evaluation_loads_serialized_training_checkpoint(tmp_path, with_environment_state):
    path = tmp_path / "training.pth"
    payload = {"model": {"weight": torch.arange(4, dtype=torch.float32)}, "epoch": 7,
               "last_mean_rewards": np.float32(3.5)}
    if with_environment_state:
        payload["env_state"] = {"position": np.array([1.5, -2.0], dtype=np.float32)}
    torch.save(payload, path)
    player = _player(path, trusted=with_environment_state)
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None
    assert len(player.loaded) == 1
    torch.testing.assert_close(player.loaded[0]["model"]["weight"], payload["model"]["weight"])
    assert player.loaded[0]['last_mean_rewards'] == 3.5
    if with_environment_state:
        np.testing.assert_array_equal(player.loaded[0]["env_state"]["position"],
                                      payload["env_state"]["position"])


@pytest.mark.parametrize("contents", [b"", b"PK\x03\x04", b"unfinished checkpoint"])
def test_partial_checkpoint_does_not_crash_or_prevent_later_load(tmp_path, contents):
    path = tmp_path / "in_progress.pth"
    path.write_bytes(contents)
    player = _player(path)
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None
    assert not player.loaded

    torch.save({"model": {"weight": torch.tensor([3.0])}}, path)
    player.checkpoint_to_load = str(path)
    player.maybe_load_new_checkpoint()
    assert len(player.loaded) == 1


def test_missing_checkpoint_is_discarded(tmp_path):
    player = _player(tmp_path / "missing.pth")
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None
    assert not player.loaded


def test_restore_failure_remains_nonfatal(tmp_path):
    path = tmp_path / "valid.pth"
    torch.save({"model": {}}, path)
    player = _player(path)

    def fail_restore(filename):
        raise ValueError("incompatible model shape")

    player.restore = fail_restore
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None


def test_no_pending_checkpoint_is_a_noop():
    player = _player(None)
    player.maybe_load_new_checkpoint()
    assert not player.loaded


@pytest.mark.parametrize('env_state', [np.arange(3), _SimulatorState(step=7)])
def test_checkpoint_with_legacy_objects_requires_explicit_trust(tmp_path, env_state):
    path = tmp_path / "legacy.pth"
    torch.save({"last_mean_rewards": np.float32(3.5), "env_state": env_state}, path)
    with pytest.raises(pickle.UnpicklingError, match="trust-checkpoint"):
        torch_ext.load_checkpoint(path)
    with pytest.raises(TypeError, match="boolean"):
        torch_ext.load_checkpoint(path, trusted="false")
    player = _player(path)
    player.maybe_load_new_checkpoint()
    assert not player.loaded
    assert player.checkpoint_to_load is None
    restored = torch_ext.load_checkpoint(path, trusted=True)
    np.testing.assert_equal(restored['env_state'], env_state)
