"""Evaluation checkpoint probes must handle full states, partial writes and no monitored dir."""

import threading

import numpy as np
import pytest
import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.common.player import BasePlayer
from tests.test_ppo_masking import _ppo_params
from tests.test_sac_correctness import EP_LEN, FakeNextStepVecEnv


def _player(path):
    player = BasePlayer.__new__(BasePlayer)
    player.checkpoint_mutex = threading.Lock()
    player.checkpoint_to_load = str(path) if path is not None else None
    player.loaded = []
    player.restore = lambda filename: player.loaded.append(torch_ext.load_checkpoint(filename))
    return player


def _evaluation_player(env, **player_config):
    # built by the constructor, as a launch builds it: `_player` skips it and sets the lock by hand
    params = _ppo_params(
        num_envs=env.num_envs, env_info=env.get_env_info(), vec_env=env, device_name="cpu",
        player={"evaluation": True, "print_stats": False, **player_config})
    return PpoPlayerContinuous(params)


@pytest.mark.parametrize("with_environment_state", [False, True])
def test_evaluation_loads_serialized_training_checkpoint(tmp_path, with_environment_state):
    path = tmp_path / "training.pth"
    payload = {"model": {"weight": torch.arange(4, dtype=torch.float32)}, "epoch": 7}
    if with_environment_state:
        payload["env_state"] = {"position": np.array([1.5, -2.0], dtype=np.float32)}
    torch.save(payload, path)
    player = _player(path)
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None
    assert len(player.loaded) == 1
    torch.testing.assert_close(player.loaded[0]["model"]["weight"], payload["model"]["weight"])
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


def test_evaluation_without_monitored_dir_has_nothing_to_load():
    player = _evaluation_player(FakeNextStepVecEnv(2))
    assert player.dir_to_monitor is None
    player.wait_for_checkpoint()
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None


def test_evaluation_without_monitored_dir_plays_its_games():
    env = FakeNextStepVecEnv(2)
    player = _evaluation_player(env, games_num=env.num_envs, max_steps=EP_LEN)
    player.run()
    # one full episode per env: the loop got past the checkpoint probe of its first step
    assert env.t.tolist() == [EP_LEN] * env.num_envs
    assert player.checkpoint_to_load is None
