"""Evaluation checkpoint probes must handle full states and partial writes."""

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


def _player(path):
    player = BasePlayer.__new__(BasePlayer)
    player.checkpoint_mutex = threading.Lock()
    player.checkpoint_to_load = str(path) if path is not None else None
    player.loaded = []
    player.restore = lambda filename: player.loaded.append(torch_ext.load_checkpoint(filename))
    return player


@pytest.mark.parametrize('best_reward', [
    np.float32(3.5), np.float64(3.5),       # PPO saved a NumPy scalar
    np.array(3.5, dtype=np.float32),        # SAC saved a 0-d NumPy array
])
def test_preexisting_numpy_reward_checkpoint_loads_by_default(tmp_path, best_reward):
    path = tmp_path / 'best.pth'
    torch.save({'model': {'weight': torch.ones(1)}, 'last_mean_rewards': best_reward}, path)

    checkpoint = torch_ext.load_checkpoint(path, map_location='cpu')
    assert checkpoint['last_mean_rewards'] == 3.5
    assert type(checkpoint['last_mean_rewards']) is type(best_reward)
    torch.testing.assert_close(checkpoint['model']['weight'], torch.ones(1))


@pytest.mark.parametrize("with_environment_state", [False, True])
def test_evaluation_loads_serialized_training_checkpoint(tmp_path, with_environment_state):
    path = tmp_path / "training.pth"
    payload = {"model": {"weight": torch.arange(4, dtype=torch.float32)}, "epoch": 7,
               "last_mean_rewards": np.float32(3.5)}
    if with_environment_state:
        payload["env_state"] = {"position": np.array([1.5, -2.0], dtype=np.float32)}
    torch.save(payload, path)
    player = _player(path)
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


def test_evaluation_without_a_monitored_directory_does_not_crash():
    # run() polls maybe_load_new_checkpoint() whenever player.evaluation is on; the mutex it
    # takes used to exist only when player.dir_to_monitor was set as well
    from rl_games.algos_torch.players import PpoPlayerContinuous
    from tests.test_ppo_masking import _ppo_params
    from tests.test_sac_correctness import FakeNextStepVecEnv

    params = _ppo_params(env_info=FakeNextStepVecEnv(2).get_env_info())
    params['config'].update(device_name='cpu', player={'evaluation': True})
    player = PpoPlayerContinuous(params)
    assert player.evaluation and player.dir_to_monitor is None

    player.maybe_load_new_checkpoint()
    player.wait_for_checkpoint()
    assert player.checkpoint_to_load is None


@pytest.mark.parametrize('env_state', [np.arange(3), _SimulatorState(step=7)])
def test_checkpoint_with_custom_environment_state_loads_by_default(tmp_path, env_state):
    # env_state is whatever the environment returns from get_env_state()
    path = tmp_path / "legacy.pth"
    torch.save({"last_mean_rewards": np.float32(3.5), "env_state": env_state}, path)
    np.testing.assert_equal(torch_ext.load_checkpoint(path)['env_state'], env_state)
    player = _player(path)
    player.maybe_load_new_checkpoint()
    assert player.checkpoint_to_load is None
    np.testing.assert_equal(player.loaded[0]['env_state'], env_state)
