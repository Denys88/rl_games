"""PBT objective functions (rl_games.common.pbt.objectives) and train_member.py --file. CPU only."""

import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import yaml

from rl_games.common.pbt import objectives
from tests.test_pbt_e2e import REPO, SCRIPT, SMALL, member_env

DONE = torch.tensor([[1], [3]])


def test_episode_return_reads_the_running_sum_of_the_first_reward_head():
    algo = types.SimpleNamespace(current_rewards=torch.tensor([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]]))
    assert objectives.episode_return({}, DONE, 1, algo).tolist() == [1.0, 2.0, 3.0, 4.0]
    # two agents per env: the mean over each env's agents
    assert objectives.episode_return({}, DONE, 2, algo).tolist() == [1.5, 3.5]


def test_game_score_counts_only_finished_games():
    infos = {"scores": np.array([10.0, 20.0, 30.0]), "lives": np.array([0, 2, 0])}
    values = objectives.game_score(infos, DONE, 1, None)
    assert values[0] == 10.0 and np.isnan(values[1]) and values[2] == 30.0
    assert objectives.game_score({"scores": np.array([5.0])}, DONE, 1, None).tolist() == [5.0]   # no lives


def test_game_score_counts_games_cut_off_by_the_time_limit():
    """A game truncated at max_episode_steps ends with lives left (envpool restarts it with full
    lives): its score counts, or an env that keeps reaching the limit never gets a value."""
    infos = {"scores": np.array([10.0, 20.0, 30.0]), "lives": np.array([0, 2, 3]),
             "time_outs": np.array([False, True, False])}
    values = objectives.game_score(infos, DONE, 1, None)
    assert values[0] == 10.0 and values[1] == 20.0 and np.isnan(values[2])
    infos["time_outs"] = torch.tensor([False, True, False])                    # a tensor works too
    assert objectives.game_score(infos, DONE, 1, None)[1] == 20.0


@pytest.mark.skipif(os.name != "posix", reason="the member runs as a subprocess")
def test_train_member_trains_a_config_file_ranked_by_its_episode_return(tmp_path):
    """--file: any rl_games config, here CartPole's, ranked by objectives.episode_return. The member
    waits until every env has finished an episode (objective_window 0), then saves its iterations
    with the episode return as the objective."""
    cmd = [sys.executable, SCRIPT, "--file", os.path.join("rl_games", "configs", "ppo_cartpole.yaml"),
           "--policy-idx", "0", "--num-policies", "1", "--max-epochs", "12", "--interval-steps", "256",
           "--seed", "1", "--train-dir", str(tmp_path), *SMALL]
    run = subprocess.run(cmd, cwd=REPO, env=member_env(), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    member = tmp_path / "pbt_workspace" / "000"
    saved = sorted(member.glob("0*.yaml"))
    assert saved, run.stdout[-3000:]
    ckpt = yaml.safe_load(saved[-1].read_text())
    assert 8.0 < ckpt["true_objective"] < 500.0         # CartPole episode returns: steps survived
