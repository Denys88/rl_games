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
from tests.test_pbt import BandWriter, FakeAlgo, make_observer
from tests.test_pbt_e2e import REPO, SCRIPT, SMALL, member_env

DONE = torch.tensor([[1], [3]])
DONE3 = torch.tensor([[0], [1], [2]])


def test_episode_return_reads_the_running_sum_of_the_first_reward_head():
    algo = types.SimpleNamespace(current_rewards=torch.tensor([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]]))
    assert objectives.episode_return({}, DONE, 1, algo).tolist() == [1.0, 2.0, 3.0, 4.0]
    # two agents per env: the mean over each env's agents
    assert objectives.episode_return({}, DONE, 2, algo).tolist() == [1.5, 3.5]


def test_game_score_counts_only_finished_games():
    """Every env ended an episode; env 1 only lost a life: the function returns the two games over
    as its done indices."""
    infos = {"scores": np.array([10.0, 20.0, 30.0]), "lives": np.array([0, 2, 0])}
    scores, done = objectives.game_score(infos, DONE3, 1, None)
    assert scores.tolist() == [10.0, 20.0, 30.0] and done.tolist() == [0, 2]
    assert objectives.game_score(infos, None, 1, None)[1].tolist() == [0, 2]   # no indices: every env's
    assert objectives.game_score({"scores": np.array([5.0])}, DONE, 1, None).tolist() == [5.0]   # no lives


def test_game_score_counts_games_cut_off_by_the_time_limit():
    """A game truncated at max_episode_steps ends with lives left (envpool restarts it with full
    lives): its score counts, or an env that keeps reaching the limit never gets a value."""
    infos = {"scores": np.array([10.0, 20.0, 30.0]), "lives": np.array([0, 2, 3]),
             "time_outs": np.array([False, True, False])}
    assert objectives.game_score(infos, DONE3, 1, None)[1].tolist() == [0, 1]
    infos["time_outs"] = torch.tensor([False, True, False])                    # a tensor works too
    assert objectives.game_score(infos, torch.tensor([[1], [2]]), 1, None)[1].tolist() == [1]


def test_lost_lives_never_fail_a_member_whose_games_outlast_its_intervals(tmp_path):
    """A lost life is no episode for PBT. A healthy member whose games run for more than
    max_consecutive_errors intervals (the better the policy, the longer they run) waits for its
    first games to end, then decides at every boundary: it never raises as diverged, also after
    the synchronized reset of a (re)start."""
    envs, life, interval = 4, 40, 30                     # a game: 5 lives x 40 steps = 200 steps; 5 intervals: 150
    obs = make_observer(tmp_path, extra_pbt={"interval_steps": interval * envs, "objective": "",
                                             "objective_fn": "rl_games.common.pbt.objectives:game_score"})
    algo = FakeAlgo(frame=0)
    algo.train_dir, algo.num_actors, algo.writer = str(tmp_path), envs, BandWriter()
    obs.after_init(algo)
    calls = []
    obs._pbt_iteration = lambda score, best, frame: calls.append((score, frame))
    obs.after_steps()
    lives, score = np.full(envs, 5), np.zeros(envs)
    for step in range(1, 601):                           # every env's games start together
        score += 1.0
        lost = np.full(envs, step % life == 0)
        lives = lives - lost
        infos = {"scores": score.copy(), "lives": lives.copy(), "time_outs": np.zeros(envs, dtype=bool)}
        obs.process_infos(infos, torch.tensor(np.nonzero(lost)[0]).reshape(-1, 1))
        over = lost & (lives == 0)
        score[over], lives[over] = 0.0, 5
        algo.frame = step * envs
        obs.after_steps()                                # raised here: 'a diverged policy'
    assert obs._interval_nonfinite == 0 and not obs._failed_envs.any()
    # the first games' score when they ended, then one iteration per boundary
    assert [frame for _, frame in calls] == [200 * envs] + [step * envs for step in range(210, 601, interval)]
    assert {score for score, _ in calls} == {200.0}


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
