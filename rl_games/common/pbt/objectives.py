"""Objective functions for `pbt.objective_fn`, for backends whose infos carry no per-episode objective:

    pbt:
      objective_fn: rl_games.common.pbt.objectives:episode_return

Each takes the observer's (infos, done_indices, num_agents, algo) and returns per-env values that
the observer reads at the envs that finished (docs/PBT.md, Objective). A NaN value leaves its env out.
"""

import numpy as np


def episode_return(infos, done_indices, num_agents, algo):
    """The undiscounted return of each env's finished episode: the learner's running reward sum
    (`algo.current_rewards`, first reward head), read before the learner resets it; per env, the
    mean over its agents. Any backend (MuJoCo, envpool, mjlab, Isaac Lab, gym)."""
    rewards = algo.current_rewards[:, 0]
    if num_agents > 1:
        rewards = rewards.view(-1, num_agents).mean(dim=1)
    return rewards


def game_score(infos, done_indices, num_agents, algo):
    """envpool Atari: the raw (unclipped) game score `infos['scores']` of each env whose game ended.
    Under `episodic_life` an episode ends at every lost life while the score runs on to game over
    (`lives == 0`): those episodes are NaN, so each env keeps its last game's score."""
    scores = np.asarray(infos["scores"], dtype=np.float64)
    if "lives" in infos:
        scores = np.where(np.asarray(infos["lives"]) == 0, scores, np.nan)
    return scores
