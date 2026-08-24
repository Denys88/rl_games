"""Unit tests for PgxGoVecEnv.process_rollout_targets (back-fill logic).

Feeds a synthetic rollout-extras stack and checks that terminal quantities
propagate backwards exactly over episode boundaries and that validity masks
are right. Pure torch; no JAX needed beyond env construction."""

import pytest
import torch

from rl_games.envs.pgx_go import PgxGoVecEnv


@pytest.fixture(scope='module')
def env():
    return PgxGoVecEnv('pgx_go', 2, komi=7.0, seed=0)


def _extras(T, N, n=81):
    return {
        '_dones': torch.zeros(T, N),
        'ownership': torch.zeros(T, N, n),
        'score_diff': torch.zeros(T, N),
        'plies': torch.zeros(T, N),
        'opp_dist': torch.zeros(T, N, n + 1),
        'move_number_pre': torch.zeros(T, N),
    }


def test_backfill_single_episode(env):
    T, N = 8, 2
    ex = _extras(T, N)
    # board 0: episode ends at t=5 with score +12, 80 plies
    ex['_dones'][5, 0] = 1.0
    ex['score_diff'][5, 0] = 12.0
    ex['plies'][5, 0] = 80.0
    ex['ownership'][5, 0, :] = 1.0
    ex['move_number_pre'][:, 0] = torch.arange(T).float() * 2

    t = env.process_rollout_targets(ex, None)

    v = t['go_terminal_valid']
    # steps 0..5 covered, 6..7 belong to the unfinished next episode
    assert v[:6, 0].eq(1).all() and v[6:, 0].eq(0).all()
    # board 1 never finished
    assert v[:, 1].eq(0).all()
    assert t['go_score'][:6, 0].eq(12.0).all()
    assert t['go_ownership'][3, 0].eq(1.0).all()
    # plies_left = terminal plies - plies at obs time
    assert t['go_plies_left'][0, 0] == 80.0
    assert t['go_plies_left'][5, 0] == 80.0 - 10.0


def test_backfill_two_episodes(env):
    T, N = 8, 1
    ex = _extras(T, N)
    # episode A ends t=2 (score +5), episode B ends t=6 (score -3)
    ex['_dones'][2, 0] = 1.0
    ex['score_diff'][2, 0] = 5.0
    ex['plies'][2, 0] = 70.0
    ex['_dones'][6, 0] = 1.0
    ex['score_diff'][6, 0] = -3.0
    ex['plies'][6, 0] = 90.0

    t = env.process_rollout_targets(ex, None)
    s = t['go_score'][:, 0]
    assert s[:3].eq(5.0).all()      # episode A steps get A's terminal
    assert s[3:7].eq(-3.0).all()    # episode B steps get B's terminal
    assert t['go_terminal_valid'][7, 0] == 0  # tail of unfinished episode C


def test_opp_dist_validity(env):
    T, N = 4, 2
    ex = _extras(T, N)
    ex['opp_dist'][1, 0, :] = 1.0 / 82
    t = env.process_rollout_targets(ex, None)
    assert t['go_opp_valid'][1, 0] == 1.0
    assert t['go_opp_valid'][0, 0] == 0.0
    assert t['go_opp_valid'][1, 1] == 0.0
