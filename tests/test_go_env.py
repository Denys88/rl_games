"""Tests for the pgx Go 9x9 vecenv (Phase 0).

Covers: symmetry tables and obs/action frame consistency, color assignment,
pass masking, terminal reward/score/ownership consistency, and seeding.
Requires a GPU-visible JAX or falls back to CPU JAX transparently.
"""

import numpy as np
import pytest
import torch

from rl_games.envs.pgx_go import PgxGoVecEnv, _build_sym_tables, _inverse_sym_tables

N_BOARDS = 128
PASS = 81


@pytest.fixture(scope='module')
def env():
    return PgxGoVecEnv('pgx_go', N_BOARDS, komi=7.0, seed=3)


def _rand_legal(mask):
    return torch.multinomial(mask.float(), 1).squeeze(1)


def test_sym_tables_are_permutations():
    to_env = _build_sym_tables(9)
    to_sym = _inverse_sym_tables(to_env)
    assert to_env.shape == (8, 82)
    for s in range(8):
        assert sorted(to_env[s].tolist()) == list(range(82))
        assert to_env[s][PASS] == PASS  # pass invariant
        assert np.array_equal(to_sym[s][to_env[s]], np.arange(82))
    # identity transform is row 0
    assert np.array_equal(to_env[0], np.arange(82))
    # all 8 transforms distinct
    assert len({tuple(r) for r in to_env}) == 8


def test_reset_shapes_and_mask(env):
    obs = env.reset()
    mask = env.get_action_masks()
    assert obs.shape == (N_BOARDS, 9, 9, 17)
    assert obs.dtype == torch.float32
    assert mask.shape == (N_BOARDS, 82)
    assert mask.dtype == torch.bool
    # pass is masked for the first 20 plies
    assert not mask[:, PASS].any()
    # (near-)empty boards: at most one point occupied (opponent's opener)
    assert (mask[:, :PASS].sum(-1) >= 80).all()


def test_color_assignment_and_first_move(env):
    obs = env.reset().cpu().numpy()
    # plane 16 is the learner's colour (0 = black, 1 = white)
    color_plane = obs[:, 0, 0, 16]
    frac_white = color_plane.mean()
    assert 0.3 < frac_white < 0.7
    # white learner boards must already contain the opponent's opening stone
    opp_stones = obs[:, :, :, 1].reshape(N_BOARDS, -1).sum(-1)
    white = color_plane == 1
    assert (opp_stones[white] == 1).all()
    assert (opp_stones[~white] == 0).all()


def test_own_move_appears_in_sym_frame(env):
    env.reset()
    mask = env.get_action_masks()
    a = _rand_legal(mask)
    obs, _, done, _ = env.step(a)
    obs = obs.cpu().numpy()
    a = a.cpu().numpy()
    done = done.cpu().numpy()
    my_stones = obs[:, :, :, 0].reshape(N_BOARDS, -1)
    hits = 0
    for i in range(N_BOARDS):
        if done[i]:
            continue
        # the learner's stone must sit at the sym-frame index it played at
        # (unless captured, impossible on move 1)
        hits += my_stones[i, a[i]] == 1
    assert hits == (1 - done).sum()


def test_move_number_progression(env):
    env.reset()
    for _ in range(3):
        mask = env.get_action_masks()
        obs, _, done, info = env.step(_rand_legal(mask))
    mn = info['move_number'].cpu().numpy()
    done = done.cpu().numpy()
    color = info['learner_color'].cpu().numpy()
    # non-done boards: learner+opponent = 2 plies per step, +1 if learner white
    expect = 6 + color
    assert (mn[done == 0] == expect[done == 0]).all()


def test_pass_unmasked_after_opening(env):
    env.reset()
    for _ in range(12):  # 24+ plies > pass_mask_moves
        mask = env.get_action_masks()
        obs, _, done, info = env.step(_rand_legal(mask))
    mask = env.get_action_masks()
    mn = info['move_number'].cpu().numpy()
    allowed = mask[:, PASS].cpu().numpy()
    assert allowed[mn >= 20].all()
    assert not allowed[mn < 20].any()


def test_terminal_consistency(env):
    env.reset()
    wins, scores, own_sums = [], [], []
    for _ in range(300):
        mask = env.get_action_masks()
        obs, rew, done, info = env.step(_rand_legal(mask))
        d = done.bool()
        if d.any():
            wins += info['win'][d].tolist()
            scores += info['score_diff'][d].tolist()
            own_sums += info['ownership'][d].sum(-1).tolist()
            # non-terminal boards report zero ownership/score
            nd = ~d
            if nd.any():
                assert info['score_diff'][nd].abs().max().item() == 0
                assert info['ownership'][nd].abs().max().item() == 0
    assert len(wins) > 50
    wins, scores, own_sums = map(np.array, (wins, scores, own_sums))
    assert set(np.unique(wins)) <= {-1.0, 1.0}
    # ownership sums to the raw stone/territory diff: score_diff +/- komi
    komi_off = np.minimum(np.abs(own_sums - scores - 7.0),
                          np.abs(own_sums - scores + 7.0))
    assert komi_off.max() < 1e-4
    # win agrees with the sign of the komi-adjusted score except PSK endings
    nonzero = scores != 0
    agree = (np.sign(scores[nonzero]) == wins[nonzero]).mean()
    assert agree > 0.95
    # both colours win sometimes
    assert 0.2 < (wins > 0).mean() < 0.8


def test_seed_determinism():
    e1 = PgxGoVecEnv('pgx_go', 16, komi=7.0, seed=7)
    e2 = PgxGoVecEnv('pgx_go', 16, komi=7.0, seed=7)
    o1, o2 = e1.reset(), e2.reset()
    assert torch.equal(o1, o2)
    for _ in range(5):
        a = _rand_legal(e1.get_action_masks())
        r1 = e1.step(a.clone())
        r2 = e2.step(a.clone())
        assert torch.equal(r1[0], r2[0])
        assert torch.equal(r1[1], r2[1])


def test_no_symmetry_mode():
    e = PgxGoVecEnv('pgx_go', 16, komi=7.0, seed=5, symmetry=False)
    e.reset()
    for _ in range(3):
        mask = e.get_action_masks()
        obs, _, done, info = e.step(_rand_legal(mask))
    assert obs.shape == (16, 9, 9, 17)


def test_env_info(env):
    info = env.get_env_info()
    assert info['action_space'].n == 82
    assert info['observation_space'].shape == (9, 9, 17)
    assert env.has_action_masks()
