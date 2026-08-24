"""League unit tests: payoff EMA, PFSP weighting, snapshot/evict, stacking,
and the env's grouped pool mode end-to-end (small batch)."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from rl_games.common.league import League, MAIN_ID


def _params(v):
    return {'w': jnp.full((3,), float(v))}


def test_add_and_record():
    lg = League(max_pool=4, payoff_ema=0.1)
    a = lg.add(_params(1), epoch=1)
    b = lg.add(_params(2), epoch=2)
    assert set(lg.members) == {a, b}
    lg.record([a] * 50, [1.0] * 50)   # main crushes a
    lg.record([b] * 50, [0.0] * 50)   # b crushes main
    assert lg.payoff[a] > 0.9
    assert lg.payoff[b] < 0.1
    # records for main-id and unknown ids are ignored
    lg.record([MAIN_ID, 999], [1.0, 1.0])


def test_pfsp_prefers_hard_opponents():
    lg = League(max_pool=8, payoff_ema=0.1, variance_warmup_games=10)
    easy = lg.add(_params(1), epoch=1)
    hard = lg.add(_params(2), epoch=2)
    lg.record([easy] * 200, [1.0] * 200)
    lg.record([hard] * 200, [0.0] * 200)
    ids = sorted(lg.members)
    w = lg._pfsp_weights(ids)
    assert w[ids.index(hard)] > 0.9
    groups = lg.sample_groups(32, p_self=0.25)
    assert (groups == MAIN_ID).sum() >= 8          # self share respected
    picked = groups[groups != MAIN_ID]
    assert (picked == hard).mean() > 0.6           # PFSP focuses on the hard one


def test_eviction_drops_beaten_snapshot():
    lg = League(max_pool=2, payoff_ema=0.1)
    a = lg.add(_params(1), epoch=1)
    b = lg.add(_params(2), epoch=2)
    lg.record([a] * 100, [1.0] * 100)   # a is beaten
    lg.record([b] * 100, [0.5] * 100)   # b is competitive
    c = lg.add(_params(3), epoch=3)     # forces eviction
    assert a not in lg.members
    assert b in lg.members and c in lg.members


def test_build_stacked():
    lg = League(max_pool=4)
    a = lg.add(_params(5))
    ids = [MAIN_ID, a, a, MAIN_ID]
    stacked = lg.build_stacked(ids, _params(9))
    w = np.asarray(stacked['w'])
    assert w.shape == (4, 3)
    assert (w[0] == 9).all() and (w[1] == 5).all() and (w[3] == 9).all()


def test_env_pool_mode_end_to_end():
    from rl_games.envs.pgx_go import PgxGoVecEnv
    from rl_games.envs.go_flax import GoResNetFlax, init_flax_params

    arch = dict(blocks=2, channels=16, gpool_every=2, value_units=32)
    env = PgxGoVecEnv('pgx_go', 32, komi=7.0, seed=0, opponent='pool',
                      pool_groups=4, **arch)
    env.reset()
    # bind groups to two different member ids
    net = GoResNetFlax(**arch)
    p1 = init_flax_params(net, seed=1)
    p2 = init_flax_params(net, seed=2)
    stacked = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, a, b, b]), p1, p2)
    env.set_pool_assignment(stacked, [0, 3, 7, 3])
    for _ in range(3):
        a = torch.multinomial(env.get_action_masks().float(), 1).squeeze(1)
        obs, r, d, info = env.step(a)
    opp_id = info['opp_id'].cpu().numpy()
    assert (opp_id[:8] == 0).all()
    assert (opp_id[8:16] == 3).all()
    assert (opp_id[16:24] == 7).all()
    assert (opp_id[24:] == 3).all()
