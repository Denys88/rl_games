"""Tests for the envpool soccer vec env, the soccer observers and the
framework-agnostic League changes. Env tests need envpool with
DmcSoccerBoxhead-v1 (envpool main after 2026-08-30, see docs/ENVPOOL_SOCCER.md).
"""

import numpy as np
import pytest
import torch

from rl_games.common.league import League


def _has_soccer():
    try:
        import envpool
        return 'DmcSoccerBoxhead-v1' in envpool.list_all_envs()
    except Exception:
        return False


needs_soccer = pytest.mark.skipif(not _has_soccer(), reason='envpool soccer env unavailable')

ENV_KW = dict(walker_type='boxhead', team_size=2, time_limit=1.0, enable_field_box=True,
              terminate_on_goal=False, pitch_size_min=[15.0, 10.0], goal_size=[1.0, 6.0, 0.6],
              num_threads=4, seed=3, device='cpu')


class _ConstPolicy(torch.nn.Module):
    """Stand-in for an rl_games model: forward(input_dict) -> {'mus', 'actions'}."""

    def __init__(self, act_dim=3):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(act_dim))

    def forward(self, input_dict):
        obs = input_dict['obs']
        mus = self.bias.expand(obs.shape[0], -1)
        return {'mus': mus, 'actions': mus}


def _make(num_envs=4, **kw):
    from rl_games.envs.envpool_soccer import EnvpoolSoccerVecEnv
    cfg = dict(ENV_KW)
    cfg.update(kw)
    return EnvpoolSoccerVecEnv('envpool_soccer', num_envs, **cfg)


@needs_soccer
def test_reset_step_shapes():
    env = _make(4)
    try:
        obs = env.reset()
        assert torch.is_tensor(obs) and obs.shape == (8, env.obs_dim) and obs.dtype == torch.float32
        assert env.get_env_info()['agents'] == 2
        assert env.action_space.shape == (3,)
        act = torch.zeros(8, 3)
        obs, rew, done, info = env.step(act)
        assert obs.shape == (8, env.obs_dim)
        assert rew.shape == (8,) and rew.dtype == torch.float32
        assert done.shape == (8,) and done.dtype == torch.bool
        assert info['time_outs'].shape == (8,)
    finally:
        env.close()


@needs_soccer
def test_rows_are_env_major_home_first():
    """The raw envpool batch arrives in completion order; after reordering the
    home players of match i must sit at rows 2i, 2i+1 and the away obs at
    env._away_obs[i]. Check via the arena keys: teammates see the same
    team_goal_mid distance-to-ball pattern? Simpler: raw stats_home_score is
    a per-team quantity — after a goal the two rows of a match agree."""
    env = _make(8)
    try:
        env.reset()
        for _ in range(3):
            env.step(torch.zeros(16, 3))
        # permutation must sort info players env_id
        raw_obs, raw_info = env.env.reset()
        env._update_perm(raw_info)
        pid = np.asarray(raw_info['players']['env_id']).reshape(-1)
        assert np.array_equal(env._ordered(pid), np.repeat(np.arange(8), 4))
    finally:
        env.close()


@needs_soccer
def test_match_end_infos_and_done_broadcast():
    env = _make(4, shaping_weights={'vel_to_ball': 0, 'vel_ball_to_goal': 0, 'veloc_forward': 0, 'goal': 100})
    try:
        env.reset()
        seen_done = False
        for t in range(60):
            obs, rew, done, info = env.step(torch.from_numpy(np.random.uniform(-1, 1, (8, 3)).astype(np.float32)))
            d = done.view(4, 2)
            assert torch.equal(d[:, 0], d[:, 1]), 'done must be per match'
            if done.any():
                seen_done = True
                for k in ('goal_diff', 'home_goals', 'away_goals', 'win', 'opp_id', 'match_len', 'scores'):
                    assert k in info and info[k].shape == (4,)
                assert info['time_outs'].view(4, 2)[:, 0].any()
                # 1 s at 40 Hz = 40 control steps
                assert info['match_len'][done.view(4, 2)[:, 0]].min().item() >= 39
                # no goals expected with random play in 1 s; pure goal reward => 0
                assert torch.all(rew == 0)
        assert seen_done
    finally:
        env.close()


@needs_soccer
def test_shaping_reward_matches_stats():
    env = _make(2, shaping_weights={'vel_to_ball': 0.5, 'vel_ball_to_goal': 2.0, 'veloc_forward': 0.25, 'goal': 100})
    try:
        env.reset()
        act = torch.from_numpy(np.random.uniform(-1, 1, (4, 3)).astype(np.float32))
        # replicate a step manually through the raw env to compare
        for _ in range(5):
            obs, rew, done, info = env.step(act)
        # recompute from the last raw obs via the wrapper's own accessors
        raw_obs, raw_info = env.env.reset()  # fresh raw state to get consistent obs
        env._update_perm(raw_info)
        closest = env._stat(raw_obs, raw_info, 'stats_closest_vel_to_ball')[:, :2]
        vbg = env._stat(raw_obs, raw_info, 'stats_vel_ball_to_goal')[:, :2]
        fwd = env._stat(raw_obs, raw_info, 'stats_veloc_forward')[:, :2]
        expected = 0.5 * np.repeat(closest.sum(1, keepdims=True), 2, 1) + 2.0 * vbg + 0.25 * fwd
        got, _ = env._home_rewards(raw_obs, raw_info, np.zeros(env.total_players, dtype=np.float32))
        assert np.allclose(got, expected.astype(np.float32), atol=1e-6)
        env.clip_vel_to_ball = True
        got, _ = env._home_rewards(raw_obs, raw_info, np.zeros(env.total_players, dtype=np.float32))
        expected = 0.5 * np.maximum(np.repeat(closest.sum(1, keepdims=True), 2, 1), 0) + 2.0 * vbg + 0.25 * fwd
        assert np.allclose(got, expected.astype(np.float32), atol=1e-6)
    finally:
        env.close()


@needs_soccer
def test_pool_assignment_applied_at_reset():
    env = _make(4, opponent='pool')
    try:
        env.set_opponent_template(_ConstPolicy())
        sd = {'bias': torch.tensor([0.7, -0.7, 0.2])}
        env.set_pool_assignment(np.array([1, -1, 1, 0]), {1: sd, 0: {'bias': torch.zeros(3)}})
        env.reset()
        assert np.array_equal(env.current_assignment(), [1, -1, 1, 0])
        away = env._opponent_actions()
        assert np.allclose(away[0], [0.7, -0.7, 0.2]) and np.allclose(away[2], [0.7, -0.7, 0.2])
        assert np.allclose(away[3], 0.0)
        assert not np.allclose(away[1], 0.0)     # random opponent
        # new assignment is deferred until the match ends
        env.set_pool_assignment(np.array([0, 0, 0, 0]))
        env.step(torch.zeros(8, 3))
        assert np.array_equal(env.current_assignment(), [1, -1, 1, 0])
        for _ in range(45):
            _, _, done, info = env.step(torch.zeros(8, 3))
            if done.any():
                assert torch.equal(info['opp_id'].cpu(), torch.tensor([1, -1, 1, 0]))
                break
        env.step(torch.zeros(8, 3))   # auto-reset step applies the pending ids
        assert np.array_equal(env.current_assignment(), [0, 0, 0, 0])
    finally:
        env.close()


@needs_soccer
def test_stuck_penalty_and_spread_bonus():
    env = _make(2, shaping_weights={'vel_to_ball': 0, 'vel_ball_to_goal': 0, 'veloc_forward': 0,
                                    'goal': 0, 'stuck_penalty': 0.5, 'spread_out': 0.0},
                stuck_steps=10, time_limit=2.0)
    try:
        env.reset()
        rews = []
        for _ in range(30):                       # zero actions: nobody moves, ball rests
            _, rew, _, info = env.step(torch.zeros(4, 3))
            rews.append(rew.view(2, 2)[:, 0].numpy().copy())
        rews = np.stack(rews)
        assert np.all(rews[:8] == 0)              # before stuck_steps: no penalty
        assert np.all(rews[12:] == -0.5)          # stuck: team penalty every step
        assert 'stuck_frac' not in info
        for _ in range(60):
            _, _, done, info = env.step(torch.zeros(4, 3))
            if done.any():
                assert info['stuck_frac'].min() > 0.5
                break
    finally:
        env.close()
    env = _make(2, shaping_weights={'vel_to_ball': 0, 'vel_ball_to_goal': 0, 'veloc_forward': 0,
                                    'goal': 0, 'spread_out': 0.1})
    try:
        env.reset()
        _, rew, _, _ = env.step(torch.zeros(4, 3))
        spread = env._ordered(np.asarray(env.last_raw_obs['stats_teammate_spread_out'])).reshape(2, 4)[:, 0]
        assert np.allclose(rew.view(2, 2)[:, 0].numpy(), 0.1 * spread.astype(np.float32))
    finally:
        env.close()


# ----------------------------------------------------------------- observers

class _FakeAlgo:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.games_to_track = 100
        self.ppo_device = 'cpu'
        self.writer = None
        self.model = _ConstPolicy()
        self.vec_env = None


def test_soccer_observer_kind_split():
    from rl_games.common.soccer_observer import SoccerObserver
    obs = SoccerObserver()
    obs.after_init(_FakeAlgo())
    infos = {
        'goal_diff': torch.tensor([2., -1., 0., 1.]),
        'home_goals': torch.tensor([2., 0., 1., 1.]),
        'away_goals': torch.tensor([0., 1., 1., 0.]),
        'opp_id': torch.tensor([-1, 0, 3, 3]),
        'match_len': torch.full((4,), 1200.),
    }
    # done indices are per-agent rows [::num_agents]: matches 0,1,3 finished
    obs.process_infos(infos, torch.tensor([[0], [2], [6]]))
    assert obs.meters['win'].current_size == 3
    assert abs(obs.meters['goal_diff'].get_mean().item() - (2 - 1 + 1) / 3) < 1e-6
    assert obs.kind_meters['random']['win'].get_mean().item() == 1.0
    assert obs.kind_meters['self']['win'].get_mean().item() == 0.0
    assert obs.kind_meters['pool']['goal_diff'].get_mean().item() == 1.0
    obs.after_clear_stats()
    assert obs.meters['win'].current_size == 0


def test_league_torch_host_fn_and_assignment():
    from rl_games.envs.envpool_soccer import state_dict_to_cpu, RANDOM_ID, MAIN_ID
    from rl_games.common.soccer_observer import SoccerLeagueObserver

    class _Env:
        num_envs = 20

        def __init__(self):
            self.assignments = []
            self.template = None
            self.main = None

        def set_opponent_template(self, model):
            self.template = model

        def set_pool_assignment(self, ids, params=None):
            self.assignments.append((np.array(ids), params))

        def set_main_params(self, sd):
            self.main = sd

    algo = _FakeAlgo()
    algo.vec_env = _Env()
    obs = SoccerLeagueObserver(league_config={
        'max_pool': 4, 'snapshot_every': 2, 'snapshot_min_gap': 1, 'remap_every': 1,
        'matchmaking': {'p_self': 0.3, 'p_pfsp': 0.4, 'p_uniform': 0.1, 'p_random': 0.2}})
    obs.after_init(algo)
    ids0, params0 = algo.vec_env.assignments[-1]
    assert (ids0 == RANDOM_ID).sum() == 4                      # 20% random anchors
    assert set(np.unique(ids0)) <= {RANDOM_ID, MAIN_ID}         # empty pool -> self-play
    assert set(params0) == {MAIN_ID} and params0[MAIN_ID]['bias'].device.type == 'cpu'

    obs.after_print_stats(frame=1, epoch_num=2, total_time=0.0)  # snapshot + remap
    assert len(obs.league.members) == 1
    ids, params = algo.vec_env.assignments[-1]
    assert (ids == RANDOM_ID).sum() == 4
    assert (ids > 0).sum() == 10                                # 50% pool slots
    assert 1 in params and params[1]['bias'].device.type == 'cpu'

    # payoff bookkeeping from finished matches: draws count 0.5
    infos = {'goal_diff': torch.tensor([1., 0., -1.]), 'home_goals': torch.zeros(3),
             'away_goals': torch.zeros(3), 'opp_id': torch.tensor([1, 1, 1])}
    obs.process_infos(infos, torch.tensor([[0], [2], [4]]))
    assert obs.league.counts[1] == 3
    assert abs(obs.league.payoff[1] - 0.5) < 0.05

    lg = League(host_fn=state_dict_to_cpu)
    mid = lg.add({'_orig_mod.w': torch.ones(2)}, epoch=1)
    assert list(lg.members[mid].params) == ['w']


# --------------------------------------------- player id / style / stochastic

class _GaussPolicy(torch.nn.Module):
    """Stub returning a fixed mean and a fixed sigma (rl_games contract)."""

    def __init__(self, act_dim=3, sigma=0.1):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(act_dim))
        self.register_buffer('sigma', torch.full((act_dim,), float(sigma)))

    def forward(self, input_dict):
        obs = input_dict['obs']
        mus = self.bias.expand(obs.shape[0], -1)
        sigmas = self.sigma.expand(obs.shape[0], -1)
        return {'mus': mus, 'sigmas': sigmas, 'actions': mus + sigmas * torch.randn_like(mus)}


def _run_to_first_done(env, n_home, max_steps=80):
    for _ in range(max_steps):
        _, _, done, info = env.step(torch.zeros(n_home, 3))
        if done.any():
            return done
    raise AssertionError('no match finished')


@needs_soccer
def test_player_id_one_hot_in_obs():
    base = _make(2)
    try:
        base_dim = base.obs_dim
    finally:
        base.close()
    env = _make(2, player_id_obs=True)
    try:
        assert env.obs_dim == base_dim + 2
        obs = env.reset()
        ids = obs[:, base_dim:]
        assert torch.equal(ids, torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
        away = env._away_obs.reshape(4, env.obs_dim)[:, base_dim:]
        assert torch.equal(away, ids)
        obs, _, _, _ = env.step(torch.zeros(4, 3))
        assert torch.equal(obs[:, base_dim:], ids)
    finally:
        env.close()










@needs_soccer
def test_stochastic_opponent_sigma_scale():
    for scale, want_std in ((0.0, 0.0), (2.0, 0.2)):
        env = _make(64, opponent='pool', opponent_deterministic=False,
                    opponent_sigma_scale=[scale, scale])
        try:
            env.set_opponent_template(_GaussPolicy(sigma=0.1))
            env.set_pool_assignment(np.zeros(64, dtype=np.int64), {0: _GaussPolicy(sigma=0.1).state_dict()})
            env.reset()
            acts = np.concatenate([env._opponent_actions().reshape(-1) for _ in range(4)])
            assert abs(acts.std() - want_std) < 0.03, acts.std()
        finally:
            env.close()
    env = _make(64, opponent='pool', opponent_deterministic=False, opponent_sigma_scale=[0.0, 3.0])
    try:
        env.set_opponent_template(_GaussPolicy(sigma=0.1))
        env.set_pool_assignment(np.zeros(64, dtype=np.int64), {0: _GaussPolicy(sigma=0.1).state_dict()})
        env.reset()
        s = env.current_opponent_sigma_scale()
        assert s.shape == (64,) and s.min() < 0.5 and s.max() > 2.5
        acts = env._opponent_actions()                           # (64, 2, 3)
        per_match = np.abs(acts).reshape(64, -1).mean(1)
        assert np.corrcoef(per_match, s)[0, 1] > 0.5
    finally:
        env.close()


@needs_soccer
def test_batched_pool_forward_matches_per_member():
    """The pool forward stacks every member and runs one vmapped call; it must
    equal running the real (jit-normalised) rl_games model member by member."""
    import yaml
    from rl_games.algos_torch.model_builder import ModelBuilder
    cfg = yaml.safe_load(open('rl_games/configs/envpool/ppo_soccer_boxhead_league_v3.yaml'))['params']
    env = _make(6, opponent='pool', player_id_obs=True)
    try:
        net = ModelBuilder().load(cfg)
        def build(seed):
            torch.manual_seed(seed)
            m = net.build({'actions_num': 3, 'input_shape': (env.obs_dim,), 'num_seqs': 1, 'value_size': 1,
                           'normalize_value': True, 'normalize_input': True}).eval()
            with torch.no_grad():
                m.running_mean_std.running_mean.add_(torch.randn(env.obs_dim, dtype=torch.float64) * seed)
            return m
        members = {0: build(1), 1: build(2), 2: build(3)}
        env.set_opponent_template(members[0])
        env.set_pool_assignment(np.array([1, 2, 0, 2, -1, 1]), {k: v.state_dict() for k, v in members.items()})
        env.reset()
        got = env._opponent_actions()
        for i, mid in enumerate([1, 2, 0, 2, -1, 1]):
            if mid < 0:
                continue
            with torch.no_grad():
                ref = members[mid]({'obs': env._away_obs[i], 'is_train': False})['mus']
            assert np.allclose(got[i], ref.clamp(-1, 1).numpy(), atol=1e-5), (i, mid)
        # main refresh lands in the stacked slot
        new = build(7)
        env.set_main_params(new.state_dict())
        got = env._opponent_actions()
        with torch.no_grad():
            ref = new({'obs': env._away_obs[2], 'is_train': False})['mus']
        assert np.allclose(got[2], ref.clamp(-1, 1).numpy(), atol=1e-5)
    finally:
        env.close()


def test_league_build_stacked_torch():
    from rl_games.envs.envpool_soccer import state_dict_to_cpu, MAIN_ID
    lg = League(max_pool=4, host_fn=state_dict_to_cpu)
    a = lg.add({'w': torch.ones(2), 'n': torch.tensor(3)}, epoch=1)
    b = lg.add({'w': torch.full((2,), 2.0), 'n': torch.tensor(4)}, epoch=2)
    main = {'w': torch.zeros(2), 'n': torch.tensor(0)}
    stacked = lg.build_stacked([a, MAIN_ID, b, 99], main)       # 99: evicted -> main
    assert set(stacked) == {'w', 'n'}
    assert torch.equal(stacked['w'], torch.tensor([[1., 1.], [0., 0.], [2., 2.], [0., 0.]]))
    assert torch.equal(stacked['n'], torch.tensor([3, 0, 4, 0]))


# ----------------------------------------------------------- shaping anneal

@needs_soccer
def test_shaping_scale_only_scales_non_goal_terms():
    env = _make(2, shaping_weights={'vel_to_ball': 0.5, 'vel_ball_to_goal': 2.0, 'veloc_forward': 0.25,
                                    'goal': 100, 'spread_out': 0.1})
    try:
        env.reset()
        raw_obs, raw_info = env.env.reset()
        env._update_perm(raw_info)
        native = np.array([1, 1, -1, -1, 0, 0, 0, 0], dtype=np.float32)   # match 0: home goal
        native = native[np.argsort(env._perm)] if env._perm is not None else native
        full, _ = env._home_rewards(raw_obs, raw_info, native)
        env.set_shaping_scale(0.5)
        half, _ = env._home_rewards(raw_obs, raw_info, native)
        env.set_shaping_scale(0.0)
        none, _ = env._home_rewards(raw_obs, raw_info, native)
        assert np.allclose(none, [[100, 100], [0, 0]])
        assert np.allclose(half - none, 0.5 * (full - none), atol=1e-4)   # float32 at ~100
        assert np.abs(full - none).max() > 0            # shaping was non-trivial
    finally:
        env.close()


def test_shaping_anneal_schedule():
    from rl_games.common.soccer_observer import SoccerObserver

    class _Env:
        def __init__(self):
            self.scales = []

        def set_shaping_scale(self, s):
            self.scales.append(float(s))

    algo = _FakeAlgo()
    algo.vec_env = _Env()
    obs = SoccerObserver(anneal_config={'start_epoch': 10, 'end_epoch': 20})
    obs.after_init(algo)
    for ep in (0, 5, 10, 15, 20, 25):
        obs.after_print_stats(frame=ep, epoch_num=ep, total_time=0.0)
    assert np.allclose(algo.vec_env.scales, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    plain = SoccerObserver()
    plain.after_init(algo)
    plain.after_print_stats(frame=1, epoch_num=1, total_time=0.0)   # no anneal: never touches the env
    assert len(algo.vec_env.scales) == 6
    floored = SoccerObserver(anneal_config={'start_epoch': 10, 'end_epoch': 20, 'floor': 0.25})
    assert np.allclose([floored.shaping_scale(e) for e in (0, 10, 15, 20, 30)], [1.0, 1.0, 0.625, 0.25, 0.25])


# ----------------------------------------------------------- population mode

@needs_soccer
def test_population_mode_rows_and_slot_onehot():
    base = _make(2)
    try:
        base_dim = base.obs_dim
    finally:
        base.close()
    env = _make(3, population=4, player_id_obs=True)
    try:
        assert env.population == 4 and env.get_number_of_agents() == 4
        assert env.get_env_info()['agents'] == 4
        assert env.obs_dim == base_dim + 2 + 4
        pairs = np.array([[0, 1], [2, 2], [3, 0]])
        env.set_pair_assignment(pairs)
        obs = env.reset()
        assert obs.shape == (12, env.obs_dim)
        assert np.array_equal(env.current_pairs(), pairs)
        slot = obs[:, -4:].argmax(dim=1).view(3, 4)
        assert torch.equal(slot, torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2], [3, 3, 0, 0]]))
        pid = obs[:, base_dim:base_dim + 2].view(3, 4, 2)
        assert torch.equal(pid[0], torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
        obs, rew, done, info = env.step(torch.zeros(12, 3))
        assert rew.shape == (12,) and done.shape == (12,) and info['time_outs'].shape == (12,)
    finally:
        env.close()


@needs_soccer
def test_population_rewards_are_per_team():
    env = _make(2, population=2, shaping_weights={'vel_to_ball': 0.5, 'vel_ball_to_goal': 2.0,
                                                  'veloc_forward': 0, 'goal': 100})
    try:
        env.set_pair_assignment(np.array([[0, 1], [1, 0]]))
        env.reset()
        raw_obs, raw_info = env.env.reset()
        env._update_perm(raw_info)
        native = np.array([1, 1, -1, -1, 0, 0, 0, 0], dtype=np.float32)    # match 0: home scored
        native = native[np.argsort(env._perm)] if env._perm is not None else native
        r, goal = env._all_rewards(raw_obs, raw_info, native)
        assert r.shape == (2, 4) and np.array_equal(goal, [1, 0])
        closest = env._stat(raw_obs, raw_info, 'stats_closest_vel_to_ball')
        vbg = env._stat(raw_obs, raw_info, 'stats_vel_ball_to_goal')
        home_c = closest[:, :2].sum(1, keepdims=True)
        away_c = closest[:, 2:].sum(1, keepdims=True)
        exp = np.concatenate([0.5 * np.repeat(home_c, 2, 1), 0.5 * np.repeat(away_c, 2, 1)], 1) + 2.0 * vbg
        exp[0, :2] += 100
        exp[0, 2:] -= 100
        assert np.allclose(r, exp.astype(np.float32), atol=1e-4)
    finally:
        env.close()


@needs_soccer
def test_population_pairs_applied_at_reset_and_reported():
    env = _make(3, population=3)
    try:
        env.set_pair_assignment(np.array([[0, 1], [1, 2], [2, 0]]))
        env.reset()
        env.set_pair_assignment(np.array([[2, 2], [0, 0], [1, 1]]))      # deferred
        for _ in range(60):
            _, _, done, info = env.step(torch.zeros(12, 3))
            if done.any():
                assert torch.equal(info['home_slot'].cpu(), torch.tensor([0, 1, 2]))
                assert torch.equal(info['away_slot'].cpu(), torch.tensor([1, 2, 0]))
                assert torch.equal(info['opp_id'].cpu(), info['away_slot'].cpu())
                assert info['goal_diff'].shape == (3,)
                d = done.view(3, 4)
                assert torch.equal(d[:, 0], d[:, 3])
                break
        else:
            raise AssertionError('no match finished')
        env.step(torch.zeros(12, 3))                                        # auto-reset applies pending
        assert np.array_equal(env.current_pairs(), [[2, 2], [0, 0], [1, 1]])
    finally:
        env.close()


def test_population_observer_payoff_and_pairs(tmp_path):
    from rl_games.common.soccer_observer import SoccerPopulationObserver

    class _Env:
        num_envs = 20
        population = 4

        def __init__(self):
            self.pairs = []

        def set_pair_assignment(self, pairs):
            self.pairs.append(np.array(pairs))

        def set_shaping_scale(self, s):
            pass

    algo = _FakeAlgo(num_agents=4)
    algo.vec_env = _Env()
    algo.nn_dir = str(tmp_path)
    obs = SoccerPopulationObserver(population_config={'remap_every': 1, 'p_self': 0.2, 'payoff_ema': 0.5,
                                                      'slot_save_every': 1000})
    obs.after_init(algo)
    pairs = algo.vec_env.pairs[-1]
    assert pairs.shape == (20, 2) and pairs.min() >= 0 and pairs.max() < 4
    assert (pairs[:, 0] == pairs[:, 1]).sum() == 4                      # 20% self matches
    # done rows are every 4th row; matches 0,1,2 finished: slot0 beat slot1 twice, drew with slot2
    infos = {'goal_diff': torch.tensor([1., 1., 0.]), 'home_goals': torch.zeros(3), 'away_goals': torch.zeros(3),
             'home_slot': torch.tensor([0, 0, 0]), 'away_slot': torch.tensor([1, 1, 2]),
             'opp_id': torch.tensor([1, 1, 2]), 'match_len': torch.full((3,), 300.)}
    obs.process_infos(infos, torch.tensor([[0], [4], [8]]))
    assert obs.counts[0, 1] == 2 and obs.counts[1, 0] == 2 and obs.counts[0, 2] == 1
    assert obs.payoff[0, 1] > 0.8 and obs.payoff[1, 0] < 0.2 and abs(obs.payoff[0, 2] - 0.5) < 1e-6
    wr = obs.slot_winrates()
    assert wr.shape == (4,) and wr[0] > wr[1]
    obs.after_print_stats(frame=1, epoch_num=1, total_time=0.0)
    assert len(algo.vec_env.pairs) == 2                                   # remapped


def test_population_config_builds_model():
    import yaml
    from rl_games.algos_torch.model_builder import ModelBuilder
    from scripts.soccer_train import register_networks
    register_networks()
    cfg = yaml.safe_load(open('rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml'))['params']
    N = cfg['network']['population_size']
    assert cfg['config']['env_config']['population'] == N
    assert cfg['config'].get('torch_compile', True) is False
    model = ModelBuilder().load(cfg).build({'actions_num': 3, 'input_shape': (113 + N,), 'num_seqs': 1,
                                            'value_size': 1, 'normalize_value': True, 'normalize_input': True})
    obs = torch.cat([torch.randn(5, 113), torch.nn.functional.one_hot(torch.tensor([0, 1, 1, 7, 3]), N).float()], 1)
    out = model({'obs': obs, 'is_train': False})
    assert out['mus'].shape == (5, 3) and out['values'].shape == (5, 1)
