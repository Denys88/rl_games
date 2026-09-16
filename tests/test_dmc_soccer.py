"""dm_control soccer self-play against envpool's native soccer contract.

Most tests run on a stub that reproduces the envpool >= 1.2.7 contract without
the physics: per-player rows (num_envs * 2 * team_size) for obs AND reward,
per-match terminated/truncated, `info["players"]["env_id"]` labelling the
rows, match blocks handed back in a rotating (non-sorted) order, and
next-step autoreset (terminal obs with done=True, the following step ignores
its action and returns the new episode's first obs with a zero reward).

The `real envpool` section at the bottom runs the same contract against the
installed envpool when it has native soccer, and is skipped otherwise.
"""
import os
import sys
import types

import gymnasium
import numpy as np
import pytest
import torch
import yaml

from rl_games.common import env_configurations
from rl_games.envs.dmc_soccer_opponents import OpponentLeague
from rl_games.envs.dmc_soccer_selfplay import (
    NATIVE_ENV_ID, SoccerSelfPlay, check_player_layout, flatten_obs, obs_keys,
    sort_batch, sort_rows)
from rl_games.envs import dmc_soccer_tools as tools

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML = os.path.join(REPO, "rl_games/configs/dm_control/boxhead_soccer_2v2_selfplay.yaml")

PLAYERS = 4
TEAM = PLAYERS // 2
ACT = 3
# 3-D positions (the scripted opponents read x, y); every other key is 1-D
_DIMS = {k: 1 for k in obs_keys(PLAYERS)}
_DIMS["ball_ego_position"] = _DIMS["team_goal_mid"] = 3
_STATS = ("stats_vel_ball_to_goal", "stats_closest_vel_to_ball")


class FakeNativeSoccer:
    """Deterministic stub of the native envpool soccer contract.

    Every per-player key holds 0.5 except joints_pos = match * 10 + player
    (row-order diagnostic) and joints_vel = episode step.
    stats_vel_ball_to_goal is 1 for every player, stats_closest_vel_to_ball
    is 1 for the first player of each team (upstream zeroes it for every
    player that is not its team's nearest to the ball).
    Episodes truncate at max_episode_steps; `goal_at` = {match: episode_step}
    scores a goal there for `scoring_team` (0 = home): reward +1 for that
    team's players, -1 for the other's, terminated=True (terminate_on_goal).

    `rotate` shifts the order the match blocks are returned in by one place
    per step, reproducing envpool's thread-completion ordering.
    """

    def __init__(self, num_envs, seed=0, max_episode_steps=600, team_size=2,
                 goal_at=None, rotate=True, scoring_team=0, **kwargs):
        self.num_envs = num_envs
        self.players = 2 * team_size
        self.team_size = team_size
        self.max_steps = max_episode_steps
        self.goal_at = goal_at or {}
        self.scoring_team = scoring_team  # 0 = home scores, 1 = away scores
        self.rotate = rotate
        self.kwargs = kwargs
        box = gymnasium.spaces.Box
        # native spaces are ONE player's: (stack, d), no players axis
        spaces = {k: box(-np.inf, np.inf, (1, d), np.float64)
                  for k, d in _DIMS.items()}
        for k in _STATS:
            spaces[k] = box(-np.inf, np.inf, (1,), np.float64)
        self.observation_space = gymnasium.spaces.Dict(spaces)
        self.action_space = box(-1.0, 1.0, (ACT,), np.float64)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.pending_reset = np.zeros(num_envs, dtype=bool)
        self.last_actions = None  # (num_envs, players, ACT), env-major
        self._shift = 0

    # --- row ordering --------------------------------------------------------

    def _env_order(self):
        return np.roll(np.arange(self.num_envs, dtype=np.int32), self._shift)

    def _pack(self, env_order, per_env_player, per_env):
        """env-major arrays -> the batch order envpool would return."""
        rows = np.concatenate([
            np.arange(e * self.players, (e + 1) * self.players)
            for e in env_order]) if self.num_envs else np.zeros(0, int)
        return ({k: v[rows] for k, v in per_env_player.items()},
                [a[env_order] for a in per_env], rows)

    def _obs(self):
        m, p = self.num_envs, self.players
        obs = {k: np.full((m * p, 1, d), 0.5) for k, d in _DIMS.items()}
        obs["joints_pos"] = (np.arange(m)[:, None] * 10
                             + np.arange(p)[None, :]).reshape(m * p, 1, 1).astype(float)
        obs["joints_vel"] = np.repeat(self.t, p).reshape(m * p, 1, 1).astype(float)
        obs["stats_vel_ball_to_goal"] = np.ones((m * p, 1))
        closest = np.zeros((m, p))
        closest[:, 0] = closest[:, self.team_size] = 1.0
        obs["stats_closest_vel_to_ball"] = closest.reshape(m * p, 1)
        return obs

    def _info(self, env_order):
        return {
            "env_id": env_order,
            "players": {"env_id": np.repeat(env_order, self.players)},
            "elapsed_step": self.t[env_order],
        }

    # --- gymnasium API -------------------------------------------------------

    def reset(self):
        self.t[:] = 0
        self.pending_reset[:] = False
        self._shift = 0
        order = self._env_order()
        obs, _, _ = self._pack(order, self._obs(), [])
        return obs, self._info(order)

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs * self.players, ACT), actions.shape
        # actions are env-major regardless of the batch order we return
        self.last_actions = actions.reshape(
            self.num_envs, self.players, ACT).copy()
        reward = np.zeros((self.num_envs, self.players), dtype=np.float32)
        term = np.zeros(self.num_envs, dtype=bool)
        trunc = np.zeros(self.num_envs, dtype=bool)
        for e in range(self.num_envs):
            if self.pending_reset[e]:  # reset step: action ignored, reward 0
                self.pending_reset[e] = False
                self.t[e] = 0
                continue
            self.t[e] += 1
            if self.goal_at.get(e) == self.t[e]:
                sign = 1.0 if self.scoring_team == 0 else -1.0
                reward[e, :self.team_size] = sign
                reward[e, self.team_size:] = -sign
                term[e] = True
            trunc[e] = self.t[e] >= self.max_steps
            self.pending_reset[e] = term[e] or trunc[e]
        if self.rotate:
            self._shift = (self._shift + 1) % max(self.num_envs, 1)
        order = self._env_order()
        obs, (rew, te, tr), _ = self._pack(
            order, self._obs(),
            [reward.reshape(-1, self.players), term, trunc])
        return (obs, rew.reshape(-1), te, tr, self._info(order))


@pytest.fixture
def fake_envpool(monkeypatch):
    mod = types.ModuleType("envpool")
    mod.made = []

    def make_gymnasium(env_name, num_envs, **kwargs):
        mod.made.append(env_name)
        return FakeNativeSoccer(num_envs, **kwargs)

    mod.make_gymnasium = make_gymnasium
    monkeypatch.setitem(sys.modules, "envpool", mod)
    return mod


def make_env(**kwargs):
    kwargs.setdefault("num_actors", 4)
    kwargs.setdefault("max_episode_steps", 50)
    return SoccerSelfPlay("dmc_soccer_selfplay", **kwargs)


# --- native row ordering -----------------------------------------------------

def test_sort_rows_is_identity_only_for_a_sorted_batch():
    assert sort_rows(np.array([0, 1, 2]), 3, 2) is None
    # match blocks returned as [2, 0, 1]: block 0 (match 2) lands at rows 4,5
    np.testing.assert_array_equal(
        sort_rows(np.array([2, 0, 1]), 3, 2), [4, 5, 0, 1, 2, 3])


def test_sort_batch_restores_env_major_rows_and_per_match_flags():
    # 3 matches x 2 players, returned as blocks [1, 2, 0]
    info = {"env_id": np.array([1, 2, 0]),
            "players": {"env_id": np.repeat([1, 2, 0], 2)}}
    check_player_layout(info, 3, 2)
    obs = {"k": np.array([10, 11, 20, 21, 0, 1])[:, None, None]}
    reward = np.array([10., 11., 20., 21., 0., 1.])
    term = np.array([False, True, False])  # match 1 no, match 2 yes, match 0 no
    obs, reward, term = sort_batch(obs, info, 3, 2, reward, term)
    np.testing.assert_array_equal(obs["k"].ravel(), [0, 1, 10, 11, 20, 21])
    np.testing.assert_array_equal(reward, [0, 1, 10, 11, 20, 21])
    np.testing.assert_array_equal(term, [False, False, True])  # match 2 done


def test_check_player_layout_rejects_a_non_block_player_batch():
    info = {"env_id": np.array([0, 1]),
            "players": {"env_id": np.array([0, 1, 0, 1])}}  # interleaved
    with pytest.raises(AssertionError, match="contiguous rows per match"):
        check_player_layout(info, 2, 2)


def test_rows_stay_env_major_across_rotating_batches(fake_envpool):
    env = make_env(num_actors=8, opponent="self", rotate=True)
    obs = env.reset()
    assert obs[:, 0].tolist() == [0, 1, 2, 3, 10, 11, 12, 13]
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((8, ACT)))
        # unchanged row identity every step, despite the rotating block order
        assert obs[:, 0].tolist() == [0, 1, 2, 3, 10, 11, 12, 13]


def test_actions_are_written_env_major_not_in_batch_order(fake_envpool):
    env = make_env(num_actors=8, opponent="self", rotate=True)
    env.reset()
    for _ in range(4):
        a = np.arange(8 * ACT, dtype=float).reshape(8, ACT)
        env.step(a)
        # the stub records actions env-major; row r of the adapter must be
        # match r // players, player r % players
        np.testing.assert_array_equal(
            env.env.last_actions.reshape(8, ACT), a)


# --- autoreset contract ----------------------------------------------------

def test_declares_next_step_autoreset_on_the_single_agent_path(fake_envpool):
    info = make_env(num_actors=8, opponent="self").get_env_info()
    assert info["autoreset_mode"] == "next_step"
    assert info["agents"] == 1  # the trainer's multi-agent guard is not hit


def test_done_rows_align_with_player_rows(fake_envpool):
    # 3 matches x 2 controlled home players = 6 rows; match 1 scores at
    # episode step 2. The trainer masks row r on the step AFTER done[r], so
    # done must repeat per player in the same match-major, player-minor order
    # as the obs rows.
    env = make_env(num_actors=6, opponent="random", goal_at={1: 2},
                   max_episode_steps=10)
    obs = env.reset()
    assert obs[:, 0].tolist() == [0, 1, 10, 11, 20, 21]  # (match, player)
    a = np.zeros((6, ACT))
    _, _, done, _ = env.step(a)
    assert not done.any()

    obs, rew, done, info = env.step(a)  # terminal step for match 1
    assert done.tolist() == [False, False, True, True, False, False]
    assert obs[2:4, 1].tolist() == [2, 2]  # terminal obs, not a reset obs
    assert info["scores"][2:4].tolist() == [1, 1]
    # goal 150 + dense (0.5 * 1 ball + 0.25 * 1 shared chase) - 0.05 time
    np.testing.assert_allclose(rew[2:4], 150 + 0.5 + 0.25 - 0.05, rtol=1e-6)
    np.testing.assert_allclose(rew[[0, 1, 4, 5]], 0.5 + 0.25 - 0.05, rtol=1e-6)

    obs, rew, done, info = env.step(a)  # reset step for match 1
    assert not done.any()  # the row the mask (built from the previous done) drops
    assert obs[2:4, 1].tolist() == [0, 0]  # new episode's first obs
    assert obs[[0, 1, 4, 5], 1].tolist() == [3, 3, 3, 3]
    assert info["scores"][2:4].tolist() == [0, 0]


def test_away_goal_is_a_negative_home_reward_not_a_score(fake_envpool):
    # the away team scoring gives the home rows -1; goal_w_concede is 0 by
    # default, so it must not subtract from the shaped reward, and it must
    # not count toward `scores` when the away team is an opponent
    env = make_env(num_actors=2, opponent="random", goal_at={0: 1},
                   scoring_team=1, max_episode_steps=10)
    env.reset()
    _, rew, done, info = env.step(np.zeros((2, ACT)))
    assert done.tolist() == [True, True]
    np.testing.assert_allclose(rew, 0.5 + 0.25 - 0.05, rtol=1e-6)  # no -150
    assert info["scores"].tolist() == [0, 0]  # conceding is not a home goal


def test_missing_native_keys_point_at_the_envpool_requirement(fake_envpool):
    # a build without the native per-player soccer keys must say so
    real = fake_envpool.make_gymnasium

    def crippled(env_name, num_envs, **kwargs):
        env = real(env_name, num_envs, **kwargs)
        env.observation_space.spaces.pop("opponent_0_ego_position")
        return env
    fake_envpool.make_gymnasium = crippled
    with pytest.raises(AssertionError, match=r"envpool >= 1\.2\.7"):
        make_env(num_actors=8, opponent="self")


def test_default_env_id_is_the_native_one(fake_envpool):
    make_env(num_actors=8, opponent="self")
    assert fake_envpool.made == [NATIVE_ENV_ID] == ["DmcSoccerBoxhead-v1"]


def test_team_size_drives_players_and_controlled_rows(fake_envpool):
    solo = make_env(num_actors=4, opponent="random", team_size=1)
    assert (solo.players, solo.team_size, solo.controlled, solo.num_matches) == (2, 1, 1, 4)
    duo = make_env(num_actors=4, opponent="random", team_size=2)
    assert (duo.players, duo.team_size, duo.controlled, duo.num_matches) == (4, 2, 2, 2)
    assert solo.obs_dim < duo.obs_dim  # one fewer opponent, no teammate keys


def test_task_options_reach_envpool(fake_envpool):
    env = make_env(num_actors=8, opponent="self", team_size=2,
                   terminate_on_goal=False, time_limit=30.0,
                   enable_field_box=True)
    assert env.env.kwargs["terminate_on_goal"] is False
    assert env.env.kwargs["time_limit"] == 30.0
    assert env.env.kwargs["enable_field_box"] is True
    assert env.env.team_size == 2  # team_size is forwarded, not a task option


# --- feature layout ------------------------------------------------------------

def test_obs_keys_follow_the_upstream_teammate_opponent_layout():
    solo = obs_keys(2)
    assert "teammate_0_ego_position" not in solo  # 1v1: no teammate
    assert "opponent_0_ego_position" in solo and "opponent_1_ego_position" not in solo
    duo = obs_keys(4)
    assert "teammate_0_ego_position" in duo
    assert {"opponent_0_ego_position", "opponent_1_ego_position"} <= set(duo)
    assert duo[-1] == "field_back_right" and duo[0] == "joints_pos"
    assert len(set(duo)) == len(duo)


def test_eval_tools_flatten_is_the_training_flatten(fake_envpool):
    assert tools.flatten_obs is flatten_obs
    env = make_env(num_actors=4, opponent="self")
    obs, info = env.env.reset()
    (obs,) = sort_batch(obs, info, env.num_matches, env.players)
    obs["joints_vel"] = obs["joints_vel"].copy()
    obs["joints_vel"][0, 0, 0] = np.nan
    obs["joints_vel"][1, 0, 0] = 5e3
    obs["joints_vel"][2, 0, 0] = -np.inf
    shared = flatten_obs(obs, env.num_matches, env.players)
    assert shared.shape == (1, PLAYERS, env.obs_dim) and shared.dtype == np.float32
    np.testing.assert_array_equal(env._flatten_obs(obs), shared.reshape(4, -1))
    assert shared[0, :3, 1].tolist() == [0.0, 1e3, 0.0]  # NaN/inf zeroed, clipped
    # within-team one-hot slot: home_i and away_i share an id
    assert shared[0, :, -TEAM:].tolist() == [[1, 0], [0, 1], [1, 0], [0, 1]]


def test_config_player_uses_the_vecenv_registry():
    # no env_creator in the registry entry: BasePlayer's default path
    # (create_env) raises KeyError; player.use_vecenv is what it reads
    assert "env_creator" not in env_configurations.configurations["dmc_soccer_selfplay"]
    with open(YAML) as f:
        cfg = yaml.safe_load(f)["params"]["config"]
    assert cfg["player"]["use_vecenv"] is True


def test_shipped_config_names_the_native_env(fake_envpool):
    with open(YAML) as f:
        env_cfg = yaml.safe_load(f)["params"]["config"]["env_config"]
    assert env_cfg["env_name"] == NATIVE_ENV_ID
    assert env_cfg["team_size"] == 2
    assert "players_per_match" not in env_cfg  # the fork's key is gone


# --- resume state ------------------------------------------------------------

@pytest.mark.parametrize("opponent, extra", [
    ("random", {}),
    ("league", {"league_types": ["random", "chaser"]}),
])
def test_env_state_roundtrip_restores_anneal_and_opponent_rngs(
        fake_envpool, opponent, extra):
    kw = dict(num_actors=4, opponent=opponent, seed=3,
              dense_anneal_steps=100, **extra)
    src = make_env(**kw)
    src.reset()
    a = np.zeros((4, ACT))
    for _ in range(7):
        src.step(a)
    state = src.get_env_state()
    assert state["anneal_step"] == 7
    assert ("league_rng" in state) == (opponent == "league")

    dst = make_env(**kw)
    dst.reset()
    dst.set_env_state(state)
    assert dst.get_env_state()["anneal_step"] == 7
    # the next away-team draw is the one src would have made
    src.step(a)
    dst.step(a)
    np.testing.assert_array_equal(src.env.last_actions[:, TEAM:],
                                  dst.env.last_actions[:, TEAM:])
    assert dst.get_env_state()["anneal_step"] == 8


def test_restored_anneal_step_drives_dense_shaping(fake_envpool):
    kw = dict(num_actors=4, opponent="self", dense_anneal_steps=100,
              dense_floor=0.15, vel_ball_w=0.5, vel_player_w=0.25)
    fresh = make_env(**kw)
    fresh.reset()
    resumed = make_env(**kw)
    resumed.reset()
    resumed.set_env_state({"anneal_step": 10 ** 6})  # past the anneal: floor
    a = np.zeros((4, ACT))
    _, r_fresh, _, _ = fresh.step(a)
    _, r_resumed, _, _ = resumed.step(a)
    # fresh: dense = 1 - 1/100 after its first step; resumed: dense_floor
    expected = (0.99 - 0.15) * (0.5 * 1.0 + 0.25 * 1.0)
    np.testing.assert_allclose(r_fresh - r_resumed, expected, rtol=1e-5)


@pytest.mark.parametrize("state", [
    None, {}, {"anneal_step": 5},
    {"anneal_step": 5, "league_rng": np.random.RandomState(1).get_state()},
])
def test_set_env_state_tolerates_older_checkpoints(fake_envpool, state):
    env = make_env(num_actors=4, opponent="random")  # no league to restore
    env.reset()
    env.set_env_state(state)
    assert env.get_env_state()["anneal_step"] == (state or {}).get("anneal_step", 0)


# --- opponent league ---------------------------------------------------------

OBS_DIM = 2


def write_ckpt(path, scale, mtime):
    """Minimal rl_games actor checkpoint FrozenPolicy accepts: one hidden
    Linear of 3 units, all weights = scale, pinned to a distinct mtime (same-
    second rewrites are indistinguishable on coarse filesystems)."""
    model = {
        "running_mean_std.running_mean": torch.zeros(OBS_DIM),
        "running_mean_std.running_var": torch.ones(OBS_DIM),
        "a2c_network.actor_mlp.0.weight": torch.full((3, OBS_DIM), scale),
        "a2c_network.actor_mlp.0.bias": torch.zeros(3),
        "a2c_network.mu.weight": torch.full((3, 3), scale),
        "a2c_network.mu.bias": torch.zeros(3),
    }
    torch.save({"model": model}, path)
    os.utime(path, (mtime, mtime))


def write_garbage(path, mtime):
    with open(path, "wb") as f:
        f.write(b"not a checkpoint")
    os.utime(path, (mtime, mtime))


def away_actions(league):
    obs = {"ball_ego_position": np.ones((1, TEAM, 3)),
           "team_goal_mid": np.ones((1, TEAM, 3))}
    return league.actions(obs, np.ones((1, TEAM, OBS_DIM), dtype=np.float32))


def test_scripted_opponents_steer_toward_their_target():
    # steer is NEGATIVE the bearing: envpool's boxhead turns CLOCKWISE for
    # steer > 0, which GROWS a landmark's egocentric bearing. The opposite
    # sign makes the chaser run away from the ball.
    from rl_games.envs.dmc_soccer_opponents import chaser
    left = {"ball_ego_position": np.array([[[1.0, 1.0, 0.0]]])}   # bearing +45
    right = {"ball_ego_position": np.array([[[1.0, -1.0, 0.0]]])}  # bearing -45
    assert chaser(left)[..., 1] < 0 and chaser(right)[..., 1] > 0


def test_scripted_league_types_require_the_boxhead_action_space():
    with pytest.raises(AssertionError, match="3-DoF boxhead"):
        OpponentLeague(1, types=["chaser"], act_dim=8)
    OpponentLeague(1, types=["random", "zero"], act_dim=8)  # fine


def test_league_action_width_follows_act_dim():
    league = OpponentLeague(1, types=["random"], act_dim=8)
    obs = {"ball_ego_position": np.ones((1, 1, 3)),
           "team_goal_mid": np.ones((1, 1, 3))}
    assert league.actions(obs, np.ones((1, 1, OBS_DIM), np.float32)).shape == (1, 1, 8)


def test_league_reloads_best_checkpoint_overwritten_in_place(tmp_path):
    # <name>.pth (the best checkpoint) keeps its path when the trainer
    # overwrites it; the league must follow the new weights
    best = str(tmp_path / "best.pth")
    write_ckpt(best, scale=0.1, mtime=1000)
    league = OpponentLeague(1, types=["league_latest"], ckpt_dir=str(tmp_path),
                            refresh_every=1)
    a_first = away_actions(league)
    write_ckpt(best, scale=0.2, mtime=1010)
    a_second = away_actions(league)
    assert league._latest.mtime == 1010
    assert not np.allclose(a_first, a_second)
    assert np.allclose(away_actions(league), a_second)  # unchanged file: kept


def test_league_load_failure_warns_once_per_path_and_error(tmp_path, capsys):
    bad = str(tmp_path / "bad.pth")
    write_garbage(bad, mtime=1000)
    league = OpponentLeague(1, types=["league_latest"], ckpt_dir=str(tmp_path))
    league._refresh_league()
    league._refresh_league()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "bad.pth" in out
    assert league._latest is None

    other = str(tmp_path / "other.pth")  # a second failing path warns itself
    write_garbage(other, mtime=1010)
    league._refresh_league()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "other.pth" in out

    write_ckpt(other, scale=0.1, mtime=1020)  # loads: its record clears
    league._refresh_league()
    assert capsys.readouterr().out == ""
    assert league._latest.path == other

    write_garbage(other, mtime=1030)  # the same failure again: reported again
    league._refresh_league()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "other.pth" in out
    assert league._latest.path == other  # previous weights kept


def test_league_warns_at_construction_without_ckpt_dir(capsys):
    OpponentLeague(2, types=["chaser", "league_old"], ckpt_dir=None)
    assert "league_ckpt_dir" in capsys.readouterr().out
    OpponentLeague(2, types=["chaser", "random"], ckpt_dir=None)
    assert capsys.readouterr().out == ""


def test_league_warns_once_when_ckpt_dir_missing(tmp_path, capsys):
    missing = str(tmp_path / "nope")
    league = OpponentLeague(1, types=["league_latest"], ckpt_dir=missing)
    league._refresh_league()
    league._refresh_league()
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "nope" in out
    # present but empty is the documented warmup before the first save
    OpponentLeague(1, types=["league_latest"],
                   ckpt_dir=str(tmp_path))._refresh_league()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("every, calls, expected", [
    (1, 3, [1, 2, 3]),
    (500, 502, [1, 501]),
])
def test_league_refresh_schedule(monkeypatch, every, calls, expected):
    league = OpponentLeague(1, types=["random"], refresh_every=every)
    seen = []
    monkeypatch.setattr(league, "_refresh_league",
                        lambda: seen.append(league._step))
    for _ in range(calls):
        away_actions(league)
    assert seen == expected


# --- tools -------------------------------------------------------------------

def test_tools_default_run_dir_is_the_shipped_config_checkpoint_dir():
    with open(YAML) as f:
        cfg = yaml.safe_load(f)["params"]["config"]
    assert tools.DEFAULT_RUN_DIR == cfg["env_config"]["league_ckpt_dir"]
    assert tools.DEFAULT_RUN_DIR == f"runs/{cfg['full_experiment_name']}/nn"


def test_pick_checkpoints_raises_on_missing_or_empty_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="--run-dir"):
        tools.pick_checkpoints(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="--run-dir"):
        tools.pick_checkpoints(str(tmp_path))
    for ep in (200, 400, 600, 800):
        (tmp_path / f"last_x_ep_{ep}_rew_1.0.pth").touch()
    names = [n for n, _ in tools.pick_checkpoints(str(tmp_path))]
    assert names == ["ckpt_ep200", "ckpt_ep600", "ckpt_ep800"]


# --- real envpool ------------------------------------------------------------

def _has_native_soccer():
    try:
        import envpool
    except Exception:
        return False
    try:
        return NATIVE_ENV_ID in envpool.list_all_envs()
    except Exception:
        return False


real_envpool = pytest.mark.skipif(
    not _has_native_soccer(),
    reason=f"envpool with native {NATIVE_ENV_ID} is not installed")


@real_envpool
@pytest.mark.parametrize("team_size, opponent, num_actors, matches", [
    (1, "self", 8, 4),
    (1, "random", 4, 4),
    (2, "self", 8, 2),
    (2, "league", 8, 4),
])
def test_real_reset_and_step_shapes(team_size, opponent, num_actors, matches):
    kw = {"league_types": ["zero", "chaser"]} if opponent == "league" else {}
    env = SoccerSelfPlay("dmc_soccer_selfplay", num_actors, team_size=team_size,
                         opponent=opponent, seed=1, max_episode_steps=20,
                         num_threads=2, **kw)
    try:
        assert (env.num_matches, env.players, env.act_dim) == (
            matches, 2 * team_size, 3)
        obs = env.reset()
        assert obs.shape == (num_actors, env.obs_dim) and obs.dtype == np.float32
        assert np.isfinite(obs).all()
        for _ in range(3):
            obs, rew, done, info = env.step(
                np.zeros((num_actors, env.act_dim), np.float32))
            assert obs.shape == (num_actors, env.obs_dim)
            assert rew.shape == (num_actors,) and rew.dtype == np.float32
            assert done.shape == (num_actors,) and done.dtype == bool
            assert info["time_outs"].shape == (num_actors,)
            assert np.isfinite(obs).all() and np.isfinite(rew).all()
    finally:
        del env


@real_envpool
def test_real_batch_is_permuted_and_the_adapter_sorts_it():
    """The permutation this adapter exists to undo must actually occur."""
    import envpool
    env = envpool.make_gymnasium(NATIVE_ENV_ID, num_envs=16, team_size=2,
                                 seed=0, num_threads=4)
    try:
        _, info = env.reset()
        orders = {tuple(np.asarray(info["env_id"]).tolist())}
        for _ in range(30):
            _, _, _, _, info = env.step(np.zeros((16 * 4, 3)))
            orders.add(tuple(np.asarray(info["env_id"]).tolist()))
            # whatever the order, the blocks are contiguous and env-labelled
            check_player_layout(info, 16, 4)
        assert len(orders) > 1, "expected envpool to permute the batch order"
        assert any(o != tuple(range(16)) for o in orders)
    finally:
        del env


@real_envpool
@pytest.mark.parametrize("team_size", [1, 2])
def test_real_home_and_away_actions_are_routed_by_row(team_size):
    """prev_action is a per-player observable: it proves the routing."""
    players = 2 * team_size
    env = SoccerSelfPlay("dmc_soccer_selfplay", 4 * team_size,
                         team_size=team_size, opponent="league",
                         league_types=["zero"], seed=2,
                         max_episode_steps=200, num_threads=2)
    try:
        env.reset()
        n = 4 * team_size
        for t in range(4):
            a = np.full((n, 3), 0.25 * (t + 1))
            env.step(a)
            prev = env._last_obs_dict["prev_action"].reshape(
                env.num_matches, players, 3)
            np.testing.assert_allclose(
                prev[:, :env.controlled].reshape(n, 3), a, atol=1e-6)
            # the "zero" league type plays all-zero actions for the away team
            np.testing.assert_allclose(prev[:, env.controlled:], 0.0, atol=1e-6)
    finally:
        del env


@real_envpool
def test_real_env_state_round_trip_reproduces_the_away_team():
    kw = dict(team_size=2, opponent="league", league_types=["random"],
              seed=5, dense_anneal_steps=100, max_episode_steps=200,
              num_threads=2)
    src = SoccerSelfPlay("dmc_soccer_selfplay", 8, **kw)
    dst = SoccerSelfPlay("dmc_soccer_selfplay", 8, **kw)
    try:
        src.reset()
        dst.reset()
        a = np.zeros((8, 3), np.float32)
        for _ in range(6):
            src.step(a)
        state = src.get_env_state()
        assert state["anneal_step"] == 6
        dst.set_env_state(state)
        src.step(a)
        dst.step(a)
        # the away team is drawn from the restored RNG: identical actions
        shape = (src.num_matches, src.players, 3)
        np.testing.assert_allclose(
            src._last_obs_dict["prev_action"].reshape(shape)[:, src.controlled:],
            dst._last_obs_dict["prev_action"].reshape(shape)[:, dst.controlled:],
            atol=1e-6)
        assert dst.get_env_state()["anneal_step"] == 7
    finally:
        del src, dst


@real_envpool
def test_real_goal_signal_is_a_per_player_plus_minus_one_and_terminates():
    """A scripted striker until the first goal; the signal contract is
    reward = +1 for every player of the scoring team, -1 for the other,
    stats_home_score/stats_away_score the same event per player, and (with
    terminate_on_goal) terminated=True for that MATCH -- never per player."""
    import envpool
    num_envs, team_size = 256, 1
    players = 2 * team_size
    rows = num_envs * players
    env = envpool.make_gymnasium(
        NATIVE_ENV_ID, num_envs=num_envs, team_size=team_size, seed=13,
        terminate_on_goal=True, max_episode_steps=1500, num_threads=8)
    try:
        obs, info = env.reset()
        for _ in range(1500):
            b = obs["ball_ego_position"].reshape(rows, 3)[:, :2]
            g = obs["opponent_goal_mid"].reshape(rows, 3)[:, :2]
            d = g - b
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            dist = np.linalg.norm(b, axis=1)
            tgt = np.where((dist < 2.5)[:, None], b + 2.0 * d, b - 1.5 * d)
            ang = np.arctan2(tgt[:, 1], tgt[:, 0])
            act = np.stack([np.where(np.abs(ang) < 0.7, -1.0, 0.0),
                            -np.clip(2.0 * ang, -1, 1),
                            (dist < 1.2).astype(float)], axis=-1)
            obs, reward, term, trunc, info = env.step(act)
            if np.any(reward != 0):
                break
        else:
            pytest.skip("no goal scored within the step budget")

        check_player_layout(info, num_envs, players)
        obs, reward, term, trunc = sort_batch(
            obs, info, num_envs, players, reward, term, trunc)
        per_match = reward.reshape(num_envs, players)
        scored = np.nonzero(per_match.any(axis=1))[0]
        assert scored.size >= 1
        for m in scored:
            r = per_match[m]
            # exactly one team gets +1, the other -1
            assert sorted(np.unique(r).tolist()) == [-1.0, 1.0]
            assert abs(r.sum()) < 1e-6 and np.abs(r).sum() == players
            # the stats flags name the same event, per player
            home = obs["stats_home_score"].reshape(num_envs, players)[m]
            away = obs["stats_away_score"].reshape(num_envs, players)[m]
            np.testing.assert_array_equal(home, (r > 0).astype(float))
            np.testing.assert_array_equal(away, (r < 0).astype(float))
            # terminate_on_goal: the MATCH ends, and it is a termination
            assert term[m] and not trunc[m]
        # matches without a goal neither score nor terminate on this step
        quiet = np.setdiff1d(np.arange(num_envs), scored)
        assert not term[quiet].any()
    finally:
        del env
