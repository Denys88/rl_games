"""dm_control soccer self-play against a fake fork-shaped envpool.

No soccer envpool build is pip-installable, so the Denys88/envpool#1
BoxheadSoccer2v2-v1 contract is stubbed: match-major obs (num_envs,
players, ...), (num_envs, players, 3) actions, per-player
info['players_reward'], next-step autoreset (terminal obs with done=True,
the following step ignores its action, returns the new episode's first obs
and a zero reward). Physics values are not reproduced.
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
from rl_games.envs.dmc_soccer_selfplay import SoccerSelfPlay, _OBS_KEYS, flatten_obs
from rl_games.envs import dmc_soccer_tools as tools

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML = os.path.join(REPO, "rl_games/configs/dm_control/boxhead_soccer_2v2_selfplay.yaml")

PLAYERS = 4
TEAM = PLAYERS // 2
# 3-D positions (the scripted opponents read x, y); every other key is 1-D
_DIMS = {k: 1 for k in _OBS_KEYS}
_DIMS["ball_ego_position"] = _DIMS["team_goal_mid"] = 3


class FakeForkSoccer:
    """Deterministic fork-contract stub.

    Every per-player key holds 0.5 except joints_pos = match * 10 + player
    (row-order diagnostic) and joints_vel = episode step.
    stats_vel_ball_to_goal is 1 for every player, stats_closest_vel_to_ball
    is 1 for the first player of each team (the env's one-nonzero-per-team
    convention). Episodes truncate at max_episode_steps; `goal_at` =
    {match: episode_step} scores a home goal there (terminates).
    """

    def __init__(self, num_envs, seed=0, max_episode_steps=600,
                 goal_at=None, **kwargs):
        self.num_envs = num_envs
        self.max_steps = max_episode_steps
        self.goal_at = goal_at or {}
        box = gymnasium.spaces.Box
        spaces = {k: box(-np.inf, np.inf, (PLAYERS, d), np.float64)
                  for k, d in _DIMS.items()}
        for k in ("stats_vel_ball_to_goal", "stats_closest_vel_to_ball"):
            spaces[k] = box(-np.inf, np.inf, (PLAYERS,), np.float64)
        self.observation_space = gymnasium.spaces.Dict(spaces)
        self.action_space = box(-1.0, 1.0, (PLAYERS, 3), np.float64)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.pending_reset = np.zeros(num_envs, dtype=bool)
        self.last_actions = None

    def _obs(self):
        m, p = self.num_envs, PLAYERS
        obs = {k: np.full((m, p, d), 0.5) for k, d in _DIMS.items()}
        obs["joints_pos"] = (np.arange(m)[:, None] * 10
                             + np.arange(p)[None, :])[..., None].astype(float)
        obs["joints_vel"] = np.broadcast_to(
            self.t[:, None, None], (m, p, 1)).astype(float)
        obs["stats_vel_ball_to_goal"] = np.ones((m, p))
        closest = np.zeros((m, p))
        closest[:, 0] = closest[:, TEAM] = 1.0
        obs["stats_closest_vel_to_ball"] = closest
        return obs

    def reset(self):
        self.t[:] = 0
        self.pending_reset[:] = False
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, PLAYERS, 3), actions.shape
        self.last_actions = actions.copy()
        reward = np.zeros((self.num_envs, PLAYERS))
        term = np.zeros(self.num_envs, dtype=bool)
        trunc = np.zeros(self.num_envs, dtype=bool)
        for e in range(self.num_envs):
            if self.pending_reset[e]:  # reset step: action ignored, reward 0
                self.pending_reset[e] = False
                self.t[e] = 0
                continue
            self.t[e] += 1
            if self.goal_at.get(e) == self.t[e]:
                reward[e, :TEAM] = 1.0
                reward[e, TEAM:] = -1.0
                term[e] = True
            trunc[e] = self.t[e] >= self.max_steps
            self.pending_reset[e] = term[e] or trunc[e]
        return self._obs(), reward[:, 0], term, trunc, {"players_reward": reward}


@pytest.fixture
def fake_envpool(monkeypatch):
    mod = types.ModuleType("envpool")
    mod.made = []

    def make_gymnasium(env_name, num_envs, **kwargs):
        mod.made.append(env_name)
        return FakeForkSoccer(num_envs, **kwargs)

    mod.make_gymnasium = make_gymnasium
    monkeypatch.setitem(sys.modules, "envpool", mod)
    return mod


def make_env(**kwargs):
    kwargs.setdefault("num_actors", 4)
    kwargs.setdefault("max_episode_steps", 50)
    return SoccerSelfPlay("dmc_soccer_selfplay", **kwargs)


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
    a = np.zeros((6, 3))
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


def test_players_assert_names_the_fork_env_id(fake_envpool):
    # the stub has 4 players; a mismatch must point at the required build
    with pytest.raises(AssertionError, match="Denys88/envpool#1.*BoxheadSoccer2v2-v1"):
        make_env(num_actors=6, opponent="self", players_per_match=6)


# --- feature layout ------------------------------------------------------------

def test_eval_tools_flatten_is_the_training_flatten(fake_envpool):
    assert tools.flatten_obs is flatten_obs
    env = make_env(num_actors=4, opponent="self")
    obs, _ = env.env.reset()
    obs["joints_vel"] = obs["joints_vel"].copy()
    obs["joints_vel"][0, 0, 0] = np.nan
    obs["joints_vel"][0, 1, 0] = 5e3
    obs["joints_vel"][0, 2, 0] = -np.inf
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
    a = np.zeros((4, 3))
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
    a = np.zeros((4, 3))
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
