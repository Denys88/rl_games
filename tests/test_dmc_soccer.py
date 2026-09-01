"""dm_control soccer self-play against a fake fork-shaped envpool.

No soccer envpool build is pip-installable, so the Denys88/envpool#1
BoxheadSoccer2v2-v1 contract is stubbed: match-major obs (num_envs,
players, ...), (num_envs, players, 3) actions, per-player
info['players_reward'], next-step autoreset (terminal obs with done=True,
the following step ignores its action, returns the new episode's first obs
and a zero reward). Physics values are not reproduced.
"""
import sys
import types

import gymnasium
import numpy as np
import pytest

from rl_games.envs.dmc_soccer_selfplay import SoccerSelfPlay, _OBS_KEYS

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
