"""Symmetric self-play vecenv adapter: envpool dm_control soccer -> rl_games.

Targets envpool's NATIVE soccer support (>= 1.2.7): `DmcSoccerBoxhead-v1`,
`DmcSoccerAnt-v1`, `DmcSoccerHumanoid-v1`. No fork is needed any more.

One envpool env is one match of `2 * team_size` players. This adapter exposes
every controlled player as an independent actor sharing ONE policy (symmetric
self-play): observations are egocentric and team-relative (team_goal_*,
opponent_goal_*, teammate_i_*, opponent_i_*), so the same policy plays both
home and away. rl_games sees num_actors = num_matches * controlled players.

Native envpool row contract (verified against envpool 1.2.7, see
docs/DMC_SOCCER_SELFPLAY.md):
  * every per-player array -- obs values AND the reward -- is batched over
    `num_envs * 2 * team_size` rows; `terminated`/`truncated` stay per-MATCH,
    one entry per env.
  * the rows come back grouped per env, `2 * team_size` contiguous rows per
    env, home team first; `info["players"]["env_id"]` labels each row and
    `info["env_id"]` labels each match row.
  * BUT the env blocks are returned in thread-completion order, NOT sorted:
    `info["env_id"]` is an arbitrary permutation that changes every step. The
    adapter re-sorts every batch back to env-major order so an rl_games row
    means the same match for the whole run (the trainer's per-row episode
    bookkeeping and RNN states require that).
  * actions, by contrast, are ALWAYS read in sorted env-major order when
    `env_id` is left at None -- envpool fills it with `arange(num_envs)`. So
    actions are written env-major and only the returned batch is permuted.

Reward shaping (the speed lever vs the sparse DeepMind setup):
    r = goal_w_score * max(player_reward, 0)         # scoring, concede unpunished
      + dense(t) * vel_ball_w   * max(vel_ball_to_goal, 0)   # one-sided ball progress
      + dense(t) * vel_player_w * team_chase                  # closest player, shared
      - time_w                                                # finish games
where dense(t) anneals 1 -> dense_floor over dense_anneal_steps env steps so
the goal term dominates once chase/dribble are bootstrapped. The dense terms
come straight from the env's stats observations. See docs/DMC_SOCCER_SELFPLAY.md
for the failure modes each piece of this prevents.
"""

import functools

import gymnasium
import numpy as np

from rl_games.common.ivecenv import IVecEnv

# envpool >= 1.2.7 registers soccer natively; boxhead is the 3-action walker
# the shipped config and the scripted opponents are written for.
NATIVE_ENV_ID = "DmcSoccerBoxhead-v1"

# per-player observation keys, in the upstream env's own order (see the
# `arena_keys` list and the walker/ball observables in envpool's soccer.cc).
# `_OTHER_SUFFIXES` is instantiated per teammate and per opponent: upstream
# names them teammate_{i}_* / opponent_{i}_*, so who is a teammate is carried
# by the key layout itself and needs no is_teammate flag.
_WALKER_KEYS = [
    "joints_pos", "joints_vel", "body_height", "end_effectors_pos",
    "world_zaxis", "sensors_velocimeter", "sensors_gyro",
    "sensors_accelerometer", "prev_action",
    "ball_ego_position", "ball_ego_linear_velocity",
    "ball_ego_angular_velocity",
]
_OTHER_SUFFIXES = (
    "ego_position", "ego_linear_velocity", "ego_end_effectors_pos",
    "ego_orientation", "end_effectors_pos",
)
_ARENA_KEYS = [
    "team_goal_back_right", "team_goal_mid", "team_goal_front_left",
    "field_front_left", "opponent_goal_back_left", "opponent_goal_mid",
    "opponent_goal_front_right", "field_back_right",
]


@functools.lru_cache(maxsize=None)
def obs_keys(players):
    """Policy-input keys for a `players`-player match, in a fixed order.

    A player sees `team_size - 1` teammates and `team_size` opponents, so the
    key list -- and the flat obs width -- depends on team size.
    """
    team_size = players // 2
    keys = list(_WALKER_KEYS)
    for i in range(team_size - 1):
        keys += [f"teammate_{i}_{s}" for s in _OTHER_SUFFIXES]
    for i in range(team_size):
        keys += [f"opponent_{i}_{s}" for s in _OTHER_SUFFIXES]
    return tuple(keys + _ARENA_KEYS)


@functools.lru_cache(maxsize=None)
def _player_onehot(num_matches, players):
    # within-team player slot: home_i and away_i share an id, so home/away
    # symmetry -- and with it shared-policy self-play -- is preserved while
    # players can specialize into roles
    team_size = players // 2
    slot = np.tile(np.arange(team_size), 2)  # [0..ts-1, 0..ts-1]
    onehot = np.eye(team_size, dtype=np.float32)[slot]
    return np.broadcast_to(onehot[None], (num_matches, players, team_size))


@functools.lru_cache(maxsize=None)
def _slot_index(rows, players):
    # within-match slot of each returned row; env blocks are contiguous and
    # exactly `players` long, so the slot is just the row index modulo players
    return np.arange(rows, dtype=np.intp) % players


def sort_rows(env_id, num_matches, players):
    """Destination row of every returned per-player row, or None if sorted.

    envpool hands the match blocks back in completion order; `env_id[b]` is
    the match in block `b`. Row `b * players + slot` therefore belongs at
    `env_id[b] * players + slot`.
    """
    env_id = np.asarray(env_id)
    if env_id.shape[0] != num_matches:
        raise RuntimeError(
            f"envpool returned {env_id.shape[0]} matches, expected "
            f"{num_matches}: the adapter needs sync mode (batch_size == "
            f"num_envs); do not set batch_size/num_threads to make it async")
    if np.array_equal(env_id, np.arange(num_matches, dtype=env_id.dtype)):
        return None  # already env-major: the common single-thread case
    rows = num_matches * players
    return (np.repeat(env_id.astype(np.intp), players) * players
            + _slot_index(rows, players))


def sort_batch(obs, info, num_matches, players, *arrays):
    """Re-sort one envpool batch into env-major order.

    Returns (obs, *arrays) with per-player rows and per-match entries moved
    back to `env_id` order. An entry of `arrays` is treated as per-player
    when its length is `num_matches * players`, per-match otherwise.
    """
    env_id = np.asarray(info["env_id"])
    dest = sort_rows(env_id, num_matches, players)
    if dest is None:
        return (obs, *arrays)
    rows = num_matches * players
    env_dest = env_id.astype(np.intp)
    obs = {k: _scatter(v, dest) for k, v in obs.items()}
    out = [_scatter(a, dest if len(a) == rows else env_dest) for a in arrays]
    return (obs, *out)


def check_player_layout(info, num_matches, players):
    """Assert every match contributes `players` contiguous rows in env order."""
    players_env = np.asarray(info["players"]["env_id"])
    expected = np.repeat(np.asarray(info["env_id"]), players)
    assert players_env.shape[0] == num_matches * players and np.array_equal(
        players_env, expected), (
        "envpool returned an unexpected player layout: expected "
        f"{players} contiguous rows per match in env_id order, got "
        f"env_id={info['env_id']} players.env_id={players_env}")


def flatten_obs(obs, num_matches, players):
    """envpool dict obs -> (M, P, obs_dim) float32 policy input.

    Rows must already be env-major (see `sort_rows`); every value is
    (M * P, 1, d) -- the middle axis is envpool's stack dim -- and is
    reshaped to (M, P, d).

    The one feature layout for training and the eval tools: `obs_keys` in
    order plus the within-team one-hot slot, NaN/inf zeroed and values
    clipped to +/-1e3 so diverged physics cannot leak into the obs
    normalizer (the clip is a normalizer guard, not a termination).
    """
    parts = [obs[k].reshape(num_matches, players, -1) for k in obs_keys(players)]
    parts.append(_player_onehot(num_matches, players))
    flat = np.concatenate(parts, axis=-1).astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(flat, -1e3, 1e3, out=flat)
    return flat


class SoccerSelfPlay(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        env_name = kwargs.pop("env_name", NATIVE_ENV_ID)
        team_size = int(kwargs.pop("team_size", 2))
        assert team_size >= 1, f"team_size must be >= 1, got {team_size}"
        # Asymmetric goal reward: punishing concedes teaches ball-avoidance
        # ("cowardice") in self-play — the policy can avoid -goal_w by never
        # touching the ball. Reward scoring, don't punish conceding.
        self.goal_w_score = kwargs.pop("goal_w_score", 150.0)
        self.goal_w_concede = kwargs.pop("goal_w_concede", 0.0)
        # Keep dense terms SMALL relative to the goal: scoring terminates the
        # episode, so a large dense stream makes *not finishing* optimal
        # (dribble-farming). goal_w must exceed the discounted dense stream.
        self.vel_ball_w = kwargs.pop("vel_ball_w", 0.5)
        self.vel_player_w = kwargs.pop("vel_player_w", 0.25)
        self.time_w = kwargs.pop("time_w", 0.05)  # per-step cost: finish games
        # team-level chase reward: share the closest player's vel-to-ball with
        # the whole team (one chaser is enough; teammate learns to position)
        self.team_chase = kwargs.pop("team_chase", True)
        # dense-shaping anneal: linearly decay the vel terms to `dense_floor`
        # over `dense_anneal_steps` env steps, so the goal reward grows
        # relatively and the policy shifts from the proxy to actual scoring.
        self.dense_anneal_steps = kwargs.pop("dense_anneal_steps", 0)
        self.dense_floor = kwargs.pop("dense_floor", 0.15)
        self._anneal_step = 0
        seed = kwargs.pop("seed", 0)
        # episode cap: shorter than the env's own time_limit (45 s = 1800
        # steps at the 0.025 s control timestep) to recycle stale episodes
        max_steps = kwargs.pop("max_episode_steps", 600)
        # opponent curriculum: "self" = symmetric self-play (policy controls
        # all players); "random" = policy controls the home team only, away
        # acts randomly; "league" = away controlled by a mix of opponent types
        # (scripted + random + frozen past checkpoints) for robustness.
        self.opponent = kwargs.pop("opponent", "self")
        league_types = kwargs.pop("league_types", None)
        league_ckpt_dir = kwargs.pop("league_ckpt_dir", None)
        league_refresh = kwargs.pop("league_refresh", 500)
        # rl_games' num_actors is the TOTAL policy batch; each match holds
        # `controlled` of them.
        self.team_size = team_size
        self.players = 2 * team_size
        controlled = self.players if self.opponent == "self" else team_size
        assert num_actors % controlled == 0, (
            f"num_actors={num_actors} must be a multiple of "
            f"controlled players per match={controlled}")
        self.controlled = controlled
        self.num_matches = num_actors // controlled

        # everything still in kwargs is an envpool task option
        # (terminate_on_goal, time_limit, enable_field_box,
        # disable_walker_contacts, ...); max_num_players is NOT one of them --
        # envpool derives it as 2 * team_size and ignores any override.
        self.env = envpool.make_gymnasium(
            env_name, num_envs=self.num_matches, seed=seed,
            team_size=team_size, max_episode_steps=max_steps, **kwargs,
        )
        obs_space = self.env.observation_space
        self._keys = obs_keys(self.players)
        missing = [k for k in self._keys if k not in obs_space.spaces]
        assert not missing, (
            f"{env_name} does not expose {missing}: the adapter needs "
            f"envpool >= 1.2.7 native dm_control soccer (per-player obs, "
            f"teammate_i_*/opponent_i_* keys)")

        self.obs_dim = 0
        for k in self._keys:
            # native per-player space is (stack, d); there is no players axis
            self.obs_dim += int(np.prod(obs_space[k].shape))
        self.obs_dim += team_size  # one-hot slot, see flatten_obs
        self.act_dim = self.env.action_space.shape[-1]
        self.rows = self.num_matches * self.players

        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, (self.act_dim,), dtype=np.float32)

        # episode goal-diff tracking (home perspective) for reporting
        self._goal_diff = np.zeros(self.num_matches, dtype=np.float32)
        self._ret_goal_diff = np.zeros(self.num_matches, dtype=np.float32)

        self.league = None
        if self.opponent == "league":
            from rl_games.envs.dmc_soccer_opponents import OpponentLeague
            self.league = OpponentLeague(
                self.num_matches, types=league_types,
                ckpt_dir=league_ckpt_dir, refresh_every=league_refresh,
                act_dim=self.act_dim,
                rng=np.random.RandomState(seed + 12345))
        self._away_rng = np.random.RandomState(seed + 54321)
        self._last_obs_dict = None

    # --- native row ordering -------------------------------------------------

    def _sorted(self, obs, info, *arrays):
        return sort_batch(obs, info, self.num_matches, self.players, *arrays)

    # --- observations / rewards ---------------------------------------------

    def _flatten_obs(self, obs):
        flat = flatten_obs(obs, self.num_matches, self.players)
        self._flat_away = flat[:, self.controlled:]  # for league opponents
        # home players come first; only they are controlled outside "self"
        flat = flat[:, :self.controlled]
        return flat.reshape(self.num_matches * self.controlled, self.obs_dim)

    def _away_obs_dict(self, obs):
        ts = self.controlled
        return {
            k: obs[k].reshape(self.num_matches, self.players, -1)[:, ts:]
            for k in ("ball_ego_position", "team_goal_mid")
        }

    def _shaped_reward(self, obs, reward):
        # native envpool returns the per-player reward directly: 0 except on
        # a goal, where it is +1 for the scoring team and -1 for the other
        # (upstream soccer.cc: `rewards[player] = team == scoring_team ? 1 : -1`).
        per_player_reward = reward.reshape(self.num_matches, self.players)
        vel_ball = obs["stats_vel_ball_to_goal"].reshape(
            self.num_matches, self.players)
        # copied: the team-chase broadcast below writes into it, and the
        # reshape is a view of the env's obs dict
        vel_player = obs["stats_closest_vel_to_ball"].reshape(
            self.num_matches, self.players).copy()
        # one-sided ball progress: reward pushing the ball toward the opponent
        # goal, but DON'T punish when the opponent pushes it toward ours —
        # uncontrollable negatives teach avoidance.
        vel_ball = np.maximum(vel_ball, 0)
        if self.team_chase:
            # team-level chase: one player near the ball is enough. Broadcast
            # the closest player's vel-to-ball to the whole team so the other
            # player is free to position instead of also chasing.
            ts = self.team_size
            for lo, hi in ((0, ts), (ts, self.players)):
                team = vel_player[:, lo:hi]
                # closest player holds the only nonzero entry (others are 0);
                # upstream zeroes stats_closest_vel_to_ball for every player
                # that is not its team's nearest to the ball
                shared = team.sum(axis=1, keepdims=True)
                vel_player[:, lo:hi] = shared
        dense = 1.0
        if self.dense_anneal_steps > 0:
            dense = max(self.dense_floor,
                        1.0 - self._anneal_step / self.dense_anneal_steps)
        rew = (self.goal_w_score * np.maximum(per_player_reward, 0)
               - self.goal_w_concede * np.maximum(-per_player_reward, 0)
               + dense * self.vel_ball_w * vel_ball
               + dense * self.vel_player_w * vel_player
               - self.time_w)
        rew = rew[:, :self.controlled]
        batch = self.num_matches * self.controlled
        return rew.reshape(batch).astype(np.float32), per_player_reward

    # --- IVecEnv -------------------------------------------------------------

    def reset(self):
        obs, info = self.env.reset()
        # validate the block layout once: every env contributes exactly
        # `players` contiguous rows, in `info["env_id"]` order
        check_player_layout(info, self.num_matches, self.players)
        (obs,) = self._sorted(obs, info)
        self._goal_diff[:] = 0
        self._last_obs_dict = obs
        return self._flatten_obs(obs)

    def step(self, actions):
        # actions are written env-major: envpool reads them against
        # arange(num_envs), independent of the order the last batch arrived in
        acts = np.asarray(actions, dtype=np.float64).reshape(
            self.num_matches, self.controlled, self.act_dim)
        if self.controlled < self.players:
            if self.league is not None and self._last_obs_dict is not None:
                away = self.league.actions(
                    self._away_obs_dict(self._last_obs_dict),
                    self._flat_away)
            else:
                away = self._away_rng.uniform(
                    -1, 1,
                    (self.num_matches, self.players - self.controlled,
                     self.act_dim))
            acts = np.concatenate([acts, away], axis=1)
        obs, reward, terminated, truncated, info = self.env.step(
            acts.reshape(self.rows, self.act_dim))
        obs, reward, terminated, truncated = self._sorted(
            obs, info, reward, terminated, truncated)
        self._last_obs_dict = obs
        self._anneal_step += 1
        done = terminated | truncated  # (M,)

        flat_obs = self._flatten_obs(obs)
        rew, per_player_reward = self._shaped_reward(obs, reward)

        # Progress metric: in self-play the goal-diff averages ~0, so report
        # GOALS per episode — any side for "self", home-only for "random"
        # (where away goals are just noise). per_player_reward[:, 0] is a home
        # player: +1 when home scores, -1 when it concedes.
        if self.opponent == "self":
            self._goal_diff += np.abs(per_player_reward[:, 0])
        else:
            self._goal_diff += np.maximum(per_player_reward[:, 0], 0)
        self._ret_goal_diff[:] = self._goal_diff
        self._goal_diff *= 1 - done

        # rows are match-major, player-minor (row = match * controlled +
        # player), so repeating each match's done per player keeps the
        # trainer's per-row autoreset mask aligned with the obs rows.
        # Termination is per MATCH, not per player: a goal ends the match for
        # everyone, so every row of a match shares its done flag.
        done_p = np.repeat(done, self.controlled)
        info_out = {
            "time_outs": np.repeat(truncated, self.controlled),
            "scores": np.repeat(self._ret_goal_diff, self.controlled),
        }
        # envpool resets on the NEXT step: the obs returned with done=True is
        # the terminal obs; the following step() ignores its action and
        # returns the new episode's first obs with a zero reward. That row is
        # not a transition -- the trainer drops it via the next_step mask.
        return flat_obs, rew, done_p, info_out

    def get_number_of_agents(self):
        return 1  # independent actors, not rl_games' multi-agent path

    def get_env_info(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
            "autoreset_mode": "next_step",  # see step(): reset-step rows are masked
        }

    def get_env_state(self):
        # resume state: the dense anneal and the opponent RNGs live in this
        # process, not in the model -- without them a resumed run restarts
        # dense(t) at 1.0 (6.7x the floor on the shipped config)
        state = {
            "anneal_step": self._anneal_step,
            "away_rng": self._away_rng.get_state(),
        }
        if self.league is not None:
            state["league_rng"] = self.league.rng.get_state()
        return state

    def set_env_state(self, env_state):
        # older checkpoints carry env_state None or lack keys: keep defaults
        if not env_state:
            return
        self._anneal_step = int(env_state.get("anneal_step", self._anneal_step))
        away_rng = env_state.get("away_rng")
        if away_rng is not None:
            self._away_rng.set_state(away_rng)
        league_rng = env_state.get("league_rng")
        if league_rng is not None and self.league is not None:
            self.league.rng.set_state(league_rng)


def _scatter(array, dest):
    out = np.empty_like(array)
    out[dest] = array
    return out
