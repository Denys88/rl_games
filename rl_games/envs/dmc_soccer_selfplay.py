"""Symmetric self-play vecenv adapter: envpool BoxHead soccer -> rl_games.

One envpool env is a 2v2 match with 4 players. This adapter exposes every
player as an independent actor sharing ONE policy (symmetric self-play):
observations are egocentric and team-relative (team_goal_*, opponent_goal_*,
others_is_teammate), so the same policy plays both home and away. rl_games
sees num_actors = num_matches * players_per_match.

Reward shaping (the speed lever vs the sparse DeepMind setup):
    r = goal_w_score * max(players_reward, 0)         # scoring, concede unpunished
      + dense(t) * vel_ball_w   * max(vel_ball_to_goal, 0)   # one-sided ball progress
      + dense(t) * vel_player_w * team_chase                  # closest player, shared
      - time_w                                                # finish games
where dense(t) anneals 1 -> dense_floor over dense_anneal_steps env steps so
the goal term dominates once chase/dribble are bootstrapped. The dense terms
come straight from the env's stats observations. See docs/DMC_SOCCER_SELFPLAY.md
for the failure modes each piece of this prevents.
"""

import gymnasium
import numpy as np

from rl_games.common.ivecenv import IVecEnv

# per-player observation keys to feed the policy, in fixed order
_OBS_KEYS = [
    "joints_pos", "joints_vel", "body_height", "end_effectors_pos",
    "world_zaxis", "sensors_velocimeter", "sensors_gyro",
    "sensors_accelerometer", "prev_action",
    "ball_ego_position", "ball_ego_linear_velocity",
    "ball_ego_angular_velocity",
    "others_ego_position", "others_ego_linear_velocity",
    "others_ego_end_effectors_pos", "others_ego_orientation",
    "others_end_effectors_pos", "others_is_teammate",
    "team_goal_back_right", "team_goal_mid", "team_goal_front_left",
    "field_front_left", "opponent_goal_back_left", "opponent_goal_mid",
    "opponent_goal_front_right", "field_back_right",
]


class SoccerSelfPlay(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        env_name = kwargs.pop("env_name", "BoxheadSoccer2v2-v1")
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
        # episode cap: shorter than dm_control's 1800 to recycle stale episodes
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
        players = kwargs.pop("players_per_match", 4)
        controlled = players if self.opponent == "self" else players // 2
        assert num_actors % controlled == 0, (
            f"num_actors={num_actors} must be a multiple of "
            f"controlled players per match={controlled}")
        self.controlled = controlled
        self.num_matches = num_actors // controlled

        self.env = envpool.make_gymnasium(
            env_name, num_envs=self.num_matches, seed=seed,
            max_episode_steps=max_steps, **kwargs,
        )
        obs_space = self.env.observation_space
        # players per match from any per-player obs key
        self.players = obs_space["ball_ego_position"].shape[0]
        assert self.players == players, (
            f"env has {self.players} players, config says {players}")
        self.batch = self.num_matches * self.controlled

        self.obs_dim = 0
        for k in _OBS_KEYS:
            shape = obs_space[k].shape  # (players, ...)
            self.obs_dim += int(np.prod(shape[1:]))
        # one-hot within-team player slot (home_i and away_i share an id, so
        # home/away symmetry — and thus shared-policy self-play — is preserved
        # while letting players specialize into roles).
        team_size = self.players // 2
        slot = np.tile(np.arange(team_size), 2)  # [0..ts-1, 0..ts-1]
        self._player_onehot = np.eye(team_size, dtype=np.float32)[slot]
        self._player_onehot = np.broadcast_to(
            self._player_onehot[None],
            (self.num_matches, self.players, team_size))
        self.obs_dim += team_size
        act_shape = self.env.action_space.shape  # (players, act_dim)
        self.act_dim = act_shape[-1]

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
                rng=np.random.RandomState(seed + 12345))
        self._last_obs_dict = None

    def _flatten_all(self, obs):
        parts = [
            obs[k].reshape(self.num_matches, self.players, -1)
            for k in _OBS_KEYS
        ]
        parts.append(self._player_onehot)
        return np.concatenate(parts, axis=-1).astype(np.float32)  # (M, P, D)

    def _flatten_obs(self, obs):
        flat = self._flatten_all(obs)
        # guard against physics divergence leaking NaN/huge values into the
        # obs normalizer (the env also terminates such episodes itself)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(flat, -1e3, 1e3, out=flat)
        self._flat_away = flat[:, self.controlled:]  # for league opponents
        # home players come first; only they are controlled outside "self"
        flat = flat[:, :self.controlled]
        return flat.reshape(self.batch, self.obs_dim)

    def _away_obs_dict(self, obs):
        ts = self.controlled
        return {
            k: obs[k].reshape(self.num_matches, self.players, -1)[:, ts:]
            for k in ("ball_ego_position", "team_goal_mid")
        }

    def _shaped_reward(self, obs, info):
        players_reward = info["players_reward"].reshape(
            self.num_matches, self.players)
        vel_ball = obs["stats_vel_ball_to_goal"].reshape(
            self.num_matches, self.players)
        vel_player = obs["stats_closest_vel_to_ball"].reshape(
            self.num_matches, self.players)
        # one-sided ball progress: reward pushing the ball toward the opponent
        # goal, but DON'T punish when the opponent pushes it toward ours —
        # uncontrollable negatives teach avoidance.
        vel_ball = np.maximum(vel_ball, 0)
        if self.team_chase:
            # team-level chase: one player near the ball is enough. Broadcast
            # the closest player's vel-to-ball to the whole team so the other
            # player is free to position instead of also chasing.
            ts = self.players // 2
            for lo, hi in ((0, ts), (ts, self.players)):
                team = vel_player[:, lo:hi]
                # closest player holds the only nonzero entry (others are 0)
                shared = team.sum(axis=1, keepdims=True)
                vel_player[:, lo:hi] = shared
        dense = 1.0
        if self.dense_anneal_steps > 0:
            dense = max(self.dense_floor,
                        1.0 - self._anneal_step / self.dense_anneal_steps)
        rew = (self.goal_w_score * np.maximum(players_reward, 0)
               - self.goal_w_concede * np.maximum(-players_reward, 0)
               + dense * self.vel_ball_w * vel_ball
               + dense * self.vel_player_w * vel_player
               - self.time_w)
        rew = rew[:, :self.controlled]
        return rew.reshape(self.batch).astype(np.float32), players_reward

    def reset(self):
        obs, _ = self.env.reset()
        self._goal_diff[:] = 0
        self._last_obs_dict = obs
        return self._flatten_obs(obs)

    def step(self, actions):
        acts = np.asarray(actions, dtype=np.float64).reshape(
            self.num_matches, self.controlled, self.act_dim)
        if self.controlled < self.players:
            if self.league is not None and self._last_obs_dict is not None:
                away = self.league.actions(
                    self._away_obs_dict(self._last_obs_dict),
                    self._flat_away)
            else:
                away = np.random.uniform(
                    -1, 1,
                    (self.num_matches, self.players - self.controlled,
                     self.act_dim))
            acts = np.concatenate([acts, away], axis=1)
        obs, _, terminated, truncated, info = self.env.step(acts)
        self._last_obs_dict = obs
        self._anneal_step += 1
        done = terminated | truncated  # (M,)

        flat_obs = self._flatten_obs(obs)
        rew, players_reward = self._shaped_reward(obs, info)

        # Progress metric: in self-play the goal-diff averages ~0, so report
        # GOALS per episode — any side for "self", home-only for "random"
        # (where away goals are just noise).
        if self.opponent == "self":
            self._goal_diff += np.abs(players_reward[:, 0])
        else:
            self._goal_diff += np.maximum(players_reward[:, 0], 0)
        self._ret_goal_diff[:] = self._goal_diff
        self._goal_diff *= 1 - done

        done_p = np.repeat(done, self.controlled)
        info_out = {
            "time_outs": np.repeat(truncated, self.controlled),
            "scores": np.repeat(self._ret_goal_diff, self.controlled),
        }
        # envpool auto-resets; obs after done is the new episode's first obs
        return flat_obs, rew, done_p, info_out

    def get_number_of_agents(self):
        return 1  # independent actors, not rl_games' multi-agent path

    def get_env_info(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
        }
