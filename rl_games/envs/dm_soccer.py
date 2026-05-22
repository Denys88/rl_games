"""DeepMind Control Suite soccer (multi-agent locomotion) wrapper for rl_games.

Wraps `dm_control.locomotion.soccer` for use as an rl_games multi-agent
environment. The walker is selectable via ``walker_type`` — ``'boxhead'``
(simple wheeled robot, 3 actions), ``'ant'`` (8-DoF quadruped) or
``'humanoid'``. The LEARNER controls the home team (``team_size`` players);
the AWAY team is driven internally by a configurable opponent policy
(random, no-op, or a frozen NN whose weights are pushed in via
``update_weights`` for self-play).

Per-step the env returns batched arrays of shape ``(team_size, ...)`` for
obs/reward/done — the standard rl_games multi-agent contract (see
:mod:`rl_games.envs.multiwalker` for the same pattern).

The native dm_soccer reward is sparse (only ±1 on goals). Random ants get
zero learning signal, so we add an optional shaped reward composed from the
already-exposed ``stats_*`` observations (velocity-to-ball, ball-to-goal
velocity, forward velocity). Weights are configurable from YAML.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# --------------------------------------------------------------------------- #
# Lazy dm_control import — keeps `import rl_games` cheap for users who don't
# need this env. dm_control pulls in MuJoCo + a lot of XML parsing.
# --------------------------------------------------------------------------- #
def _load_soccer():
    from dm_control.locomotion import soccer  # noqa: WPS433
    return soccer


# Observation keys that are scalar shaping signals exposed by dm_soccer's
# ``Task`` instance — we strip these from the policy obs and instead use them
# to compute the shaped reward (otherwise the agent could just read the
# shaping reward directly off its own input).
_STAT_KEYS = (
    'stats_vel_to_ball',
    'stats_closest_vel_to_ball',
    'stats_veloc_forward',
    'stats_vel_ball_to_goal',
    'stats_home_avg_teammate_dist',
    'stats_teammate_spread_out',
    'stats_home_score',
    'stats_away_score',
)


def _flatten_obs(obs_dict: Dict[str, np.ndarray], skip_keys=()) -> np.ndarray:
    """Flatten dm_control's per-player OrderedDict obs into one 1-D vector.

    dm_control prepends a length-1 time axis to every observation by default
    (``aggregator=None``); we ravel through it so the policy sees a flat Box.
    """
    parts = []
    for k, v in obs_dict.items():
        if k in skip_keys:
            continue
        parts.append(np.asarray(v, dtype=np.float32).ravel())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class _NearGoalBallInitializer:
    """Spawn the ball in a configurable box (curriculum hook).

    Set `x_range=(x_lo, x_hi)` and `y_range=(y_lo, y_hi)` in field coords
    (+x = opp goal end in dm_soccer convention) to spawn the ball near
    a specific location at episode start AND after every goal respawn
    (MultiturnTask calls the same initializer).

    Use case: curriculum stage 1 spawns ball ~1m from opp goal so any
    chase-policy trips it in; later stages widen the range until at
    final stage it's identical to UniformInitializer (±0.6*field).
    Walkers still spawn uniformly across the full field so the policy
    sees varied approach geometries.
    """

    _INIT_BALL_Z = 0.5
    _SPAWN_RATIO = 0.6

    def __init__(self, x_range, y_range,
                 max_collision_avoidance_retries: int = 100):
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self._max_retries = int(max_collision_avoidance_retries)
        self._ball_geom_ids = None
        self._walker_geom_ids = None
        self._all_geom_ids = None

    def _set_ball(self, ball, physics, random_state):
        x = random_state.uniform(self.x_range[0], self.x_range[1])
        y = random_state.uniform(self.y_range[0], self.y_range[1])
        ball.set_pose(physics, [x, y, self._INIT_BALL_Z])
        ball.set_velocity(physics, velocity=0., angular_velocity=0.)

    def _set_walker(self, walker, walker_range, physics, random_state):
        walker.reinitialize_pose(physics, random_state)
        x, y = random_state.uniform(-walker_range, walker_range)
        (_, _, z), quat = walker.get_pose(physics)
        walker.set_pose(physics, [x, y, z], quat)
        rotation = random_state.uniform(-np.pi, np.pi)
        quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
        walker.shift_pose(physics, quaternion=quat)
        walker.set_velocity(physics, velocity=0., angular_velocity=0.)

    def _init_ids(self, task, physics):
        self._ball_geom_ids = {physics.bind(task.ball.geom)}
        self._walker_geom_ids = []
        for player in task.players:
            walker_geoms = player.walker.mjcf_model.find_all('geom')
            self._walker_geom_ids.append(
                set(physics.bind(walker_geoms).element_id))
        self._all_geom_ids = set(self._ball_geom_ids)
        for ws in self._walker_geom_ids:
            self._all_geom_ids |= ws

    def __call__(self, task, physics, random_state):
        # Mirror UniformInitializer's collision-retry loop, but use our
        # custom ball range and standard walker range.
        if not self._all_geom_ids:
            self._init_ids(task, physics)
        walker_range = np.asarray(task.arena.size) * self._SPAWN_RATIO
        for _ in range(self._max_retries):
            self._set_ball(task.ball, physics, random_state)
            for player in task.players:
                self._set_walker(player.walker, walker_range, physics,
                                  random_state)
            physics.forward()
            ok = True
            for contact in physics.data.contact:
                if contact.geom1 in self._all_geom_ids and \
                   contact.geom2 in self._all_geom_ids:
                    ok = False
                    break
            if ok:
                return
        # Give up — caller proceeds with whatever pose we ended on.


class DMSoccerAntEnv(gym.Env):
    """rl_games-compatible wrapper around dm_soccer ANT 2v2 (or NvN).

    Args:
        walker_type: Which dm_soccer creature to use. One of ``'boxhead'``
            (simple 3-action wheeled robot — easiest to train), ``'ant'``
            (8-DoF quadruped) or ``'humanoid'``. Obs/action dims are
            auto-detected from the dm_control specs, so the rest of the
            wrapper adapts automatically.
        team_size: Players per team. ``2`` is the canonical setup; the env
            handles 1v1 / 3v3 / 4v4 the same way.
        time_limit: Match length in seconds. dm_soccer terminates on time-out
            (truncated) or first goal if ``terminate_on_goal=True``.
        enable_field_box: Add walls so the ball can't roll out of the field.
            On for training (denser experience), off for cinematic eval.
        terminate_on_goal: End the episode the instant a goal is scored.
        shaped_reward: Add dense shaping rewards on top of the sparse goal
            reward. Without this, random ants almost never see signal.
        shaping_weights: Dict overriding the default per-component weights.
        opponent_kind: How the AWAY team acts. One of:
            - ``'random'``: uniform random actions (good warmup baseline)
            - ``'noop'``: zero actions (stationary opponents, easy curriculum)
            - ``'policy'``: NN policy whose state_dict is set via
              :meth:`update_weights`. Until weights arrive, falls back to
              random. Used by :class:`SelfPlayManager`.
        opponent_factory: Callable ``() -> object`` returning a torch policy
            that exposes ``act(obs_np) -> actions_np``. Required when
            ``opponent_kind='policy'``. Construction is deferred to the Ray
            worker process.
        seed: Per-env RNG seed; each Ray worker gets ``seed + worker_idx``.
        central_value: If True, ``reset/step`` return a dict with both per-
            agent obs and a concatenated ``state`` for asymmetric (centralized
            critic) training. Matches MultiWalker's pattern.
    """

    DEFAULT_SHAPING_WEIGHTS: Dict[str, float] = {
        # The two we care about most: chase the ball, and push it toward the
        # opponent's goal. Anything you can score with subsumes these.
        'vel_to_ball': 0.05,
        'vel_ball_to_goal': 1.0,
        # Small forward-velocity bonus to break the ants out of the
        # "lie still" local optimum at the start of training. Decays naturally
        # once the ball-chasing reward dominates.
        'veloc_forward': 0.01,
        # Native sparse goal reward (already in dm_soccer reward); we just
        # scale it. ±1 per goal scored / conceded → ±10 by default.
        'goal': 10.0,
        # Per-step bonus for ball being closer to opponent goal than to own
        # goal — gives a positional progress signal even when no one is
        # actively pushing the ball. Off by default (set >0 to enable).
        # Computed as `tanh((d_own - d_opp) / 5.0)` so it's bounded in [-1,1].
        'ball_field_progress': 0.0,
        # Linear bonus when the ball gets within `ball_near_goal_radius`
        # metres of the opponent goal. Scaled by (radius - distance) so the
        # signal grows as the ball approaches the goal — this is the only
        # shaping with a strict monotone gradient toward actual scoring.
        # Default off; enabled per-config (e.g. v3).
        'ball_near_goal_bonus': 0.0,
        # Per-agent bonus when the agent is positioned *behind* the ball
        # relative to the opponent goal (i.e. on the own-goal side).
        # Computed as `max(0, cos(angle(ant→ball, ball→opp_goal)))` so it's
        # in [0, 1] per step. This rewards "approach ball from the correct
        # side so a forward push moves it toward goal" — the missing piece
        # the chase reflex doesn't teach.
        'behind_ball_bonus': 0.0,
        # Per-step penalty proportional to ball→opp_goal distance:
        #   r += -w * d(ball, opp_goal)
        # Always-on dense gradient: small negative when ball is far from the
        # goal, ~zero when it's at the goal mouth, big positive on actual
        # score (via `goal`). Unlike `ball_field_progress` this never goes
        # positive on its own (it's a penalty), and unlike `ball_near_goal_bonus`
        # it's not truncated to a small radius — the signal exists everywhere
        # on the field. This is the cleanest "make ball be near goal" reward.
        'ball_dist_to_goal_penalty': 0.0,
    }
    DEFAULT_BALL_NEAR_GOAL_RADIUS: float = 3.0

    def __init__(
        self,
        name: str = 'dm_soccer',
        walker_type: str = 'ant',
        team_size: int = 2,
        time_limit: float = 45.0,
        enable_field_box: bool = True,
        terminate_on_goal: bool = True,
        disable_walker_contacts: bool = False,
        shaped_reward: bool = True,
        shaping_weights: Optional[Dict[str, float]] = None,
        opponent_kind: str = 'random',
        opponent_factory: Optional[Callable[[], Any]] = None,
        opponent_factory_config: Optional[Dict[str, Any]] = None,
        central_value: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.name = name
        self.walker_type = str(walker_type).lower()
        self.team_size = int(team_size)
        self.time_limit = float(time_limit)
        self.enable_field_box = bool(enable_field_box)
        self.terminate_on_goal = bool(terminate_on_goal)
        self.disable_walker_contacts = bool(disable_walker_contacts)
        self.shaped_reward = bool(shaped_reward)
        self.shaping_weights = dict(self.DEFAULT_SHAPING_WEIGHTS)
        if shaping_weights:
            self.shaping_weights.update(shaping_weights)
        self.opponent_kind = opponent_kind
        self.opponent_factory = opponent_factory
        self.opponent_factory_config = opponent_factory_config
        # If YAML supplied a config dict, build the factory in-process.
        # We do it lazily inside update_weights() to avoid importing torch on
        # workers that never receive a weight push (e.g. random-opp baseline).
        self.use_central_value = bool(central_value)
        self._seed = seed

        # Build the underlying dm_soccer env. The default field is 32-48m ×
        # 24-36m which is enormous for a ~0.5m ant — when `field_size` is
        # provided we bypass `soccer.load` and assemble the env manually so
        # we can shrink the pitch for easier exploration.
        soccer = _load_soccer()
        _walker_types = {
            'boxhead': soccer.WalkerType.BOXHEAD,
            'ant': soccer.WalkerType.ANT,
            'humanoid': soccer.WalkerType.HUMANOID,
        }
        if self.walker_type not in _walker_types:
            raise ValueError(
                f"walker_type must be one of {sorted(_walker_types)}, "
                f"got {self.walker_type!r}"
            )
        dm_walker_type = _walker_types[self.walker_type]
        self._random_state = np.random.RandomState(seed)
        self._field_size = kwargs.pop('field_size', None)
        if self._field_size is None:
            self._dm_env = soccer.load(
                team_size=self.team_size,
                time_limit=self.time_limit,
                random_state=self._random_state,
                disable_walker_contacts=self.disable_walker_contacts,
                enable_field_box=self.enable_field_box,
                terminate_on_goal=self.terminate_on_goal,
                walker_type=dm_walker_type,
            )
        else:
            from dm_control import composer
            from dm_control.locomotion.soccer import (
                MultiturnTask, RandomizedPitch, SoccerBall, Task,
                _make_players,
            )
            w, h = self._field_size
            # Optional override for goal opening. Pitch.py default formula is
            # (depth, height*0.33, depth) on a (depth=_SIDE_WIDTH/2 ≈ 2.66m)
            # base. For a 15×10 field that's ~3.3m goal width — only ~6 ant
            # body-lengths. Passing `goal_size=(depth, width, height)` here
            # lets v5+ configs widen the goal so PPO sees scoring events
            # often enough to actually learn.
            self._goal_size = kwargs.pop('goal_size', None)
            arena = RandomizedPitch(
                min_size=(float(w), float(h)),
                max_size=(float(w), float(h)),
                keep_aspect_ratio=False,
                field_box=self.enable_field_box,
                goal_size=tuple(self._goal_size) if self._goal_size else None,
            )
            # Curriculum hook: when `ball_spawn_x_range`/`ball_spawn_y_range`
            # are provided, spawn the ball within that box instead of the
            # default ±0.6×field. Set both ranges close to the opp-goal end
            # (positive x = away-team goal end in dm_soccer convention) to
            # bootstrap scoring on early curriculum stages. Walkers still
            # spawn uniformly across the full field.
            self._ball_spawn_x_range = kwargs.pop('ball_spawn_x_range', None)
            self._ball_spawn_y_range = kwargs.pop('ball_spawn_y_range', None)
            initializer = None
            if self._ball_spawn_x_range is not None:
                initializer = _NearGoalBallInitializer(
                    x_range=tuple(self._ball_spawn_x_range),
                    y_range=tuple(self._ball_spawn_y_range)
                            if self._ball_spawn_y_range
                            else (-0.6 * h / 2, 0.6 * h / 2),
                )
            task_cls = Task if self.terminate_on_goal else MultiturnTask
            task = task_cls(
                players=_make_players(self.team_size, dm_walker_type),
                arena=arena,
                ball=SoccerBall(),
                disable_walker_contacts=self.disable_walker_contacts,
                initializer=initializer,
            )
            self._dm_env = composer.Environment(
                task=task,
                time_limit=self.time_limit,
                random_state=self._random_state,
            )

        # Per-player specs are identical for ANT — read once, expose as gym Box.
        action_specs = self._dm_env.action_spec()
        obs_specs = self._dm_env.observation_spec()
        assert len(action_specs) == 2 * self.team_size, (
            f'expected {2 * self.team_size} players, got {len(action_specs)}'
        )

        # Action space: each ant has 8 continuous actuators in [-1, 1].
        a_spec = action_specs[0]
        self.action_space = gym.spaces.Box(
            low=a_spec.minimum.astype(np.float32),
            high=a_spec.maximum.astype(np.float32),
            shape=a_spec.shape,
            dtype=np.float32,
        )

        # Compute the policy obs dim (we strip stats_* keys so the agent can't
        # read its own shaping reward directly).
        sample_obs = self._dm_env.reset().observation[0]
        self._policy_obs_keys = [k for k in sample_obs.keys() if k not in _STAT_KEYS]
        flat = _flatten_obs(sample_obs, skip_keys=_STAT_KEYS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32
        )
        if self.use_central_value:
            # Centralized critic state = concat of all home-team flat obs.
            self.state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(flat.shape[0] * self.team_size,),
                dtype=np.float32,
            )

        # Opponent policy. For 'policy' kind we instantiate lazily on the
        # first weight push so we don't need torch at __init__ time.
        self._opponent_policy = None  # set on first update_weights() call

        # Match-level bookkeeping (used for info logging).
        self._steps = 0
        self._home_goals = 0
        self._away_goals = 0
        self._last_home_score = 0.0
        self._last_away_score = 0.0
        self.concat_infos = True

    # -- rl_games multi-agent hooks ----------------------------------------- #

    def get_number_of_agents(self) -> int:
        return self.team_size

    def has_action_mask(self) -> bool:
        return False

    def update_weights(self, weights: Dict[str, Any]) -> None:
        """Receive a learner state_dict and load it into the frozen opponent.

        Called by :class:`rl_games.algos_torch.self_play_manager.SelfPlayManager`
        when the learner crosses the configured update threshold. Weights
        arrive on CPU (Ray serializes from there). On the first call we
        construct the opponent network via ``opponent_factory``.
        """
        if self.opponent_kind not in ('policy', 'fixed'):
            return
        if self._opponent_policy is None:
            factory = self.opponent_factory
            if factory is None and self.opponent_factory_config is not None:
                # Lazy import keeps random/noop workers torch-free.
                from rl_games.envs.dm_soccer_opponent import make_opponent_factory
                # Default the obs/action shape from this env if YAML omitted them.
                cfg = dict(self.opponent_factory_config)
                cfg.setdefault('obs_shape', self.observation_space.shape)
                cfg.setdefault('actions_num', int(self.action_space.shape[0]))
                factory = make_opponent_factory(**cfg)
            if factory is None:
                raise RuntimeError(
                    "opponent_kind='policy' but neither opponent_factory nor "
                    "opponent_factory_config was provided"
                )
            self._opponent_policy = factory()
        # 'fixed' opponents ignore further pushes after the first load — they
        # stay locked to whatever was loaded at env init (typically a strong
        # baseline checkpoint that the home team must consistently outscore).
        if self.opponent_kind == 'fixed' and self._opponent_policy.is_ready:
            return
        # Delegate to the policy wrapper — it owns the load_state_dict logic
        # and any prefix stripping (e.g. torch.compile's '_orig_mod.' prefix,
        # see project memory).
        self._opponent_policy.load_weights(weights)

    # -- gym API ------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._random_state = np.random.RandomState(seed)
        ts = self._dm_env.reset()
        self._steps = 0
        self._home_goals = 0
        self._away_goals = 0
        self._last_home_score = 0.0
        self._last_away_score = 0.0
        # Self-play: re-draw opponent from the pool at every match start so a
        # single env worker exposes the learner to a mix of historical
        # checkpoints over time. Skipped silently if pool is empty.
        if self._opponent_policy is not None and hasattr(self._opponent_policy, 'resample'):
            self._opponent_policy.resample(self._random_state)
        # Stash placeholder for opponent-team obs (used in step before the
        # first dm_env.step fires).
        self._last_away_obs = ts.observation[self.team_size :]
        obs = self._build_home_obs(ts.observation)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        # ``action`` arrives as (team_size, action_dim) per rl_games convention.
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(self.team_size, -1)

        # Build the full per-player action list: home actions from learner,
        # away actions from the opponent policy.
        away_actions = self._opponent_actions(self._dm_env.observation_spec())
        all_actions: List[np.ndarray] = []
        for i in range(self.team_size):
            all_actions.append(np.clip(action[i], -1.0, 1.0))
        for i in range(self.team_size):
            all_actions.append(away_actions[i])

        ts = self._dm_env.step(all_actions)
        self._steps += 1

        # Native rewards are per-player; for the home team they're identical
        # (team-shared scoring reward). We keep them per-agent for shaping.
        home_obs_full = ts.observation[: self.team_size]
        away_obs_full = ts.observation[self.team_size :]
        # Stash for the next opponent step (saves a re-query).
        self._last_away_obs = away_obs_full

        # Goal accounting (for info logging) from stats_* fields.
        home_score = float(np.asarray(home_obs_full[0]['stats_home_score']).ravel()[0])
        away_score = float(np.asarray(home_obs_full[0]['stats_away_score']).ravel()[0])
        if home_score > self._last_home_score:
            self._home_goals += int(round(home_score - self._last_home_score))
        if away_score > self._last_away_score:
            self._away_goals += int(round(away_score - self._last_away_score))
        self._last_home_score = home_score
        self._last_away_score = away_score

        rewards = self._compute_home_rewards(ts, home_obs_full)

        # dm_env's step_type LAST = terminal (goal) or truncated (timeout).
        # discount==0 means a hard terminal; discount==1 with LAST is a
        # time-limit truncation. rl_games wants the split.
        is_last = bool(ts.last())
        terminated = bool(is_last and (ts.discount == 0.0))
        truncated = bool(is_last and not terminated)
        # Broadcast to per-agent dones (rl_games concatenates).
        terms = np.full((self.team_size,), terminated, dtype=bool)
        truncs = np.full((self.team_size,), truncated, dtype=bool)

        obs = self._build_home_obs(ts.observation)
        info: Dict[str, Any] = {
            'time_outs': truncs,
        }
        if is_last:
            # Episode-summary stats — picked up by rl_games' info aggregator.
            info['home_goals'] = self._home_goals
            info['away_goals'] = self._away_goals
            info['goal_diff'] = self._home_goals - self._away_goals
            info['ep_len'] = self._steps
            # 'scores' is the conventional key DefaultAlgoObserver scans
            # for episode-level scalars. Surface goal-diff there so it
            # lands on TB as `scores/mean` next to the shaped reward.
            info['scores'] = float(self._home_goals - self._away_goals)
        return obs, rewards.astype(np.float32), terms, truncs, info

    def render(self, mode: str = 'rgb_array', height: int = 480, width: int = 640,
               camera_id: int = 3):
        # dm_soccer exposes 20 cameras per ant arena. camera 0 ('top_down') is
        # broken in dm_control 1.0.41 — renders just skybox blue. camera 3
        # ('soccer_ball/ball_cam_far') is the canonical wide pitch view: full
        # field, both goals, ball, and all four ants in frame.
        # Pass camera_id=N to override (e.g. egocentric 7/11/15/19).
        return self._dm_env.physics.render(
            camera_id=camera_id, height=height, width=width)

    def seed(self, seed: int) -> None:
        self._seed = seed
        self._random_state = np.random.RandomState(seed)

    def close(self) -> None:
        self._dm_env.close()

    # -- internals ---------------------------------------------------------- #

    def _build_home_obs(self, all_player_obs):
        """Stack home-team flat obs (and optionally a centralized state)."""
        home = all_player_obs[: self.team_size]
        flat = np.stack(
            [_flatten_obs(o, skip_keys=_STAT_KEYS) for o in home]
        ).astype(np.float32)
        if self.use_central_value:
            return {
                'obs': flat,
                'state': flat.reshape(-1),  # concat of all teammate flat obs
            }
        return flat

    def _opponent_actions(self, action_specs) -> List[np.ndarray]:
        """Compute the away team's actions for the upcoming step."""
        if self.opponent_kind == 'noop':
            return [np.zeros(self.action_space.shape, dtype=np.float64)
                    for _ in range(self.team_size)]
        if self.opponent_kind == 'policy' and self._opponent_policy is not None:
            # Build flat away-team obs and ask the policy. Mirroring around
            # the field is handled by dm_soccer itself — each player already
            # observes ego-centric quantities, so a policy trained for "home"
            # works out of the box for "away".
            away_flat = np.stack([
                _flatten_obs(o, skip_keys=_STAT_KEYS)
                for o in self._last_away_obs
            ]).astype(np.float32)
            return list(self._opponent_policy.act(away_flat))
        # Fallback — random uniform in [-1, 1].
        return [self._random_state.uniform(-1.0, 1.0,
                                           size=self.action_space.shape)
                for _ in range(self.team_size)]

    def _compute_home_rewards(self, ts, home_obs_full) -> np.ndarray:
        """Combine sparse goal reward with optional dense shaping."""
        # Per-player goal reward from dm_soccer (1.0 home goal, -1.0 conceded).
        if ts.reward is None:
            base = np.zeros(self.team_size, dtype=np.float32)
        else:
            base = np.asarray(ts.reward[: self.team_size], dtype=np.float32)
        rewards = self.shaping_weights['goal'] * base

        if not self.shaped_reward:
            return rewards

        # Shaping signals are per-player scalars exposed in obs as stats_*.
        # ball_field_progress and ball_near_goal_bonus are computed once per
        # step from team-shared geometry (player 0's ego frame is fine —
        # distances are frame-invariant).
        bp_w = self.shaping_weights.get('ball_field_progress', 0.0)
        ng_w = self.shaping_weights.get('ball_near_goal_bonus', 0.0)
        dg_w = self.shaping_weights.get('ball_dist_to_goal_penalty', 0.0)
        ng_radius = float(self.shaping_weights.get(
            'ball_near_goal_radius', self.DEFAULT_BALL_NEAR_GOAL_RADIUS))
        ball_progress = 0.0
        near_goal_bonus = 0.0
        dist_to_goal_penalty = 0.0

        if bp_w > 0.0 or ng_w > 0.0 or dg_w > 0.0:
            obs0 = home_obs_full[0]
            ball_pos = np.asarray(obs0['ball_ego_position']).ravel()
            opp_goal = np.asarray(obs0['opponent_goal_mid']).ravel()
            d_opp = float(np.linalg.norm(ball_pos[:2] - opp_goal[:2]))
            if ng_w > 0.0:
                # Linear ramp inside the radius — gives a monotone gradient
                # the closer the ball is to the goal centre. Outside, no
                # contribution (no penalty for ball being elsewhere).
                near_goal_bonus = max(0.0, ng_radius - d_opp)
            if bp_w > 0.0:
                own_goal = np.asarray(obs0['team_goal_mid']).ravel()
                d_own = float(np.linalg.norm(ball_pos[:2] - own_goal[:2]))
                ball_progress = float(np.tanh((d_own - d_opp) / 5.0))
            if dg_w > 0.0:
                # Always-on negative shaping: r += -w * d_opp.
                # Equivalent to "small negative everywhere except near goal".
                dist_to_goal_penalty = -d_opp

        bb_w = self.shaping_weights.get('behind_ball_bonus', 0.0)

        for i, obs in enumerate(home_obs_full):
            # Team-shared ball-chase signal: every player is rewarded by the
            # *closest* teammate's velocity-to-ball, not its own. This is the
            # rate-of-change of min(dist over teammates) — only one robot needs
            # to chase the ball; the other keeps the same reward while it
            # positions, instead of both swarming the ball.
            v_to_ball = float(np.asarray(obs['stats_closest_vel_to_ball']).ravel()[0])
            v_ball_goal = float(np.asarray(obs['stats_vel_ball_to_goal']).ravel()[0])
            v_fwd = float(np.asarray(obs['stats_veloc_forward']).ravel()[0])
            behind = 0.0
            if bb_w > 0.0:
                # ant→ball direction in ant ego frame is just `ball_ego_position`.
                # ball→opp_goal direction in same frame: opp_goal - ball.
                ball = np.asarray(obs['ball_ego_position']).ravel()[:2]
                goal = np.asarray(obs['opponent_goal_mid']).ravel()[:2]
                ant_to_ball = ball  # ant is at origin in its own ego frame
                ball_to_goal = goal - ball
                a_norm = float(np.linalg.norm(ant_to_ball))
                b_norm = float(np.linalg.norm(ball_to_goal))
                if a_norm > 1e-3 and b_norm > 1e-3:
                    cos = float(np.dot(ant_to_ball, ball_to_goal) / (a_norm * b_norm))
                    behind = max(0.0, cos)  # only positive contribution
            rewards[i] += (
                self.shaping_weights['vel_to_ball'] * v_to_ball
                + self.shaping_weights['vel_ball_to_goal'] * v_ball_goal
                + self.shaping_weights['veloc_forward'] * v_fwd
                + bp_w * ball_progress
                + ng_w * near_goal_bonus
                + bb_w * behind
                + dg_w * dist_to_goal_penalty
            )
        return rewards


# --------------------------------------------------------------------------- #
# Factory used by env_configurations.py
# --------------------------------------------------------------------------- #
def create_dm_soccer(**kwargs) -> DMSoccerAntEnv:
    return DMSoccerAntEnv(**kwargs)
