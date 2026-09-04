"""EnvPool DeepMind soccer (``DmcSoccer{Boxhead,Ant,Humanoid}-v1``) for rl_games.

envpool >= 1.2.7 (main after 2026-08-30) ships a native C++ port of
``dm_control.locomotion.soccer``. One envpool instance simulates
``num_actors`` matches; every match has ``2 * team_size`` players laid out
home team first, then away team (see ``envpool/mujoco/locomotion/soccer.cc``).

This wrapper exposes the HOME team to the learner with parameter sharing:
the ``team_size`` home players of every match are flattened into one batch
of ``num_actors * team_size`` streams, env-major / agent-minor (the same
layout ``VmasVecEnv`` uses, so ``all_done_indices[::num_agents]`` picks one
entry per match). The AWAY team is driven inside the wrapper by

  * ``opponent: random`` — uniform random actions, or
  * ``opponent: pool``   — frozen torch copies of the learner (self-play /
    league). Each match is assigned a pool member id; ``RANDOM_ID`` (-1) means
    random actions, ``MAIN_ID`` (0) the latest learner weights. Assignments
    are applied lazily when a match resets so an opponent never changes
    mid-match. The driver is ``SoccerLeagueObserver``
    (rl_games/common/soccer_observer.py). All members' state dicts are
    stacked along a leading axis and evaluated in ONE ``torch.func.vmap`` call
    per step (matches grouped by member, padded to the largest group) instead
    of one forward per member. With ``opponent_deterministic: False`` the away
    team samples ``mu + sigma * scale * N(0, 1)`` where ``scale`` is drawn per
    match from ``opponent_sigma_scale: [lo, hi]`` so pool members also differ
    in how noisy they play.

``player_id_obs: True`` appends a one-hot of the robot's index within its team
to every observation row (home and away) so the parameter-shared policy can
break the symmetry between teammates (striker / defender roles).

``population: N`` (population league) makes EVERY robot of every match a
learner row (``get_number_of_agents()`` = 2 * team_size, rows env-major
``[home0, home1, away0, away1]``). Each match carries a (home_slot, away_slot)
pair set via ``set_pair_assignment`` (applied at the match's next reset) and
every row gets a slot one-hot as its LAST N obs dims; the
``population_actor_critic`` network routes rows to their slot's parameters.
Both teams receive the shaped reward from their own players' stats (envpool
computes ``stats_*`` relative to each player's own goal). No opponent
inference runs in this mode; infos carry ``home_slot`` / ``away_slot``.

Rewards: the native ±1 goal reward (scaled by ``goal``) plus the lean
shaping that unlocked scoring in the dm_control experiments — closest
teammate's velocity to ball (team shared), ball velocity to goal, forward
velocity. Shaping signals are read from the ``stats_*`` observation keys,
which are stripped from the policy observation.

Observations are returned as torch tensors on ``device`` (the away-team
inference runs there too), so rl_games treats this as a tensor env.
"""

import copy

import gymnasium as gym
import numpy as np
import torch
from torch.func import functional_call, vmap

from rl_games.common.ivecenv import IVecEnv

RANDOM_ID = -1   # away team plays uniform random actions
MAIN_ID = 0      # away team = latest learner weights (pure self-play)

STAT_KEYS = (
    'stats_vel_to_ball',
    'stats_closest_vel_to_ball',
    'stats_veloc_forward',
    'stats_vel_ball_to_goal',
    'stats_home_avg_teammate_dist',
    'stats_teammate_spread_out',
    'stats_home_score',
    'stats_away_score',
)

DEFAULT_SHAPING = {
    'vel_to_ball': 0.2,        # closest teammate's velocity to ball (shared)
    'vel_ball_to_goal': 1.0,   # ball velocity toward the opponent goal
    'veloc_forward': 0.0,      # own forward velocity
    'goal': 100.0,             # scales the native +-1 goal reward
    # per-step team bonus while stats_teammate_spread_out (teammates > 5 m apart)
    'spread_out': 0.0,
    # per-step team penalty once the whole home team AND the ball have been
    # stationary for stuck_steps control steps (parked-at-the-post stall)
    'stuck_penalty': 0.0,
}

_WALKER_ENV_IDS = {
    'boxhead': 'DmcSoccerBoxhead-v1',
    'ant': 'DmcSoccerAnt-v1',
    'humanoid': 'DmcSoccerHumanoid-v1',
}

_ENV_PASSTHROUGH = (
    'time_limit', 'max_episode_steps', 'enable_field_box', 'terminate_on_goal',
    'disable_walker_contacts', 'keep_aspect_ratio', 'pitch_size_min',
    'pitch_size_max', 'goal_size', 'walker_bounce', 'ball_bounce', 'num_threads',
    'thread_affinity_offset',
)


def _strip_prefix(state_dict, prefix='_orig_mod.'):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()}


def _unscript_norms(model):
    """Replace jit-scripted RunningMeanStd children (rl_games BaseModelNetwork)
    with plain modules: TorchScript modules cannot run under torch.func.vmap."""
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    for name, shape in (('running_mean_std', getattr(model, 'obs_shape', None)),
                        ('value_mean_std', (getattr(model, 'value_size', 1),))):
        sm = getattr(model, name, None)
        if isinstance(sm, torch.jit.ScriptModule):
            if isinstance(shape, dict):
                raise NotImplementedError('dict observations are not supported by the batched pool')
            plain = RunningMeanStd(shape).to(next(iter(sm.state_dict().values())).device)
            plain.load_state_dict(sm.state_dict())
            setattr(model, name, plain.eval())
    return model


def state_dict_to_cpu(state_dict):
    """Detached CPU copy of a state dict (pool members live on the host)."""
    return {k: v.detach().to('cpu', copy=True) for k, v in _strip_prefix(state_dict).items()}


class EnvpoolSoccerVecEnv(IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        import envpool

        self.num_envs = int(num_actors)
        walker = str(kwargs.pop('walker_type', 'boxhead')).lower()
        env_id = kwargs.pop('env_id', _WALKER_ENV_IDS.get(walker, walker))
        self.team_size = int(kwargs.pop('team_size', 2))
        self.device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.opponent = kwargs.pop('opponent', 'random')
        if self.opponent not in ('random', 'pool'):
            raise ValueError(f"opponent must be 'random' or 'pool', got {self.opponent!r}")
        self.opponent_deterministic = bool(kwargs.pop('opponent_deterministic', True))
        # per-match multiplier on the away policy's sigma (stochastic opponents)
        lo, hi = kwargs.pop('opponent_sigma_scale', (1.0, 1.0))
        self.opponent_sigma_range = (float(lo), float(hi))
        self.player_id_obs = bool(kwargs.pop('player_id_obs', False))
        # population mode: all players are learner rows; each match pairs two
        # slots (home, away); a slot one-hot (last N dims) tags every row
        self.population = int(kwargs.pop('population', 0))
        self.shaping = dict(DEFAULT_SHAPING)
        self.shaping.update(kwargs.pop('shaping_weights', None) or {})
        # multiplier on every shaping term except 'goal' (annealed by the observer)
        self.shaping_scale = 1.0
        seed = int(kwargs.pop('seed', 0))
        self._rng = np.random.RandomState(seed + 1)
        self.stuck_steps = int(kwargs.pop('stuck_steps', 40))      # 1 s at 40 Hz
        self.stuck_speed = float(kwargs.pop('stuck_speed', 0.1))   # m/s
        # Liu et al. 2019: vel-to-ball "thresholded at zero" (approaching only)
        self.clip_vel_to_ball = bool(kwargs.pop('clip_vel_to_ball', False))

        env_kwargs = {'team_size': self.team_size, 'seed': seed}
        for k in _ENV_PASSTHROUGH:
            if k in kwargs:
                env_kwargs[k] = kwargs.pop(k)
        env_kwargs.update(kwargs)
        self.env = envpool.make_gymnasium(
            env_id, num_envs=self.num_envs, batch_size=self.num_envs, **env_kwargs)

        self.players = 2 * self.team_size
        self.total_players = self.num_envs * self.players
        self.num_home = self.num_envs * self.team_size

        obs_spaces = self.env.observation_space.spaces
        missing = [k for k in ('stats_closest_vel_to_ball', 'stats_vel_ball_to_goal',
                               'stats_veloc_forward') if k not in obs_spaces]
        if missing:
            raise RuntimeError(f'soccer env lacks shaping stats {missing}')
        self.obs_keys = [k for k in obs_spaces if k not in STAT_KEYS]
        self.base_obs_dim = int(sum(np.prod(obs_spaces[k].shape) for k in self.obs_keys))
        # one-hot index within the team, same layout for home and away rows
        self._pid_onehot = (np.eye(self.team_size, dtype=np.float32)[np.arange(self.players) % self.team_size]
                            if self.player_id_obs else None)
        self.obs_dim = (self.base_obs_dim + (self.team_size if self.player_id_obs else 0)
                        + self.population)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        act_dim = int(self.env.action_space.shape[-1])
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.act_dim = act_dim

        # per-match bookkeeping
        self._goals_home = np.zeros(self.num_envs, dtype=np.int64)
        self._goals_away = np.zeros(self.num_envs, dtype=np.int64)
        self._match_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._still_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._stuck_steps_total = np.zeros(self.num_envs, dtype=np.int64)
        self._match_opp = np.full(self.num_envs, RANDOM_ID, dtype=np.int64)
        self._opp_sigma_scale = np.ones(self.num_envs, dtype=np.float32)
        self._pairs = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._pending_pairs = None
        self._pending_assignment = None
        self._need_reset = np.zeros(self.num_envs, dtype=bool)
        self._perm = None        # player-row permutation to env-major order
        self._identity = np.arange(self.total_players)
        self._away_obs = None    # torch (num_envs, team_size, obs_dim)

        # opponent pool: one stacked state dict (leading axis = slot), vmapped
        self._base = None          # frozen, un-scripted copy of the learner's model
        self._vforward = None
        self._stacked = None       # {name: tensor(M, ...)} on device
        self._slot_of = {}         # member_id -> slot
        self._groups = None        # cached (idx, mask) grouping of matches by slot

    # ---------------------------------------------------------------- pool API

    def set_opponent_template(self, model):
        """Register the learner's model architecture (deep-copied, frozen)."""
        model = getattr(model, '_orig_mod', model)
        base = _unscript_norms(copy.deepcopy(model).to(self.device).eval())
        for p in base.parameters():
            p.requires_grad_(False)
        self._base = base
        self._base_sd = {k: v.detach() for k, v in base.state_dict().items()}

        def f(sd, obs):
            out = functional_call(base, sd, ({'obs': obs, 'is_train': False},))
            mus = out['mus']
            sig = out.get('sigmas')
            return mus, (torch.zeros_like(mus) if sig is None else sig)

        self._vforward = vmap(f, randomness='different')
        self._stacked = None
        self._slot_of = {}
        self._groups = None

    def _member_tensors(self, state_dict):
        """Map a member's state dict onto the template's keys (device, dtype);
        missing keys fall back to the template (load_state_dict strict=False)."""
        sd = _strip_prefix(state_dict)
        unexpected = [k for k in sd if k not in self._base_sd]
        if unexpected:
            print(f'[EnvpoolSoccer] pool member: unexpected keys {unexpected[:3]}')
        return {k: (sd[k].detach().to(device=v.device, dtype=v.dtype) if k in sd else v)
                for k, v in self._base_sd.items()}

    def _rebuild_stack(self, members):
        """members: {member_id: state_dict} -> stacked tensors + slot map."""
        if self._base is None:
            raise RuntimeError('set_opponent_template() must be called before pushing weights')
        ids = sorted(int(m) for m in members)
        trees = [self._member_tensors(members[m]) for m in ids]
        self._stacked = {k: torch.stack([t[k] for t in trees]) for k in self._base_sd} if trees else None
        self._slot_of = {m: i for i, m in enumerate(ids)}
        self._groups = None

    def _slot_state_dict(self, member_id):
        s = self._slot_of[member_id]
        return {k: v[s] for k, v in self._stacked.items()}

    def set_main_params(self, state_dict):
        if MAIN_ID in self._slot_of:
            s = self._slot_of[MAIN_ID]
            for k, v in self._member_tensors(state_dict).items():
                self._stacked[k][s].copy_(v)
            return
        members = {m: self._slot_state_dict(m) for m in self._slot_of}
        members[MAIN_ID] = state_dict
        self._rebuild_stack(members)

    def set_pool(self, params):
        """params: {member_id: state_dict}. Members absent from params are dropped
        (except MAIN_ID, refreshed via set_main_params)."""
        members = {int(m): sd for m, sd in params.items()}
        if MAIN_ID not in members and MAIN_ID in self._slot_of:
            members[MAIN_ID] = self._slot_state_dict(MAIN_ID)
        self._rebuild_stack(members)

    def set_pool_assignment(self, member_ids, params=None):
        """Desired opponent id per match; applied when each match next resets."""
        member_ids = np.asarray(member_ids, dtype=np.int64).reshape(-1)
        if member_ids.shape[0] != self.num_envs:
            raise ValueError(f'expected {self.num_envs} ids, got {member_ids.shape[0]}')
        if params is not None:
            self.set_pool(params)
        self._pending_assignment = member_ids.copy()

    def current_assignment(self):
        return self._match_opp.copy()

    def current_opponent_sigma_scale(self):
        return self._opp_sigma_scale.copy()

    def set_pair_assignment(self, pairs):
        """Population mode: desired (home_slot, away_slot) per match; applied
        when each match next resets."""
        if self.population <= 0:
            raise RuntimeError('set_pair_assignment needs population mode')
        pairs = np.asarray(pairs, dtype=np.int64).reshape(self.num_envs, 2)
        if pairs.min() < 0 or pairs.max() >= self.population:
            raise ValueError('slot out of range')
        self._pending_pairs = pairs.copy()

    def current_pairs(self):
        return self._pairs.copy()

    def set_shaping_scale(self, scale):
        """Scale all shaping terms except the goal reward (curriculum -> sparse)."""
        self.shaping_scale = float(scale)

    # legacy SelfPlayManager hook: push latest weights as MAIN and play it everywhere
    def set_weights(self, indices, weights):
        sd = weights['model'] if isinstance(weights, dict) and 'model' in weights else weights
        self.set_main_params(sd)
        if self._pending_assignment is None:
            self.set_pool_assignment(np.full(self.num_envs, MAIN_ID))

    # ------------------------------------------------------------- internals

    def _apply_pending(self, mask):
        """Called for matches that just reset: apply the pending opponent ids
        and redraw their sigma scale."""
        if not mask.any():
            return
        lo, hi = self.opponent_sigma_range
        self._opp_sigma_scale[mask] = self._rng.uniform(lo, hi, size=int(mask.sum()))
        if self._pending_pairs is not None:
            self._pairs[mask] = self._pending_pairs[mask]
        if self._pending_assignment is None:
            return
        self._match_opp[mask] = self._pending_assignment[mask]
        self._groups = None

    def _update_perm(self, info):
        """envpool returns matches in completion order; build the row permutation
        that puts player rows into env-major order (players keep their in-match
        order, home team first). Actions are sent env-major with the default
        env_id, so they need no permutation."""
        pid = np.asarray(info['players']['env_id']).reshape(-1)
        if pid.shape[0] != self.total_players:
            raise RuntimeError(f'expected {self.total_players} player rows, got {pid.shape[0]}')
        perm = np.argsort(pid, kind='stable')
        self._perm = None if np.array_equal(perm, self._identity) else perm

    def _ordered(self, arr, info=None):
        return arr if self._perm is None else arr[self._perm]

    def _flatten_obs(self, obs, info):
        parts = [self._ordered(np.asarray(obs[k]), info).reshape(self.total_players, -1)
                 for k in self.obs_keys]
        flat = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        flat = flat.reshape(self.num_envs, self.players, self.base_obs_dim)
        if self._pid_onehot is not None:
            pid = np.broadcast_to(self._pid_onehot, (self.num_envs,) + self._pid_onehot.shape)
            flat = np.concatenate([flat, pid], axis=2)
        if self.population > 0:
            row_slot = np.repeat(self._pairs, self.team_size, axis=1)          # (E, players)
            onehot = np.eye(self.population, dtype=np.float32)[row_slot]        # (E, players, N)
            flat = np.concatenate([flat, onehot], axis=2)
        return flat

    def _stat(self, obs, info, key):
        return self._ordered(np.asarray(obs[key]), info).reshape(self.num_envs, self.players)

    def _process_obs(self, obs, info):
        self.last_raw_obs, self.last_raw_info = obs, info   # for eval/diagnostics
        flat = torch.from_numpy(self._flatten_obs(obs, info)).to(self.device)
        if self.population > 0:
            self._away_obs = None
            return flat.reshape(self.total_players, self.obs_dim)
        home = flat[:, :self.team_size].reshape(self.num_home, self.obs_dim)
        self._away_obs = flat[:, self.team_size:]
        return home

    def _slot_groups(self):
        """Group matches by pool slot: (idx (M, G), mask (M, G), random mask).
        G = largest group; padded entries point at match 0 and are masked."""
        if self._groups is None:
            if self.opponent == 'pool' and self._stacked is not None:
                slots = np.array([self._slot_of.get(int(m), -1) for m in self._match_opp], dtype=np.int64)
            else:
                slots = np.full(self.num_envs, -1, dtype=np.int64)
            rand = slots < 0
            M = len(self._slot_of) if self._stacked is not None else 0
            groups = [np.nonzero(slots == s)[0] for s in range(M)]
            G = max([len(g) for g in groups] + [1])
            idx = np.zeros((M, G), dtype=np.int64)
            mask = np.zeros((M, G), dtype=bool)
            for s, g in enumerate(groups):
                idx[s, :len(g)] = g
                mask[s, :len(g)] = True
            self._groups = (idx, mask, rand, torch.from_numpy(idx).to(self.device))
        return self._groups

    def _opponent_actions(self):
        out = np.empty((self.num_envs, self.team_size, self.act_dim), dtype=np.float32)
        idx, mask, rand, idx_t = self._slot_groups()
        if rand.any():
            out[rand] = self._rng.uniform(-1.0, 1.0, size=(int(rand.sum()), self.team_size, self.act_dim))
        if not mask.any():
            return out
        M, G = idx.shape
        obs = self._away_obs[idx_t].reshape(M, G * self.team_size, self.obs_dim)
        with torch.no_grad():
            mus, sig = self._vforward(self._stacked, obs)
            act = mus
            if not self.opponent_deterministic:
                scale = torch.from_numpy(self._opp_sigma_scale[idx]).to(self.device)
                scale = scale.repeat_interleave(self.team_size, dim=1)[..., None]
                act = mus + sig * scale * torch.randn_like(mus)
            act = act.clamp(-1.0, 1.0).reshape(M, G, self.team_size, self.act_dim).cpu().numpy()
        out[idx[mask]] = act[mask]
        return out

    def _all_rewards(self, obs, info, reward):
        """Shaped reward for every player row (E, players); each team uses its
        own players' stats (envpool computes stats_* relative to each player's
        goal). Returns (rewards, home goal signal)."""
        T, P = self.team_size, self.players
        reward = self._ordered(np.asarray(reward, dtype=np.float32), info).reshape(self.num_envs, P)
        goal = reward
        closest = self._stat(obs, info, 'stats_closest_vel_to_ball')
        # only the closest teammate reports a non-zero value; share it across the team
        closest = np.concatenate(
            [np.repeat(closest[:, t * T:(t + 1) * T].sum(axis=1, keepdims=True), T, axis=1)
             for t in range(P // T)], axis=1)
        if self.clip_vel_to_ball:
            closest = np.maximum(closest, 0.0)
        vbg = self._stat(obs, info, 'stats_vel_ball_to_goal')
        fwd = self._stat(obs, info, 'stats_veloc_forward')
        k = self.shaping_scale
        r = (self.shaping['goal'] * goal
             + k * self.shaping['vel_to_ball'] * closest
             + k * self.shaping['vel_ball_to_goal'] * vbg
             + k * self.shaping['veloc_forward'] * fwd)
        if self.shaping['spread_out'] != 0.0:
            spread = self._ordered(np.asarray(obs['stats_teammate_spread_out'])).reshape(
                self.num_envs, P).astype(np.float32)
            r = r + k * self.shaping['spread_out'] * spread
        # stuck detector: the learner's robots (home team, or everyone in
        # population mode) and the ball (velocity relative to home player 0's
        # body) below stuck_speed for stuck_steps control steps
        n_robots = P if self.population > 0 else T
        vel = self._ordered(np.asarray(obs['sensors_velocimeter'])).reshape(
            self.num_envs, P, -1)[:, :n_robots, :2]
        robots_still = (np.linalg.norm(vel, axis=-1) < self.stuck_speed).all(axis=1)
        ball_rel = self._ordered(np.asarray(obs['ball_ego_linear_velocity'])).reshape(
            self.num_envs, self.players, -1)[:, 0, :2]
        still = robots_still & (np.linalg.norm(ball_rel, axis=-1) < self.stuck_speed)
        self._still_steps = np.where(still, self._still_steps + 1, 0)
        stuck = self._still_steps >= self.stuck_steps
        self._stuck_steps_total += stuck
        if self.shaping['stuck_penalty'] != 0.0:
            r = r - k * self.shaping['stuck_penalty'] * stuck[:, None].astype(np.float32)
        return r.astype(np.float32), goal[:, 0]

    def _home_rewards(self, obs, info, reward):
        r, goal = self._all_rewards(obs, info, reward)
        return r[:, :self.team_size], goal

    # --------------------------------------------------------------- IVecEnv

    def reset(self):
        obs, info = self.env.reset()
        self._update_perm(info)
        self._goals_home[:] = 0
        self._goals_away[:] = 0
        self._match_steps[:] = 0
        self._still_steps[:] = 0
        self._stuck_steps_total[:] = 0
        self._need_reset[:] = False
        self._apply_pending(np.ones(self.num_envs, dtype=bool))
        return self._process_obs(obs, info)

    def step(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().float().cpu().numpy()
        rows = self.players if self.population > 0 else self.team_size
        if self.population > 0:
            full = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0).reshape(self.total_players, self.act_dim)
        else:
            home = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
            home = home.reshape(self.num_envs, self.team_size, self.act_dim)
            away = self._opponent_actions()
            full = np.concatenate([home, away], axis=1).reshape(self.total_players, self.act_dim)
        obs, reward, terminated, truncated, info = self.env.step(full.astype(np.float64))
        self._update_perm(info)
        # match rows come back in completion order too
        order = np.argsort(np.asarray(info['env_id']).reshape(-1), kind='stable')
        reward = np.asarray(reward, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)[order]
        truncated = np.asarray(truncated, dtype=bool).reshape(-1)[order]

        # envpool auto-reset: the step after a finished match returns the fresh
        # match's first observation with zero reward; its (ignored) action must
        # not be credited. Apply pending opponent assignments for those matches.
        fresh = self._need_reset.copy()
        self._apply_pending(fresh)

        if self.population > 0:
            rewards, goal_signal = self._all_rewards(obs, info, reward)
        else:
            rewards, goal_signal = self._home_rewards(obs, info, reward)
        rewards[fresh] = 0.0
        done = terminated | truncated
        done &= ~fresh

        self._goals_home += (goal_signal > 0) & ~fresh
        self._goals_away += (goal_signal < 0) & ~fresh
        self._match_steps += 1
        self._match_steps[fresh] = 1
        self._still_steps[fresh] = 0
        self._stuck_steps_total[fresh] = 0

        infos = {
            'time_outs': torch.from_numpy(np.repeat(truncated & ~fresh, rows)).to(self.device),
        }
        if done.any():
            diff = self._goals_home - self._goals_away
            infos['goal_diff'] = torch.from_numpy(diff.astype(np.float32)).to(self.device)
            infos['home_goals'] = torch.from_numpy(self._goals_home.astype(np.float32)).to(self.device)
            infos['away_goals'] = torch.from_numpy(self._goals_away.astype(np.float32)).to(self.device)
            infos['win'] = torch.from_numpy(np.sign(diff).astype(np.float32)).to(self.device)
            infos['opp_id'] = torch.from_numpy(self._match_opp.copy()).to(self.device)
            infos['match_len'] = torch.from_numpy(self._match_steps.astype(np.float32)).to(self.device)
            infos['stuck_frac'] = torch.from_numpy(
                (self._stuck_steps_total / np.maximum(self._match_steps, 1)).astype(np.float32)).to(self.device)
            infos['scores'] = infos['goal_diff']
            if self.population > 0:
                infos['home_slot'] = torch.from_numpy(self._pairs[:, 0].copy()).to(self.device)
                infos['away_slot'] = torch.from_numpy(self._pairs[:, 1].copy()).to(self.device)
                infos['opp_id'] = infos['away_slot']
            self._goals_home[done] = 0
            self._goals_away[done] = 0
            self._match_steps[done] = 0
        self._need_reset = done

        home_obs = self._process_obs(obs, info)
        rewards_t = torch.from_numpy(rewards.reshape(self.num_envs * rows)).to(self.device)
        dones_t = torch.from_numpy(np.repeat(done, rows)).to(self.device)
        return home_obs, rewards_t, dones_t, infos

    def get_number_of_agents(self):
        return self.players if self.population > 0 else self.team_size

    def has_action_mask(self):
        return False

    def get_env_info(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'agents': self.get_number_of_agents(),
        }

    def render(self, env_ids=None, **kwargs):
        return self.env.render(env_ids=env_ids if env_ids is not None else [0])

    def close(self):
        self.env.close()


def create_envpool_soccer(**kwargs):
    return EnvpoolSoccerVecEnv('', kwargs.pop('num_actors', 16), **kwargs)
