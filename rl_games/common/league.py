"""Checkpoint league for Go self-play (plan Phase 3), AlphaStar-lite.

Holds pool members (flax param pytrees, GPU-resident), a payoff row for the
learner (main) against every member (EMA + counts), PFSP matchmaking over
board groups, snapshot gating and eviction. The env consumes the pool as a
stacked pytree via PgxGoVecEnv.set_pool_assignment; the driver is
GoLeagueObserver (rl_games/common/go_observer.py).

Member kinds:
  main      — the learner; not stored here, always slot id 0
  snapshot  — frozen copies of main
  exploiter — separately trained agents (entered via add())
  anchor    — never sampled for training, eval only (not stored here either)

Payoff convention: payoff[m] = EMA of P(main beats member m).
"""

import numpy as np
import jax
import jax.numpy as jnp

MAIN_ID = 0  # reserved: latest main (self-play)


class Member:
    __slots__ = ('member_id', 'params', 'kind', 'created_epoch')

    def __init__(self, member_id, params, kind, created_epoch):
        self.member_id = member_id
        self.params = params
        self.kind = kind
        self.created_epoch = created_epoch


class League:
    def __init__(self, max_pool=32, pfsp_mode='hard', pfsp_floor=0.02,
                 payoff_ema=0.005, variance_warmup_games=2000, seed=0):
        self.max_pool = max_pool
        self.pfsp_mode = pfsp_mode
        self.pfsp_floor = pfsp_floor
        self.payoff_ema = payoff_ema
        self.variance_warmup_games = variance_warmup_games
        self.members = {}          # member_id -> Member (never MAIN_ID)
        self.payoff = {}           # member_id -> EMA P(main wins)
        self.counts = {}           # member_id -> games recorded
        self._next_id = 1
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------- pool ops

    def add(self, params, kind='snapshot', epoch=0):
        """Add a member; evict if the pool is full. Returns member id.

        Params are moved to host memory — a big-net pool (32 x 26M) would
        otherwise pin gigabytes of GPU; build_stacked re-uploads only the
        pool_groups slices actually assigned."""
        params = jax.device_get(params)
        if len(self.members) >= self.max_pool:
            self._evict()
        mid = self._next_id
        self._next_id += 1
        self.members[mid] = Member(mid, params, kind, epoch)
        self.payoff[mid] = 0.5     # optimistic-neutral start
        self.counts[mid] = 0
        return mid

    def _evict(self):
        """Drop the oldest snapshot main beats hardest. Exploiters survive
        while they still trouble main (win rate vs main > 40%)."""
        candidates = [m for m in self.members.values() if m.kind == 'snapshot']
        if not candidates:
            candidates = [m for m in self.members.values()
                          if self.payoff.get(m.member_id, 0.5) > 0.6]
        if not candidates:
            candidates = list(self.members.values())
        victim = max(candidates,
                     key=lambda m: (self.payoff.get(m.member_id, 0.5),
                                    -m.created_epoch))
        mid = victim.member_id
        del self.members[mid], self.payoff[mid], self.counts[mid]

    # ------------------------------------------------------------- results

    def record(self, member_ids, wins):
        """member_ids, wins: 1-D arrays of finished games (main's side)."""
        member_ids = np.asarray(member_ids).reshape(-1)
        wins = np.asarray(wins).reshape(-1)
        for mid in np.unique(member_ids):
            if mid == MAIN_ID or mid not in self.payoff:
                continue
            w = wins[member_ids == mid]
            n = len(w)
            # batched EMA: apply per-game decay n times
            decay = (1.0 - self.payoff_ema) ** n
            self.payoff[mid] = self.payoff[mid] * decay + w.mean() * (1 - decay)
            self.counts[mid] += n

    # --------------------------------------------------------- matchmaking

    def _pfsp_weights(self, ids):
        w = []
        for mid in ids:
            p = self.payoff[mid]
            if self.counts[mid] < self.variance_warmup_games:
                w.append(max(p * (1 - p), self.pfsp_floor))   # variance mode
            elif self.pfsp_mode == 'hard':
                w.append(max((1 - p) ** 2, self.pfsp_floor))
            else:
                w.append(max(1 - p, self.pfsp_floor))
        w = np.asarray(w, dtype=np.float64)
        return w / w.sum()

    def sample_groups(self, n_groups, p_self=0.35, p_pfsp=0.50, p_uniform=0.15):
        """Member id per board group. Empty pool -> all self-play."""
        ids = sorted(self.members.keys())
        if not ids:
            return np.full(n_groups, MAIN_ID, dtype=np.int64)
        out = np.full(n_groups, MAIN_ID, dtype=np.int64)
        n_self = max(1, int(round(n_groups * p_self)))
        pool_slots = n_groups - n_self
        if pool_slots <= 0:
            return out
        pfsp_w = self._pfsp_weights(ids)
        n_pfsp = int(round(pool_slots * (p_pfsp / max(p_pfsp + p_uniform, 1e-8))))
        picks = list(self._rng.choice(ids, size=n_pfsp, p=pfsp_w))
        picks += list(self._rng.choice(ids, size=pool_slots - n_pfsp))
        self._rng.shuffle(picks)
        out[n_self:] = picks
        return out

    # ------------------------------------------------------------ stacking

    def build_stacked(self, group_ids, main_params):
        """Stack per-group params into one pytree with leading axis n_groups.

        Ids of members evicted since the assignment was sampled fall back to
        main (the assignment may be reused across snapshots/evictions)."""
        trees = [main_params if (mid == MAIN_ID or mid not in self.members)
                 else self.members[mid].params
                 for mid in group_ids]
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)

    # ------------------------------------------------------------- metrics

    def min_winrate_vs_pool(self):
        rated = [self.payoff[m] for m in self.members if self.counts[m] > 100]
        return min(rated) if rated else None

    def stats(self):
        return {
            'pool_size': len(self.members),
            'min_winrate_vs_pool': self.min_winrate_vs_pool(),
            'payoff': dict(self.payoff),
            'counts': dict(self.counts),
        }
