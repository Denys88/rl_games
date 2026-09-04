"""Soccer observers: match statistics and the league driver.

Attached via scripts/soccer_train.py (yaml configs cannot inject observers):

    runner = Runner(SoccerLeagueObserver(league_config=cfg['params']['config']['league']))

SoccerObserver logs (all under soccer/), computed from finished matches:
  - goal_diff, winrate, drawrate, goals_for, goals_against, match_len
  - winrate_vs_{random,self,pool}, goal_diff_vs_{random,self,pool}
    (split by the away team's pool member id — RANDOM_ID / MAIN_ID / other).
    The random slice doubles as a free "vs random" eval, no extra env needed.

Shaping anneal (both observers, config key ``shaping_anneal``): every shaping
term except the goal reward is multiplied by a scale that goes linearly from 1
at ``start_epoch`` to ``floor`` (default 0) at ``end_epoch`` (env.set_shaping_scale), logged as
soccer/shaping_scale. Dense shaping bootstraps play, the sparse goal reward
takes over.

SoccerPopulationObserver drives the population league (env_config.population:
N + population_actor_critic): N x N payoff matrix from finished matches,
(home_slot, away_slot) matchmaking every remap_every epochs, per-slot
standalone checkpoints in nn/slots/.

SoccerLeagueObserver (AlphaStar-lite, mirrors GoLeagueObserver) additionally
  - records finished matches into the League payoff row (win=1, draw=0.5),
  - snapshots main into the pool on schedule or on dominance,
  - resamples the per-match opponent assignment every remap_every epochs
    (self / PFSP / uniform / random anchor); the env applies it at match reset,
  - refreshes the latest-main weights in the self-play slots every
    push_main_every epochs.
"""

import os

import numpy as np
import torch

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
from rl_games.envs.envpool_soccer import MAIN_ID, RANDOM_ID, state_dict_to_cpu


def _opponent_kind(opp_id):
    if opp_id == RANDOM_ID:
        return 'random'
    if opp_id == MAIN_ID:
        return 'self'
    return 'pool'


class SoccerObserver(AlgoObserver):
    KINDS = ('random', 'self', 'pool')

    def __init__(self, anneal_config=None):
        super().__init__()
        self.algo = None
        self.writer = None
        self.anneal = None
        if anneal_config:
            self.anneal = (int(anneal_config.get('start_epoch', 0)),
                           int(anneal_config['end_epoch']),
                           float(anneal_config.get('floor', 0.0)))

    def shaping_scale(self, epoch_num):
        """1 until start_epoch, linear to `floor` (default 0) at end_epoch."""
        if self.anneal is None:
            return 1.0
        start, end, floor = self.anneal
        if epoch_num <= start:
            return 1.0
        if epoch_num >= end:
            return floor
        return floor + (1.0 - floor) * (1.0 - (epoch_num - start) / float(end - start))

    def _meter(self):
        return torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        self.meters = {k: self._meter() for k in
                       ('goal_diff', 'win', 'draw', 'goals_for', 'goals_against', 'match_len', 'stuck_frac')}
        self.kind_meters = {kind: {'win': self._meter(), 'goal_diff': self._meter()}
                            for kind in self.KINDS}

    def _match_indices(self, done_indices):
        idx = done_indices.flatten()
        if torch.is_tensor(idx):
            idx = idx.cpu().numpy()
        return np.unique(np.asarray(idx) // self.algo.num_agents)

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or 'goal_diff' not in infos:
            return
        idx = self._match_indices(done_indices)
        if len(idx) == 0:
            return
        diff = infos['goal_diff'][idx].float().unsqueeze(1)
        win = (diff > 0).float()
        draw = (diff == 0).float()
        self.meters['goal_diff'].update(diff)
        self.meters['win'].update(win)
        self.meters['draw'].update(draw)
        self.meters['goals_for'].update(infos['home_goals'][idx].float().unsqueeze(1))
        self.meters['goals_against'].update(infos['away_goals'][idx].float().unsqueeze(1))
        for k in ('match_len', 'stuck_frac'):
            if k in infos:
                self.meters[k].update(infos[k][idx].float().unsqueeze(1))
        if 'opp_id' in infos:
            opp = infos['opp_id'][idx].cpu().numpy()
            for kind in self.KINDS:
                sel = np.array([_opponent_kind(int(o)) == kind for o in opp])
                if sel.any():
                    sel_t = torch.from_numpy(sel).to(diff.device)
                    self.kind_meters[kind]['win'].update(win[sel_t])
                    self.kind_meters[kind]['goal_diff'].update(diff[sel_t])
            self._record_league(opp, diff.squeeze(1).cpu().numpy())
        self._record_match(infos, idx, diff.squeeze(1).cpu().numpy())

    def _record_league(self, opp_ids, goal_diffs):
        pass

    def _record_match(self, infos, idx, goal_diffs):
        pass

    def after_clear_stats(self):
        for m in self.meters.values():
            m.clear()
        for d in self.kind_meters.values():
            for m in d.values():
                m.clear()

    def _log(self, tag, meter, frame):
        if meter.current_size > 0 and self.writer is not None:
            self.writer.add_scalar(tag, meter.get_mean().item(), frame)

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.anneal is not None:
            scale = self.shaping_scale(epoch_num)
            self.algo.vec_env.set_shaping_scale(scale)
            if self.writer is not None:
                self.writer.add_scalar('soccer/shaping_scale', scale, frame)
        for name, m in self.meters.items():
            if name not in ('win', 'draw'):    # logged as winrate/drawrate below
                self._log(f'soccer/{name}', m, frame)
        for kind, d in self.kind_meters.items():
            self._log(f'soccer/winrate_vs_{kind}', d['win'], frame)
            self._log(f'soccer/goal_diff_vs_{kind}', d['goal_diff'], frame)
        self._log('soccer/winrate', self.meters['win'], frame)
        self._log('soccer/drawrate', self.meters['draw'], frame)


class SoccerLeagueObserver(SoccerObserver):

    def __init__(self, league_config=None, seed_pool=None, anneal_config=None):
        super().__init__(anneal_config=anneal_config)
        cfg = league_config or {}
        # seed_pool: list of rl_games checkpoint paths preloaded into the pool
        # as snapshots (resume after a crash: the pool is not checkpointed).
        self.seed_pool = list(seed_pool or [])
        self.snapshot_every = cfg.get('snapshot_every', 100)
        self.snapshot_if_winrate = cfg.get('snapshot_if_winrate', 0.55)
        self.snapshot_dominance = cfg.get('snapshot_dominance', 0.70)
        self.snapshot_min_gap = cfg.get('snapshot_min_gap',
                                        max(1, cfg.get('snapshot_every', 100) // 4))
        self.remap_every = cfg.get('remap_every', 5)
        self.push_main_every = cfg.get('push_main_every', 1)
        mm = cfg.get('matchmaking', {})
        self.p_self = mm.get('p_self', 0.30)
        self.p_pfsp = mm.get('p_pfsp', 0.40)
        self.p_uniform = mm.get('p_uniform', 0.10)
        self.p_random = mm.get('p_random', 0.20)
        from rl_games.common.league import League
        pfsp = cfg.get('pfsp', {})
        self.league = League(
            max_pool=cfg.get('max_pool', 16),
            pfsp_mode=pfsp.get('mode', 'hard'),
            pfsp_floor=pfsp.get('floor', 0.02),
            payoff_ema=cfg.get('payoff_ema', 0.02),
            variance_warmup_games=cfg.get('variance_warmup_games', 200),
            seed=cfg.get('seed', 0),
            host_fn=state_dict_to_cpu,
        )
        self._last_snapshot_epoch = 0
        self._last_remap_epoch = -1

    def _env(self):
        return self.algo.vec_env

    def _export_main(self):
        return state_dict_to_cpu(self.algo.model.state_dict())

    def after_init(self, algo):
        super().after_init(algo)
        env = self._env()
        env.set_opponent_template(algo.model)
        self._seed_pool_from_checkpoints()
        self._remap(epoch_num=0)

    def _seed_pool_from_checkpoints(self):
        for path in self.seed_pool:
            ckpt = torch_ext.load_checkpoint(path)
            epoch = int(ckpt.get('epoch', 0))
            mid = self.league.add(state_dict_to_cpu(ckpt['model']),
                                  kind='snapshot', epoch=epoch)
            self._last_snapshot_epoch = max(self._last_snapshot_epoch, epoch)
            print(f'[League] seeded member {mid} from {path} (epoch {epoch})')

    def _record_league(self, opp_ids, goal_diffs):
        score = np.where(goal_diffs > 0, 1.0, np.where(goal_diffs < 0, 0.0, 0.5))
        self.league.record(opp_ids, score)

    def _sample_assignment(self):
        n = self._env().num_envs
        n_random = int(round(n * self.p_random))
        # sample_groups fills the first n_self slots with MAIN; carve the random
        # anchor slots out of that share so the pool fractions are untouched
        ids = self.league.sample_groups(n, self.p_self + self.p_random,
                                        self.p_pfsp, self.p_uniform)
        ids[:n_random] = RANDOM_ID
        self.league._rng.shuffle(ids)
        return ids

    def _remap(self, epoch_num):
        env = self._env()
        ids = self._sample_assignment()
        params = {MAIN_ID: self._export_main()}
        for mid in np.unique(ids):
            if mid in self.league.members:
                params[int(mid)] = self.league.members[mid].params
        env.set_pool_assignment(ids, params)
        self._last_remap_epoch = epoch_num

    def _maybe_snapshot(self, epoch_num):
        since = epoch_num - self._last_snapshot_epoch
        if since < max(1, self.snapshot_min_gap):
            return
        due = since >= self.snapshot_every
        if not due:
            stats = self.league.stats()
            rated = [p for m, p in stats['payoff'].items() if stats['counts'][m] > 50]
            dominant = bool(rated) and (
                np.mean([p >= self.snapshot_if_winrate for p in rated]) >= self.snapshot_dominance)
            if not dominant:
                return
        self._last_snapshot_epoch = epoch_num
        before = {m: (mem.created_epoch, self.league.payoff.get(m))
                  for m, mem in self.league.members.items()}
        mid = self.league.add(self._export_main(), kind='snapshot', epoch=epoch_num)
        print(f'[League] epoch {epoch_num}: snapshot -> member {mid} '
              f'(pool size {len(self.league.members)})')
        for m, (ep, p) in before.items():
            if m not in self.league.members:
                print(f'[League] evicted member {m} (epoch {ep}, main winrate {p:.2f})')

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        self._maybe_snapshot(epoch_num)
        if epoch_num % self.remap_every == 0 and epoch_num != self._last_remap_epoch:
            self._remap(epoch_num)
        elif epoch_num % self.push_main_every == 0:
            self._env().set_main_params(self._export_main())
        if self.writer is not None:
            self.writer.add_scalar('league/pool_size', len(self.league.members), frame)
            mwr = self.league.min_winrate_vs_pool()
            if mwr is not None:
                self.writer.add_scalar('league/min_winrate_vs_pool', mwr, frame)
            stats = self.league.stats()
            if stats['payoff']:
                self.writer.add_scalar('league/mean_winrate_vs_pool',
                                       float(np.mean(list(stats['payoff'].values()))), frame)


class SoccerPopulationObserver(SoccerObserver):
    """Population league: N learners in one population_actor_critic network,
    every match pairs two slots (both teams learn). Keeps the N x N payoff
    matrix (EMA of P(row beats column), draw = 0.5), resamples pairings every
    remap_every epochs and writes per-slot standalone checkpoints."""

    def __init__(self, population_config=None, anneal_config=None):
        super().__init__(anneal_config=anneal_config)
        cfg = population_config or {}
        self.remap_every = int(cfg.get('remap_every', 5))
        self.p_self = float(cfg.get('p_self', 0.1))
        self.mode = cfg.get('mode', 'uniform')           # 'uniform' | 'even' (PFSP toward 50/50)
        self.pfsp_floor = float(cfg.get('pfsp_floor', 0.02))
        self.payoff_ema = float(cfg.get('payoff_ema', 0.02))
        self.log_matrix_every = int(cfg.get('log_matrix_every', 50))
        self.slot_save_every = int(cfg.get('slot_save_every', 500))
        self._rng = np.random.RandomState(int(cfg.get('seed', 0)))
        self.N = None
        self.payoff = None
        self.counts = None

    def after_init(self, algo):
        super().after_init(algo)
        self.N = int(algo.vec_env.population)
        self.payoff = np.full((self.N, self.N), 0.5)
        self.counts = np.zeros((self.N, self.N), dtype=np.int64)
        self._remap()

    # ------------------------------------------------------------- results

    def _record_match(self, infos, idx, goal_diffs):
        if 'home_slot' not in infos:
            return
        home = infos['home_slot'][idx].cpu().numpy()
        away = infos['away_slot'][idx].cpu().numpy()
        score = np.where(goal_diffs > 0, 1.0, np.where(goal_diffs < 0, 0.0, 0.5))
        a = self.payoff_ema
        for i, j, s in zip(home, away, score):
            if i == j:
                continue
            self.payoff[i, j] = (1 - a) * self.payoff[i, j] + a * s
            self.payoff[j, i] = (1 - a) * self.payoff[j, i] + a * (1 - s)
            self.counts[i, j] += 1
            self.counts[j, i] += 1

    def slot_winrates(self):
        off = ~np.eye(self.N, dtype=bool)
        return np.array([self.payoff[i][off[i]].mean() for i in range(self.N)])

    # --------------------------------------------------------- matchmaking

    def sample_pairs(self):
        n = self.algo.vec_env.num_envs
        n_self = int(round(n * self.p_self))
        pairs = np.zeros((n, 2), dtype=np.int64)
        s = self._rng.randint(0, self.N, size=n_self)
        pairs[:n_self] = np.stack([s, s], axis=1)
        m = n - n_self
        if self.mode == 'even' and self.N > 1:
            w = np.maximum(self.payoff * (1 - self.payoff), self.pfsp_floor)
            np.fill_diagonal(w, 0.0)
            flat = self._rng.choice(self.N * self.N, size=m, p=(w / w.sum()).ravel())
            pairs[n_self:, 0], pairs[n_self:, 1] = flat // self.N, flat % self.N
        else:
            i = self._rng.randint(0, self.N, size=m)
            j = (i + self._rng.randint(1, max(self.N, 2), size=m)) % self.N
            pairs[n_self:] = np.stack([i, j], axis=1)
        self._rng.shuffle(pairs)
        return pairs

    def _remap(self):
        self.algo.vec_env.set_pair_assignment(self.sample_pairs())

    # ------------------------------------------------------------- logging

    def _save_slots(self, epoch_num):
        from rl_games.algos_torch.population_network import extract_slot
        env = self.algo.vec_env
        base_dim = env.obs_dim - self.N
        out_dir = os.path.join(self.algo.nn_dir, 'slots')
        os.makedirs(out_dir, exist_ok=True)
        sd = self.algo.model.state_dict()
        for k in range(self.N):
            torch.save({'model': extract_slot(sd, k, base_dim), 'epoch': epoch_num, 'slot': k},
                       os.path.join(out_dir, f'slot{k}_ep{epoch_num}.pth'))
        print(f'[Population] epoch {epoch_num}: wrote {self.N} slot checkpoints to {out_dir}')

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if epoch_num % self.remap_every == 0:
            self._remap()
        wr = self.slot_winrates()
        if self.writer is not None:
            for k in range(self.N):
                self.writer.add_scalar(f'population/winrate_slot{k}', float(wr[k]), frame)
            self.writer.add_scalar('population/winrate_spread', float(wr.max() - wr.min()), frame)
        if epoch_num % self.log_matrix_every == 0:
            with np.printoptions(precision=2, suppress=True, linewidth=200):
                print(f'[Population] epoch {epoch_num} payoff (row beats col):\n{self.payoff}')
        if self.slot_save_every > 0 and epoch_num % self.slot_save_every == 0:
            self._save_slots(epoch_num)
