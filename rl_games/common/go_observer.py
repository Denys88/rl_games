"""Go training observer: win-rate tracking and periodic anchor evaluations.

Attached via scripts/go_train.py (yaml configs cannot inject observers):

    runner = Runner(GoObserver(eval_every=50, eval_games=512))

Logs (all under go/):
  - train_winrate                 : rolling win rate vs the current in-env
                                    opponent (from info['win'])
  - train_winrate_black/_white    : the same split by the learner's colour in
                                    the finished game (komi-balance check)
  - game_plies                    : mean finished-game length in plies
  - score_diff                    : mean terminal score diff, learner persp.
  - eval_winrate_<anchor>,
    eval_score_diff_<anchor>      : sampled policy vs fixed anchors, both
                                    colors, no symmetry, every eval_every epochs

Eval envs are built lazily on first use (each costs a jit compile).
"""

import numpy as np
import torch

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver


class GoObserver(AlgoObserver):
    def __init__(self, eval_every=50, eval_games=512, use_baseline=True,
                 baseline_dir='baselines'):
        super().__init__()
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.use_baseline = use_baseline
        self.baseline_dir = baseline_dir
        self._eval_envs = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        n = algo.games_to_track
        dev = algo.ppo_device
        self.win_meter = torch_ext.AverageMeter(1, n).to(dev)
        self.win_black = torch_ext.AverageMeter(1, n).to(dev)
        self.win_white = torch_ext.AverageMeter(1, n).to(dev)
        self.plies_meter = torch_ext.AverageMeter(1, n).to(dev)
        self.score_meter = torch_ext.AverageMeter(1, n).to(dev)
        self._last_eval_epoch = -1

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or 'win' not in infos:
            return
        idx = done_indices.flatten()
        if len(idx) == 0:
            return
        wins = (infos['win'][idx] > 0).float().unsqueeze(1)
        self.win_meter.update(wins)
        if 'game_color' in infos:
            color = infos['game_color'][idx]
            black, white = color == 0, color == 1
            if black.any():
                self.win_black.update(wins[black])
            if white.any():
                self.win_white.update(wins[white])
        if 'plies' in infos:
            self.plies_meter.update(infos['plies'][idx].float().unsqueeze(1))
        if 'score_diff' in infos:
            self.score_meter.update(infos['score_diff'][idx].float().unsqueeze(1))

    def after_clear_stats(self):
        for m in (self.win_meter, self.win_black, self.win_white,
                  self.plies_meter, self.score_meter):
            m.clear()

    def _build_eval_envs(self):
        from rl_games.envs.pgx_go import PgxGoVecEnv
        from rl_games.envs.go_anchors import make_random_opponent, make_baseline_opponent

        env_cfg = dict(self.algo.env_config)
        common = dict(
            komi=env_cfg.get('komi', 7.0),
            pass_mask_moves=env_cfg.get('pass_mask_moves', 20),
            symmetry=False,
            seed=12345,
        )
        envs = {'random': PgxGoVecEnv('pgx_go', self.eval_games,
                                      opponent=make_random_opponent(), **common)}
        if self.use_baseline:
            try:
                envs['baseline'] = PgxGoVecEnv(
                    'pgx_go', self.eval_games,
                    opponent=make_baseline_opponent(self.baseline_dir), **common)
            except Exception as e:  # download/deps can fail; keep training alive
                print(f'[GoObserver] baseline anchor unavailable: {e}')
        return envs

    @torch.no_grad()
    def _play_vs(self, env):
        obs = env.reset()
        wins = scores = 0.0
        games = 0
        while games < self.eval_games:
            masks = env.get_action_masks()
            obs_dict = self.algo.obs_to_tensors(obs)
            res = self.algo.get_masked_action_values(obs_dict, masks)
            obs, _, done, info = env.step(res['actions'])
            d = done.bool()
            if d.any():
                wins += (info['win'][d] > 0).float().sum().item()
                scores += info['score_diff'][d].float().sum().item()
                games += int(d.sum().item())
        return wins / games, scores / games

    def _log_meter(self, tag, meter, frame):
        if meter.current_size > 0 and self.writer is not None:
            self.writer.add_scalar(tag, meter.get_mean().item(), frame)

    def after_print_stats(self, frame, epoch_num, total_time):
        self._log_meter('go/train_winrate', self.win_meter, frame)
        self._log_meter('go/train_winrate_black', self.win_black, frame)
        self._log_meter('go/train_winrate_white', self.win_white, frame)
        self._log_meter('go/game_plies', self.plies_meter, frame)
        self._log_meter('go/score_diff', self.score_meter, frame)

        # this hook can fire more than once per epoch — evaluate only once
        if (self.eval_every <= 0 or epoch_num % self.eval_every != 0
                or epoch_num == self._last_eval_epoch):
            return
        self._last_eval_epoch = epoch_num
        if self._eval_envs is None:
            self._eval_envs = self._build_eval_envs()
        self.algo.set_eval()
        for name, env in self._eval_envs.items():
            wr, sd = self._play_vs(env)
            print(f'[GoObserver] epoch {epoch_num}: vs {name}: winrate {wr:.3f}, score diff {sd:+.1f}')
            if self.writer is not None:
                self.writer.add_scalar(f'go/eval_winrate_{name}', wr, frame)
                self.writer.add_scalar(f'go/eval_score_diff_{name}', sd, frame)
        self.algo.set_train()


class GoLeagueObserver(GoObserver):
    """League driver (plan Phase 3): replaces SelfPlayManager.

    Env must be PgxGoVecEnv with opponent: pool. Each epoch it
      - records finished games into the league payoff row (win vs opp_id),
      - pushes the latest main params into the self-play group slots,
      - snapshots main into the pool on schedule or on dominance,
      - remaps board groups (self / PFSP / uniform) every remap_every epochs.

    Snapshots and payoff are in-memory only (a restart loses the pool but not
    the learner — acceptable for now; persistence is a TODO).
    """

    def __init__(self, league_config=None, **kwargs):
        super().__init__(**kwargs)
        cfg = league_config or {}
        self.snapshot_every = cfg.get('snapshot_every', 100)
        self.snapshot_if_winrate = cfg.get('snapshot_if_winrate', 0.55)
        self.snapshot_dominance = cfg.get('snapshot_dominance', 0.70)
        self.snapshot_min_gap = cfg.get('snapshot_min_gap',
                                        max(1, cfg.get('snapshot_every', 100) // 4))
        self.remap_every = cfg.get('remap_every', 10)
        self.push_main_every = cfg.get('push_main_every', 5)
        mm = cfg.get('matchmaking', {})
        self.p_self = mm.get('p_self', 0.35)
        self.p_pfsp = mm.get('p_pfsp', 0.50)
        self.p_uniform = mm.get('p_uniform', 0.15)
        from rl_games.common.league import League
        pfsp = cfg.get('pfsp', {})
        self.league = League(
            max_pool=cfg.get('max_pool', 32),
            pfsp_mode=pfsp.get('mode', 'hard'),
            pfsp_floor=pfsp.get('floor', 0.02),
            seed=cfg.get('seed', 0),
        )
        self._last_snapshot_epoch = 0
        self._main_flax = None
        self._last_ids = None

    def _env(self):
        return self.algo.vec_env

    def _export_main(self):
        from rl_games.envs.go_flax import params_from_torch
        env = self._env()
        self._main_flax = params_from_torch(
            self.algo.model.state_dict(), **env._net_cfg)
        return self._main_flax

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)
        if not isinstance(infos, dict) or 'opp_id' not in infos:
            return
        idx = done_indices.flatten()
        if len(idx) == 0:
            return
        wins = (infos['win'][idx] > 0).float().cpu().numpy()
        mids = infos['opp_id'][idx].cpu().numpy()
        self.league.record(mids, wins)

    def _remap(self, resample=True):
        env = self._env()
        if resample or self._main_flax is None:
            self._export_main()
        if resample or self._last_ids is None:
            self._last_ids = self.league.sample_groups(
                env.pool_groups, self.p_self, self.p_pfsp, self.p_uniform)
        stacked = self.league.build_stacked(self._last_ids, self._main_flax)
        env.set_pool_assignment(stacked, self._last_ids)

    def _maybe_snapshot(self, epoch_num):
        # the after_print_stats hook can fire twice per epoch, and the
        # dominance path needs a cooldown or it snapshots every epoch
        since = epoch_num - self._last_snapshot_epoch
        if since < max(1, self.snapshot_min_gap):
            return
        due = since >= self.snapshot_every
        if not due:
            stats = self.league.stats()
            rated = [p for m, p in stats['payoff'].items()
                     if stats['counts'][m] > 100]
            dominant = (rated and
                        np.mean([p >= self.snapshot_if_winrate for p in rated])
                        >= self.snapshot_dominance)
            if not dominant:
                return
        self._last_snapshot_epoch = epoch_num
        params = self._export_main()
        mid = self.league.add(params, kind='snapshot', epoch=epoch_num)
        print(f'[League] epoch {epoch_num}: snapshot -> member {mid} '
              f'(pool size {len(self.league.members)})')

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        self._maybe_snapshot(epoch_num)
        if epoch_num % self.remap_every == 0:
            self._remap(resample=True)
        elif epoch_num % self.push_main_every == 0:
            # refresh the latest-main params in the self-play slots without
            # changing the matchups
            self._export_main()
            self._remap(resample=False)
        if self.writer is not None:
            self.writer.add_scalar('league/pool_size',
                                   len(self.league.members), frame)
            mwr = self.league.min_winrate_vs_pool()
            if mwr is not None:
                self.writer.add_scalar('league/min_winrate_vs_pool', mwr, frame)
