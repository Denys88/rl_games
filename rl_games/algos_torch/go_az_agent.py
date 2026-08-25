"""Gumbel AlphaZero + Reanalyze agent for Go 9x9, in rl_games architecture.

Registered as `algo: name: go_az` — configured entirely from YAML, uses the
standard ModelBuilder network (go_resnet under discrete_a2c), the runner's
observer/writer/checkpoint conventions, and the GoPlayer for inference.
Epoch == one self-play generation:

    1. self-play `games_per_gen` games in one JAX batch, every move from
       Gumbel MCTS on the current flax params
    2. Reanalyze: refresh a chunk of old buffer policy targets with the
       current net
    3. torch updates: CE(policy vs search weights) + MSE(value vs shaped
       outcome) + aux heads, random dihedral augmentation per sample
    4. torch -> flax sync for the next generation

Checkpoints are ModelA2C state dicts ('a2c_network.'-prefixed), loadable by
PpoPlayerDiscrete/GoPlayer and every existing eval script. `--checkpoint`
warm-starts from any go_resnet checkpoint, PPO ones included.

Config block (under params.config):

    alphazero:
      games_per_gen: 768
      num_simulations: 16
      max_num_considered_actions: 16
      temperature_plies: 24      # opening sampling from action_weights
      batch_size: 1024
      train_ratio: 1.0           # updates per new position
      reanalyze_frac: 0.4
      buffer_positions: 1500000
      eval_every: 10             # generations
      eval_games: 256
      eval_temperature: 0.35     # AZ policies are soft; temp-1 evals mislead
"""

import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import DefaultAlgoObserver


class GoAZAgent:
    def __init__(self, base_name, params):
        self.config = config = params['config']
        self.az = dict(config.get('alphazero', {}))
        self.device = config.get('device', 'cuda:0')
        self.ppo_device = self.device  # observer compatibility
        self.games_to_track = config.get('games_to_track', 100)
        self.max_epochs = config.get('max_epochs', 100000)
        self.save_freq = config.get('save_frequency', 10)
        self.env_config = config.get('env_config', {})
        self.komi = float(self.env_config.get('komi', 7.0))

        # ---- model through the standard builder (player-compatible ckpts)
        from rl_games.algos_torch import model_builder
        builder = model_builder.ModelBuilder()
        self.network = builder.load(params)
        self.model = self.network.build({
            'actions_num': 82, 'input_shape': (9, 9, 17), 'num_seqs': 1,
            'value_size': 1, 'normalize_value': False, 'normalize_input': False,
        }).to(self.device)
        self.net = self.model.a2c_network
        self.arch = {k: params['network'].get(k, d) for k, d in
                     (('blocks', 6), ('channels', 64),
                      ('gpool_every', 2), ('value_units', 128))}

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.get('learning_rate', 5e-5)),
            weight_decay=float(config.get('weight_decay', 1e-4)))

        # ---- experiment dirs / writer / observer (rl_games conventions)
        self.experiment_name = config.get(
            'full_experiment_name',
            config['name'] + datetime.now().strftime('_%d-%H-%M-%S'))
        train_dir = config.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        os.makedirs(self.nn_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(os.path.join(self.experiment_dir, 'summaries'))
        self.algo_observer = config.get('features', {}).get('observer') \
            or DefaultAlgoObserver()
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.epoch_num = 0
        self.frame = 0

        # sym tables for train-time dihedral augmentation
        from rl_games.envs.pgx_go import _build_sym_tables
        self.sym_tables = torch.from_numpy(
            _build_sym_tables(9).astype(np.int64)).to(self.device)

        # JAX side built lazily in train() (keeps construction cheap)
        self._jax_ready = False
        self._eval_env = None

    # ------------------------------------------------------------ jax setup

    def _init_jax(self):
        if self._jax_ready:
            return
        import jax
        from pgx import go
        from rl_games.envs.go_az import (GameBuffer, make_selfplay,
                                         make_reanalyzer)
        self._jax = jax
        self.genv = go.Go(size=9, komi=self.komi)
        self.selfplay = make_selfplay(
            self.genv, self.arch,
            num_simulations=self.az.get('num_simulations', 16),
            max_num_considered_actions=self.az.get('max_num_considered_actions', 16),
            temperature_plies=self.az.get('temperature_plies', 24),
            komi=self.komi)
        self.reanalyzer = make_reanalyzer(
            self.genv, self.arch,
            num_simulations=self.az.get('num_simulations', 16),
            max_num_considered_actions=self.az.get('max_num_considered_actions', 16))
        self.buffer = GameBuffer(self.az.get('buffer_positions', 1_500_000))
        self.rng = jax.random.PRNGKey(self.config.get('seed', 0) or 0)
        self._jax_ready = True

    def _to_flax(self):
        from rl_games.envs.go_flax import params_from_torch
        return params_from_torch(self.model.state_dict(), **self.arch)

    # -------------------------------------------------------- save/restore

    def get_full_state_weights(self):
        return {'model': self.model.state_dict(), 'epoch': self.epoch_num,
                'optimizer': self.optimizer.state_dict(), 'frame': self.frame,
                'go_az': True}

    def save(self, fn):
        torch_ext.save_checkpoint(fn, self.get_full_state_weights())

    def restore(self, fn, set_epoch=True):
        ckpt = torch_ext.load_checkpoint(fn)
        sd = ckpt['model']
        own = self.model.state_dict()
        clean = {}
        for k, v in sd.items():
            k = k.replace('_orig_mod.', '')
            if k not in own and 'a2c_network.' + k in own:
                k = 'a2c_network.' + k  # bare-net checkpoint (script-era AZ)
            clean[k] = v
        filtered = {k: v for k, v in clean.items() if k in own}
        self.model.load_state_dict(filtered, strict=False)
        print(f'[go_az] restored {len(filtered)}/{len(own)} tensors from {fn}')
        # A PPO checkpoint's value head predicts NORMALIZED values (trained
        # behind RunningMeanStd). AZ search and MSE use raw outputs, so fold
        # the normalizer into value_fc2: v_true = sqrt(var+eps)*v_norm + mean.
        vm = clean.get('value_mean_std.running_mean')
        vv = clean.get('value_mean_std.running_var')
        if vm is not None and vv is not None and not ckpt.get('go_az'):
            std = float(np.sqrt(float(vv.reshape(-1)[0]) + 1e-5))
            mean = float(vm.reshape(-1)[0])
            with torch.no_grad():
                fc2 = self.net.value_fc2
                fc2.weight.mul_(std)
                fc2.bias.mul_(std).add_(mean)
            print(f'[go_az] folded value normalizer into head '
                  f'(std {std:.3f}, mean {mean:.3f})')
        # only resume counters/optimizer from our own checkpoints — a foreign
        # (e.g. PPO) checkpoint is a weight-only warm start
        if ckpt.get('go_az') and set_epoch:
            self.epoch_num = ckpt.get('epoch', 0)
            self.frame = ckpt.get('frame', 0)
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass

    # -------------------------------------------------------------- eval

    def _evaluate(self):
        from rl_games.envs.go_anchors import make_baseline_opponent
        from rl_games.envs.pgx_go import PgxGoVecEnv
        games = self.az.get('eval_games', 256)
        temp = float(self.az.get('eval_temperature', 0.35))
        if self._eval_env is None:
            self._eval_env = PgxGoVecEnv(
                'pgx_go', games, komi=self.komi, symmetry=False, seed=97531,
                opponent=make_baseline_opponent('baselines'))
        env = self._eval_env
        self.model.eval()
        obs = env.reset()
        wins = n = 0
        while n < games:
            mask = env.get_action_masks()
            with torch.no_grad():
                logits, _, _ = self.net({'obs': obs})
            logits[~mask] = -torch.inf
            a = torch.multinomial(
                torch.softmax(logits / max(temp, 1e-6), -1), 1).squeeze(1)
            obs, _, d, info = env.step(a)
            db = d.bool()
            if db.any():
                wins += int((info['win'][db] > 0).sum().item())
                n += int(db.sum().item())
        self.model.train()
        return wins / n

    # -------------------------------------------------------------- train

    def _train_steps(self, new_positions):
        az = self.az
        batch = az.get('batch_size', 1024)
        steps = max(1, int(new_positions * az.get('train_ratio', 1.0) / batch))
        pl_sum = vl_sum = 0.0
        aux_sums = {}
        for _ in range(steps):
            ci, gi, pi = self.buffer.sample(batch)
            obs_np, pol_np, z_np, own_np, score_np, pleft_np = \
                self.buffer.gather(ci, gi, pi)
            dev = self.device
            obs = torch.from_numpy(obs_np).to(dev).float().reshape(-1, 81, 17)
            pol = torch.from_numpy(pol_np).to(dev)
            zt = torch.from_numpy(z_np).to(dev)
            own = torch.from_numpy(own_np).to(dev)
            score = torch.from_numpy(score_np).to(dev)
            pleft = torch.from_numpy(pleft_np).to(dev)
            sym = torch.randint(0, 8, (obs.shape[0],), device=dev)
            idx81 = self.sym_tables[sym][:, :81]
            obs = torch.gather(obs, 1, idx81.unsqueeze(-1).expand(-1, -1, 17))
            pol = torch.gather(pol, 1, self.sym_tables[sym])
            own = torch.gather(own, 1, idx81)
            obs = obs.reshape(-1, 9, 9, 17)

            logits, value, _ = self.net({
                'obs': obs, 'is_train': True,
                'go_ownership': own,
                'go_terminal_valid': torch.ones_like(zt),
                'go_score': score, 'go_plies_left': pleft})
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pol * logp).sum(-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), zt)
            aux = self.net.get_aux_loss() or {}
            loss = policy_loss + value_loss + sum(aux.values())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            pl_sum += policy_loss.item()
            vl_sum += value_loss.item()
            for k, v in aux.items():
                aux_sums[k] = aux_sums.get(k, 0.0) + v.item()
        return steps, pl_sum / steps, vl_sum / steps, \
            {k: v / steps for k, v in aux_sums.items()}

    def train(self):
        self._init_jax()
        jax = self._jax
        az = self.az
        self.algo_observer.after_init(self)
        flax_params = self._to_flax()
        last_eval = float('nan')
        start = time.perf_counter()

        while self.epoch_num < self.max_epochs:
            self.epoch_num += 1
            t0 = time.perf_counter()
            self.rng, k = jax.random.split(self.rng)
            data = self.selfplay(flax_params, k, az.get('games_per_gen', 768))
            self.buffer.add(data)
            new_pos = int(data['lengths'].sum())
            self.frame += new_pos
            sp_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            re_n = int(az.get('batch_size', 1024) * az.get('train_ratio', 1.0)
                       * az.get('reanalyze_frac', 0.4))
            if re_n > 0 and len(self.buffer.chunks) > 1:
                ci, gi, pi = self.buffer.sample(re_n)
                moves = self.buffer.moves_for(ci, gi)
                self.rng, k = jax.random.split(self.rng)
                fresh = self.reanalyzer(flax_params, moves, pi, k)
                self.buffer.write_weights(ci, gi, pi, fresh)
            re_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            steps, pl, vl, aux = self._train_steps(new_pos)
            tr_t = time.perf_counter() - t0
            flax_params = self._to_flax()

            fps = new_pos / (sp_t + re_t + tr_t)
            self.writer.add_scalar('losses/policy_ce', pl, self.frame)
            self.writer.add_scalar('losses/value_mse', vl, self.frame)
            for k2, v in aux.items():
                self.writer.add_scalar('losses/' + k2, v, self.frame)
            self.writer.add_scalar('info/buffer_positions',
                                   self.buffer.total, self.frame)
            self.writer.add_scalar('info/game_plies',
                                   float(data['lengths'].mean()), self.frame)
            self.writer.add_scalar('performance/positions_per_sec', fps, self.frame)
            print(f'fps az: {fps:,.0f} gen: {self.epoch_num}/{self.max_epochs} '
                  f'positions: {self.frame} pl {pl:.3f} vl {vl:.3f} '
                  f'(sp {sp_t:.1f}s re {re_t:.1f}s tr {tr_t:.1f}s)', flush=True)

            if self.epoch_num % az.get('eval_every', 10) == 0:
                last_eval = self._evaluate()
                self.writer.add_scalar('go/eval_winrate_baseline',
                                       last_eval, self.frame)
                print(f'[go_az] gen {self.epoch_num}: eval vs baseline = '
                      f'{last_eval:.3f}', flush=True)
                self.algo_observer.after_print_stats(
                    self.frame, self.epoch_num, time.perf_counter() - start)

            if self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(
                    self.nn_dir,
                    f'last_{self.config["name"]}_gen_{self.epoch_num}'))

        self.save(os.path.join(self.nn_dir, self.config['name']))
        return last_eval, self.epoch_num
