#!/usr/bin/env python
"""Gumbel AlphaZero + Reanalyze trainer for Go 9x9 (plan Phase 4).

Loop per generation:
  1. self-play G games in one JAX batch, every move from Gumbel MCTS
     (policy targets = completed-Q action_weights; value target = shaped
     outcome; aux targets = final ownership / score, full coverage)
  2. reanalyze a chunk of old buffer positions with the current net
     (policy targets refreshed in place, MuZero-style)
  3. train the torch net: CE(policy) + MSE(value) + aux heads, with random
     dihedral augmentation per sample
  4. sync torch -> flax for the next generation's search
Evals vs the AlphaZero baseline + checkpoints every eval_every generations.

Warm-startable from any go_resnet checkpoint (--init).
"""

import argparse
import os
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.35')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def build_sym_tables():
    from rl_games.envs.pgx_go import _build_sym_tables
    return torch.from_numpy(_build_sym_tables(9).astype(np.int64))  # (8, 82)


class GameBuffer:
    """Rolling buffer of finished games (compact numpy per game batch)."""

    def __init__(self, max_positions):
        self.max_positions = max_positions
        self.chunks = []  # list of generation dicts
        self.total = 0

    def add(self, gen):
        self.chunks.append(gen)
        self.total += int(gen['lengths'].sum())
        while self.total > self.max_positions and len(self.chunks) > 1:
            old = self.chunks.pop(0)
            self.total -= int(old['lengths'].sum())

    def sample(self, batch):
        """-> (chunk_idx, game_idx, ply) arrays of length batch."""
        weights = np.array([c['lengths'].sum() for c in self.chunks], dtype=np.float64)
        ci = np.random.choice(len(self.chunks), size=batch, p=weights / weights.sum())
        gi = np.zeros(batch, dtype=np.int64)
        pi = np.zeros(batch, dtype=np.int64)
        for k in range(batch):
            c = self.chunks[ci[k]]
            lens = c['lengths']
            probs = lens / lens.sum()
            g = np.random.choice(len(lens), p=probs)
            gi[k] = g
            pi[k] = np.random.randint(lens[g])
        return ci, gi, pi

    def gather(self, ci, gi, pi):
        b = len(ci)
        obs = np.zeros((b, 1384), dtype=np.uint8)  # unpackbits(173 bytes)
        pol = np.zeros((b, 82), dtype=np.float32)
        z = np.zeros(b, dtype=np.float32)
        own = np.zeros((b, 81), dtype=np.float32)
        score = np.zeros(b, dtype=np.float32)
        plies_left = np.zeros(b, dtype=np.float32)
        for k in range(b):
            c = self.chunks[ci[k]]
            g, t = gi[k], pi[k]
            obs[k] = np.unpackbits(c['obs_bits'][t, g])
            pol[k] = c['weights'][t, g].astype(np.float32)
            mover_black = (t % 2 == 0)
            sign = 1.0 if mover_black else -1.0
            z[k] = c['z_black'][g] * sign
            own[k] = c['ownership_black'][g] * sign
            score[k] = c['score_black'][g] * sign
            plies_left[k] = c['lengths'][g] - t
        return obs[:, :1377], pol, z, own, score, plies_left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default=None, help='warm-start go_resnet checkpoint')
    ap.add_argument('--name', default='go9_az')
    ap.add_argument('--games-per-gen', type=int, default=1024)
    ap.add_argument('--sims', type=int, default=16)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--train-ratio', type=float, default=1.0)
    ap.add_argument('--reanalyze-frac', type=float, default=0.4)
    ap.add_argument('--buffer-positions', type=int, default=1_500_000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--generations', type=int, default=100000)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--eval-games', type=int, default=256)
    args = ap.parse_args()

    from pgx import go
    import jax
    from rl_games.envs.go_network import GoResNetBuilder
    from rl_games.envs.go_flax import params_from_torch
    from rl_games.envs.go_az import make_selfplay, make_reanalyzer
    from torch.utils.tensorboard import SummaryWriter

    ARCH = dict(blocks=6, channels=64, gpool_every=2, value_units=128)
    env = go.Go(size=9, komi=7.0)

    builder = GoResNetBuilder()
    builder.load(dict(ARCH, aux_heads=['ownership', 'score_dist', 'plies_left']))
    net = builder.build('go_resnet', input_shape=(9, 9, 17), actions_num=82).cuda()
    if args.init:
        sd = torch.load(args.init, map_location='cpu', weights_only=False)['model']
        clean = {}
        for k, v in sd.items():
            for pre in ('_orig_mod.', 'a2c_network.'):
                k = k.replace(pre, '')
            clean[k] = v
        missing = net.load_state_dict(
            {k: v for k, v in clean.items() if k in net.state_dict()}, strict=False)
        print(f'[az] warm-start from {args.init} ({missing})', flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    selfplay = make_selfplay(env, ARCH, num_simulations=args.sims)
    reanalyze = make_reanalyzer(env, ARCH, num_simulations=args.sims)
    buffer = GameBuffer(args.buffer_positions)
    sym_tables = build_sym_tables().cuda()

    from rl_games.envs.go_anchors import make_baseline_opponent
    from rl_games.envs.pgx_go import PgxGoVecEnv
    eval_env = None

    outdir = f'runs/{args.name}'
    os.makedirs(outdir + '/nn', exist_ok=True)
    writer = SummaryWriter(outdir + '/summaries')
    rng = jax.random.PRNGKey(0)

    def to_flax():
        return params_from_torch(net.state_dict(), **ARCH)

    def evaluate(gen):
        nonlocal eval_env
        net.eval()
        if eval_env is None:
            eval_env = PgxGoVecEnv('pgx_go', args.eval_games, komi=7.0,
                                   symmetry=False, seed=97531,
                                   opponent=make_baseline_opponent('baselines'))
        obs = eval_env.reset()
        wins = n = 0
        while n < args.eval_games:
            mask = eval_env.get_action_masks()
            with torch.no_grad():
                logits, _, _ = net({'obs': obs})
            logits[~mask] = -torch.inf
            probs = torch.softmax(logits, -1)
            a = torch.multinomial(probs, 1).squeeze(1)
            obs, _, d, info = eval_env.step(a)
            db = d.bool()
            if db.any():
                wins += int((info['win'][db] > 0).sum().item())
                n += int(db.sum().item())
        wr = wins / n
        writer.add_scalar('az/eval_winrate_baseline', wr, gen)
        print(f'[az] gen {gen}: eval vs baseline = {wr:.3f}', flush=True)
        net.train()

    flax_params = to_flax()
    for gen in range(1, args.generations + 1):
        t0 = time.perf_counter()
        rng, k = jax.random.split(rng)
        data = selfplay(flax_params, k, args.games_per_gen)
        buffer.add(data)
        sp_time = time.perf_counter() - t0

        # ---- reanalyze a chunk of old positions in place ----
        t0 = time.perf_counter()
        re_n = int(args.batch * args.train_ratio * args.reanalyze_frac)
        if re_n > 0 and len(buffer.chunks) > 1:
            # one fixed-shape call (variable shapes would retrace the jit)
            ci, gi, pi = buffer.sample(re_n)
            moves = np.zeros((re_n, 162), dtype=np.int32)
            for k in range(re_n):
                moves[k] = buffer.chunks[ci[k]]['moves'][:, gi[k]]
            rng, kk = jax.random.split(rng)
            fresh = reanalyze(flax_params, moves, pi, kk)
            for k in range(re_n):
                buffer.chunks[ci[k]]['weights'][pi[k], gi[k]] = fresh[k]
        re_time = time.perf_counter() - t0

        # ---- torch training ----
        t0 = time.perf_counter()
        new_pos = int(data['lengths'].sum())
        steps = max(1, int(new_pos * args.train_ratio / args.batch))
        pl_l = vl_l = ol_l = 0.0
        for _ in range(steps):
            ci, gi, pi = buffer.sample(args.batch)
            obs_np, pol_np, z_np, own_np, score_np, pleft_np = buffer.gather(ci, gi, pi)
            obs = torch.from_numpy(obs_np).cuda().float().reshape(-1, 81, 17)
            pol = torch.from_numpy(pol_np).cuda()
            zt = torch.from_numpy(z_np).cuda()
            own = torch.from_numpy(own_np).cuda()
            score = torch.from_numpy(score_np).cuda()
            pleft = torch.from_numpy(pleft_np).cuda()
            # random dihedral augmentation per sample
            sym = torch.randint(0, 8, (obs.shape[0],), device='cuda')
            idx81 = sym_tables[sym][:, :81]              # (B, 81)
            obs = torch.gather(obs, 1, idx81.unsqueeze(-1).expand(-1, -1, 17))
            pol = torch.gather(pol, 1, sym_tables[sym])
            own = torch.gather(own, 1, idx81)
            obs = obs.reshape(-1, 9, 9, 17)

            in_dict = {'obs': obs, 'is_train': True,
                       'go_ownership': own, 'go_terminal_valid': torch.ones_like(zt),
                       'go_score': score, 'go_plies_left': pleft}
            logits, value, _ = net(in_dict)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pol * logp).sum(-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), zt)
            aux = net.get_aux_loss() or {}
            loss = policy_loss + 1.0 * value_loss + sum(aux.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            pl_l += policy_loss.item(); vl_l += value_loss.item()
            ol_l += aux.get('ownership', torch.zeros(())).item()
        tr_time = time.perf_counter() - t0
        flax_params = to_flax()

        writer.add_scalar('az/policy_loss', pl_l / steps, gen)
        writer.add_scalar('az/value_loss', vl_l / steps, gen)
        writer.add_scalar('az/ownership_loss', ol_l / steps, gen)
        writer.add_scalar('az/buffer_positions', buffer.total, gen)
        mean_len = float(data['lengths'].mean())
        writer.add_scalar('az/game_plies', mean_len * 1.0, gen)
        print(f'[az] gen {gen}: {new_pos} pos (len {mean_len:.0f}), '
              f'sp {sp_time:.1f}s re {re_time:.1f}s tr {tr_time:.1f}s '
              f'pl {pl_l/steps:.3f} vl {vl_l/steps:.3f}', flush=True)

        if gen % args.eval_every == 0:
            evaluate(gen)
            torch.save({'model': net.state_dict(), 'gen': gen},
                       f'{outdir}/nn/{args.name}_gen{gen}.pth')


if __name__ == '__main__':
    main()
