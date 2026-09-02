#!/usr/bin/env python
"""Evaluate a big-net (b18c384nbt) checkpoint vs the small-net champion.

  1) sampled winrate vs the AlphaZero baseline (reference number)
  2) raw vs raw:      big (sampled) vs champion (flax opponent, temp 1)
  3) search vs search: big@SIMS vs champion@SIMS, each side's Gumbel search
     running on its own net (mixed architectures)

  python scripts/go_eval_big.py --big <ckpt> [--games 512] [--sims 16]
"""

import argparse
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import yaml

CHAMPION = 'runs/go9_league_srch_24-17-52-19/nn/last_go9_league_srch_ep_15000_rew_0.55594206.pth'
BIG_ARCH = dict(blocks=18, channels=384, gpool_every=3, value_units=256,
                block_type='nbt', bottleneck_channels=192)
SMALL_ARCH = dict(blocks=6, channels=64, gpool_every=2, value_units=128)


def clean_sd(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    out = {}
    for k, v in sd.items():
        for pre in ('_orig_mod.', 'a2c_network.'):
            k = k.replace(pre, '')
        out[k] = v
    return out


def flax_params(ckpt_path, arch):
    from rl_games.envs.go_flax import params_from_torch
    return params_from_torch(clean_sd(ckpt_path), **arch)


def torch_net(ckpt_path, arch):
    from rl_games.envs.go_network import GoResNetBuilder
    b = GoResNetBuilder()
    b.load(dict(arch))
    net = b.build('go_resnet', input_shape=(9, 9, 17), actions_num=82).cuda().eval()
    sd = clean_sd(ckpt_path)
    net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()},
                        strict=False)
    return net


def play(env, act_fn, games):
    obs = env.reset()
    wins = n = 0
    while n < games:
        mask = env.get_action_masks()
        a = act_fn(obs, mask, env)
        obs, _, d, info = env.step(a)
        db = d.bool()
        if db.any():
            wins += int((info['win'][db] > 0).sum().item())
            n += int(db.sum().item())
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--big', required=True)
    ap.add_argument('--games', type=int, default=512)
    ap.add_argument('--sims', type=int, default=16)
    ap.add_argument('--skip-baseline', action='store_true')
    args = ap.parse_args()

    import jax
    from rl_games.envs.pgx_go import PgxGoVecEnv
    from rl_games.envs.go_anchors import make_baseline_opponent
    from rl_games.envs.go_flax import make_flax_opponent
    from rl_games.envs.go_search import make_search_policy, make_search_opponent
    from pgx import go

    net = torch_net(args.big, BIG_ARCH)

    def raw_sampled(obs, mask, env):
        with torch.no_grad():
            logits, _, _ = net({'obs': obs})
        logits[~mask] = -torch.inf
        return torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)

    # 1) vs AlphaZero baseline
    if not args.skip_baseline:
        env = PgxGoVecEnv('pgx_go', args.games, komi=7.0, symmetry=False,
                          seed=2468, opponent=make_baseline_opponent('baselines'))
        wr = play(env, raw_sampled, args.games)
        print(f'RESULT big raw vs AZ-baseline: {wr:.3f}', flush=True)
        del env

    # 2) raw vs raw against the champion (flax opponent, temp 1)
    champ_flax = flax_params(CHAMPION, SMALL_ARCH)
    opp_fn, _ = make_flax_opponent(temperature=1.0, **SMALL_ARCH)
    env = PgxGoVecEnv('pgx_go', args.games, komi=7.0, symmetry=False, seed=1357,
                      opponent=opp_fn, opponent_params=champ_flax)
    wr = play(env, raw_sampled, args.games)
    print(f'RESULT big raw vs champion raw: {wr:.3f}', flush=True)
    del env

    # 3) search vs search, each side on its own net
    genv = go.Go(size=9, komi=7.0)
    big_flax = flax_params(args.big, BIG_ARCH)
    big_search = make_search_policy(genv, num_simulations=args.sims, **BIG_ARCH)
    champ_opp = make_search_opponent(genv, num_simulations=args.sims, **SMALL_ARCH)
    env = PgxGoVecEnv('pgx_go', min(args.games, 256), komi=7.0, symmetry=False,
                      seed=9753, opponent=champ_opp, opponent_params=champ_flax)
    rng = jax.random.PRNGKey(4)

    def big_search_act(obs, mask, env):
        nonlocal rng
        rng, k = jax.random.split(rng)
        a_env, _ = big_search(big_flax, env.get_jax_state(), k)
        return env.env_actions_to_sym(a_env).long()

    wr = play(env, big_search_act, min(args.games, 256))
    print(f'RESULT big@{args.sims} vs champion@{args.sims}: {wr:.3f}', flush=True)
    print('EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
