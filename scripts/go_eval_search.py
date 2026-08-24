#!/usr/bin/env python
"""Search-vs-raw evaluation (plan Phase 2 / 3.3).

Plays a checkpoint WITH Gumbel search (learner side) against the SAME
checkpoint raw (as the in-env flax opponent), for each simulation budget.
sims=0 must reproduce raw-vs-raw (~50%); the win rate should then rise
monotonically with sims — the flattening point is the value head's ceiling.

    python scripts/go_eval_search.py -f rl_games/configs/go/ppo_go9_selfplay.yaml \
        -c runs/.../nn/last_....pth --sims 0 8 16 32 64 --games 256
"""

import argparse
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml


def run_eval(cfg, ckpt, sims, games, temperature):
    import torch
    from rl_games.algos_torch.go_player import GoPlayer

    params = cfg['params']
    params['config']['num_actors'] = games
    env_cfg = params['config']['env_config']
    env_cfg['symmetry'] = False
    env_cfg['opponent'] = 'net'
    for k in ('blocks', 'channels', 'gpool_every', 'value_units'):
        env_cfg[k] = params['network'].get(k, {'blocks': 6, 'channels': 64,
                                              'gpool_every': 2, 'value_units': 128}[k])
    params['config']['player'] = {
        'use_vecenv': True,
        'search': {'enabled': sims > 0, 'num_simulations': sims,
                   'temperature': temperature},
    }

    player = GoPlayer(params)
    player.restore(ckpt)
    player.has_batch_dimension = True
    env = player.env
    # raw copy of the same checkpoint as the opponent
    env.update_weights({'model': player.model.state_dict()})

    obs = env.reset()
    wins, n = 0.0, 0
    while n < games:
        mask = env.get_action_masks()
        actions = player.get_masked_action(obs, mask, True)
        obs, _, done, info = env.step(actions)
        d = done.bool()
        if d.any():
            wins += (info['win'][d] > 0).float().sum().item()
            n += int(d.sum().item())
    env_close = getattr(env, 'close', None)
    if env_close:
        env_close()
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('--sims', type=int, nargs='+', default=[0, 8, 16, 32])
    ap.add_argument('--games', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0)
    args = ap.parse_args()

    with open(args.file) as f:
        cfg = yaml.safe_load(f)

    print(f'checkpoint: {args.checkpoint}')
    for sims in args.sims:
        import copy
        wr = run_eval(copy.deepcopy(cfg), args.checkpoint, sims,
                      args.games, args.temperature)
        print(f'sims={sims:4d}  winrate vs raw self: {wr:.3f}')


if __name__ == '__main__':
    main()
