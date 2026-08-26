#!/usr/bin/env python
"""b18c384nbt from-scratch league training, two phases:

  A) plain league (opponent: pool) for --phase-a epochs
  B) search-strengthened league (opponent: pool_search, gumbel@--sims)
     continuing from A's last checkpoint to --phase-a + --phase-b epochs
"""

import argparse
import glob
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml


def run_phase(cfg, name, max_epochs, checkpoint=None, search_sims=0):
    c = cfg['params']['config']
    c['name'] = name
    c['max_epochs'] = max_epochs
    if search_sims > 0:
        c['env_config']['opponent'] = 'pool_search'
        c['env_config']['opponent_search_sims'] = search_sims
        # entropy anneal finished in phase A — pin it low
        c['lr_schedule'] = 'None'
        c['schedule_entropy'] = False
        c['entropy_coef'] = 0.003
    from rl_games.torch_runner import Runner
    from rl_games.common.go_observer import GoLeagueObserver
    observer = GoLeagueObserver(league_config=c.get('league', {}),
                                eval_every=100, eval_games=256)
    runner = Runner(observer)
    runner.load(cfg)
    runner.run({'train': True, 'play': False, 'checkpoint': checkpoint})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase-a', type=int, default=3200)
    ap.add_argument('--phase-b', type=int, default=800)
    ap.add_argument('--sims', type=int, default=8)
    ap.add_argument('--name', default='go9_big')
    ap.add_argument('--skip-a', default=None,
                    help='checkpoint to resume phase B from (skips phase A)')
    args = ap.parse_args()

    with open('rl_games/configs/go/ppo_go9_big.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.skip_a:
        ckpt = args.skip_a
    else:
        run_phase(cfg, args.name + '_a', args.phase_a)
        cands = sorted(glob.glob(f'runs/{args.name}_a_*/nn/last_*.pth'),
                       key=os.path.getmtime)
        ckpt = cands[-1]
        print(f'[big] phase A done -> {ckpt}', flush=True)
        with open('rl_games/configs/go/ppo_go9_big.yaml') as f:
            cfg = yaml.safe_load(f)

    run_phase(cfg, args.name + '_b', args.phase_a + args.phase_b,
              checkpoint=ckpt, search_sims=args.sims)
    print('[big] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
