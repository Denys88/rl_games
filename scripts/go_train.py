#!/usr/bin/env python
"""Train Go 9x9 with the GoObserver attached (anchor evals + win-rate logging).

Usage:
    python scripts/go_train.py -f rl_games/configs/go/ppo_go9_selfplay.yaml
    python scripts/go_train.py -f ... -c runs/.../nn/last_....pth   # resume
    python scripts/go_train.py -f ... --max-epochs 2000 --eval-every 100
"""

import argparse
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='yaml config')
    ap.add_argument('-c', '--checkpoint', default=None)
    ap.add_argument('--play', action='store_true', help='play instead of train')
    ap.add_argument('--max-epochs', type=int, default=0)
    ap.add_argument('--name', default=None, help='override run name')
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--eval-games', type=int, default=512)
    ap.add_argument('--no-baseline', action='store_true')
    ap.add_argument('--league', action='store_true',
                    help='use GoLeagueObserver (env_config opponent: pool)')
    args = ap.parse_args()

    with open(args.file) as f:
        cfg = yaml.safe_load(f)
    if args.max_epochs > 0:
        cfg['params']['config']['max_epochs'] = args.max_epochs
    if args.name:
        cfg['params']['config']['name'] = args.name

    from rl_games.torch_runner import Runner
    from rl_games.common.go_observer import GoObserver, GoLeagueObserver

    common = dict(eval_every=args.eval_every, eval_games=args.eval_games,
                  use_baseline=not args.no_baseline)
    if args.league or cfg['params']['config'].get('env_config', {}).get('opponent') == 'pool':
        observer = GoLeagueObserver(
            league_config=cfg['params']['config'].get('league', {}), **common)
    else:
        observer = GoObserver(**common)
    runner = Runner(observer)
    runner.load(cfg)
    runner.run({'train': not args.play, 'play': args.play,
                'checkpoint': args.checkpoint})


if __name__ == '__main__':
    main()
