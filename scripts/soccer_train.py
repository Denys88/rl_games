#!/usr/bin/env python
"""Train envpool soccer with the soccer observers attached.

Usage:
    python scripts/soccer_train.py -f rl_games/configs/envpool/ppo_soccer_boxhead_league.yaml
    python scripts/soccer_train.py -f ... -c runs/.../nn/last_....pth   # resume
    python scripts/soccer_train.py -f ... --max-epochs 2000 --name probe

The league observer is used whenever env_config.opponent == 'pool' (or
--league); otherwise plain match statistics are logged (opponent: random).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='yaml config')
    ap.add_argument('-c', '--checkpoint', default=None)
    ap.add_argument('--play', action='store_true', help='play instead of train')
    ap.add_argument('--max-epochs', type=int, default=0)
    ap.add_argument('--name', default=None, help='override run name')
    ap.add_argument('--num-actors', type=int, default=0)
    ap.add_argument('--league', action='store_true',
                    help='force SoccerLeagueObserver (env_config opponent: pool)')
    ap.add_argument('--seed-pool', nargs='*', default=None,
                    help='checkpoints preloaded into the league pool (resume)')
    args = ap.parse_args()

    with open(args.file) as f:
        cfg = yaml.safe_load(f)
    conf = cfg['params']['config']
    if args.max_epochs > 0:
        conf['max_epochs'] = args.max_epochs
    if args.name:
        conf['name'] = args.name
    if args.num_actors > 0:
        conf['num_actors'] = args.num_actors

    from rl_games.torch_runner import Runner
    from rl_games.common.soccer_observer import SoccerObserver, SoccerLeagueObserver

    if args.league or conf.get('env_config', {}).get('opponent') == 'pool':
        observer = SoccerLeagueObserver(league_config=conf.get('league', {}),
                                        seed_pool=args.seed_pool,
                                        anneal_config=conf.get('shaping_anneal'))
    else:
        observer = SoccerObserver(anneal_config=conf.get('shaping_anneal'))
    runner = Runner(observer)
    runner.load(cfg)
    runner.run({'train': not args.play, 'play': args.play,
                'checkpoint': args.checkpoint})


if __name__ == '__main__':
    main()
