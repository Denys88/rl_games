#!/usr/bin/env python
"""Train Craftax with the CraftaxObserver attached (craftax/score, achievement rates).

    python scripts/craftax_train.py -f rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml
    python scripts/craftax_train.py -f ... --max-epochs 500 --name probe --train-dir runs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('-c', '--checkpoint', default=None)
    ap.add_argument('--max-epochs', type=int, default=0)
    ap.add_argument('--name', default=None)
    ap.add_argument('--train-dir', default=None)
    ap.add_argument('--num-actors', type=int, default=0)
    args = ap.parse_args()

    with open(args.file) as f:
        cfg = yaml.safe_load(f)
    conf = cfg['params']['config']
    if args.max_epochs > 0:
        conf['max_epochs'] = args.max_epochs
    if args.name:
        conf['name'] = args.name
    if args.train_dir:
        conf['train_dir'] = args.train_dir
    if args.num_actors > 0:
        conf['num_actors'] = args.num_actors

    from rl_games.torch_runner import Runner
    from rl_games.envs.craftax_vecenv import CraftaxObserver
    runner = Runner(CraftaxObserver())
    runner.load(cfg)
    runner.run({'train': True, 'play': False, 'checkpoint': args.checkpoint})


if __name__ == '__main__':
    main()
