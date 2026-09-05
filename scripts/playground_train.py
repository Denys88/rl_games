#!/usr/bin/env python
"""Train a mujoco_playground env with the PlaygroundObserver attached.

    LD_LIBRARY_PATH=/usr/lib/wsl/lib python scripts/playground_train.py -f rl_games/configs/playground/ppo_panda_pick_state.yaml
    ... --max-epochs 100 --name probe

Under WSL2 the CUDA toolkit's stub libcuda shadows the driver's; Warp then
reports no CUDA device. The script re-execs itself with
LD_LIBRARY_PATH=/usr/lib/wsl/lib prepended when that directory exists.
"""
import argparse
import os
import sys

WSL_LIB = '/usr/lib/wsl/lib'
if os.path.isdir(WSL_LIB) and WSL_LIB not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = WSL_LIB + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

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
    from rl_games.envs.playground_vecenv import PlaygroundObserver
    runner = Runner(PlaygroundObserver())
    runner.load(cfg)
    runner.run({'train': True, 'play': False, 'checkpoint': args.checkpoint})


if __name__ == '__main__':
    main()
