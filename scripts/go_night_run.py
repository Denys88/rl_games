#!/usr/bin/env python
"""Overnight: continue league training 3000 -> 12000 epochs, then run the
main-exploiter acceptance test against the final checkpoint."""

import glob
import os
import subprocess
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml

RESUME = 'runs/go9_league_v1_23-20-47-43/nn/last_go9_league_v1_ep_3000_rew_0.6177974.pth'
NAME = 'go9_league_night'
MAX_EPOCHS = 12000

with open('rl_games/configs/go/ppo_go9_league.yaml') as f:
    cfg = yaml.safe_load(f)
c = cfg['params']['config']
c['name'] = NAME
c['max_epochs'] = MAX_EPOCHS
# entropy anneal already finished in the first 3000 epochs — pin it low
# instead of letting a restarted schedule bounce it back up
c['lr_schedule'] = 'None'
c['schedule_entropy'] = False
c['entropy_coef'] = 0.003
c['save_frequency'] = 500

from rl_games.torch_runner import Runner
from rl_games.common.go_observer import GoLeagueObserver

observer = GoLeagueObserver(league_config=c.get('league', {}),
                            eval_every=200, eval_games=256)
runner = Runner(observer)
runner.load(cfg)
runner.run({'train': True, 'play': False, 'checkpoint': RESUME})

print('[night] league training done, launching exploiter', flush=True)
finals = sorted(glob.glob(f'runs/{NAME}_*/nn/last_{NAME}_ep_*.pth'),
                key=os.path.getmtime)
target = finals[-1]
print(f'[night] exploiter target: {target}', flush=True)
subprocess.run([sys.executable, 'scripts/go_exploiter.py',
                '-f', 'rl_games/configs/go/ppo_go9_selfplay.yaml',
                '--target', target, '--epochs', '200',
                '--name', 'go9_night_exploiter'], check=False)
print('[night] ALL DONE', flush=True)
