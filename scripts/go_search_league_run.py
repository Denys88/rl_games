#!/usr/bin/env python
"""Continue league training with search-strengthened opponents (Gumbel @8
sims for every pool member), resuming from the overnight ep12000 net."""

import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml

RESUME = 'runs/go9_league_night_23-22-53-53/nn/last_go9_league_night_ep_12000_rew_0.70798427.pth'
NAME = 'go9_league_srch'
MAX_EPOCHS = 16000

with open('rl_games/configs/go/ppo_go9_league.yaml') as f:
    cfg = yaml.safe_load(f)
c = cfg['params']['config']
c['name'] = NAME
c['max_epochs'] = MAX_EPOCHS
c['lr_schedule'] = 'None'
c['schedule_entropy'] = False
c['entropy_coef'] = 0.003
c['save_frequency'] = 500
c['env_config']['opponent'] = 'pool_search'
c['env_config']['opponent_search_sims'] = 8

from rl_games.torch_runner import Runner
from rl_games.common.go_observer import GoLeagueObserver

observer = GoLeagueObserver(league_config=c.get('league', {}),
                            eval_every=200, eval_games=256)
runner = Runner(observer)
runner.load(cfg)
runner.run({'train': True, 'play': False, 'checkpoint': RESUME})
print('[srch] DONE', flush=True)
