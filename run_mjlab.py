"""Run MJLab environments with rl_games PPO."""

import argparse
import yaml
import warp as wp
wp.init()

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='rl_games/configs/mjlab/ppo_g1_velocity.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Register the env name the config asks for
    env_configurations.register(config['params']['config']['env_name'], {
        'vecenv_type': 'MJLAB',
    })

    runner = Runner(algo_observer=IsaacAlgoObserver())
    runner.load(config)
    runner.reset()
    runner.run({'train': True})


if __name__ == '__main__':
    main()
