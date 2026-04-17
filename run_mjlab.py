"""Run MJLab environments with rl_games PPO."""

import argparse
import yaml
import warp as wp
wp.init()

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='rl_games/configs/mjlab/ppo_g1_velocity.yaml')
    args = parser.parse_args()

    # Register mjlab env
    env_configurations.register('mjlab_g1_velocity', {
        'vecenv_type': 'MJLAB',
    })

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({'train': True})


if __name__ == '__main__':
    main()
