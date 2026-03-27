"""Run Newton GPU Ant with rl_games PPO."""

import argparse
import yaml
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations
from rl_games.envs.newton_ant import NewtonAnt


def register_newton_envs():
    env_configurations.register('newton_ant', {
        'env_creator': lambda **kwargs: NewtonAnt(**kwargs),
        'vecenv_type': 'WARP',
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='rl_games/configs/newton/ppo_newton_ant.yaml')
    args = parser.parse_args()

    register_newton_envs()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({'train': True})


if __name__ == '__main__':
    main()
