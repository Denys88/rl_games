"""Run Warp GPU CartPole with rl_games (PPO discrete, PPO continuous, SAC).

Usage:
    python run_warp_cartpole.py --config rl_games/configs/warp/ppo_warp_cartpole.yaml
    python run_warp_cartpole.py --config rl_games/configs/warp/ppo_warp_cartpole_continuous.yaml
    python run_warp_cartpole.py --config rl_games/configs/warp/sac_warp_cartpole.yaml
"""

import argparse
import yaml
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.envs.warp_cartpole import WarpCartPole


def register_warp_envs():
    """Register Warp CartPole environments with rl_games."""

    # Discrete CartPole
    env_configurations.register('warp_cartpole', {
        'env_creator': lambda **kwargs: WarpCartPole(continuous=False, **kwargs),
        'vecenv_type': 'WARP',
    })

    # Continuous CartPole
    env_configurations.register('warp_cartpole_continuous', {
        'env_creator': lambda **kwargs: WarpCartPole(continuous=True, **kwargs),
        'vecenv_type': 'WARP',
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='rl_games/configs/warp/ppo_warp_cartpole.yaml')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    register_warp_envs()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    runner = Runner()
    runner.load(config)
    runner.reset()

    if args.play:
        runner.run({'train': False, 'play': True, 'checkpoint': args.checkpoint})
    else:
        runner.run({'train': True})


if __name__ == '__main__':
    main()
