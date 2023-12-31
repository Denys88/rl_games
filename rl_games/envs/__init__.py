

from rl_games.envs.test_network import TestNetBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('testnet', TestNetBuilder)

import gym

gym.envs.register(
     id='MarioEnv-v0',
     entry_point='rl_games.envs.mario:MarioEnv'
)