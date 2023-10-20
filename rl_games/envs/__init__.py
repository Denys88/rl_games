

from rl_games.envs.test_network import TestNetBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('testnet', TestNetBuilder)