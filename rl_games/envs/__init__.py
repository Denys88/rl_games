

from rl_games.envs.test_network import TestNetBuilder, TestNetAuxLossBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('testnet', TestNetBuilder)
model_builder.register_network('testnet_aux_loss', TestNetAuxLossBuilder)