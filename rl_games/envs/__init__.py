

from rl_games.envs.connect4_network import ConnectBuilder
from rl_games.algos_torch import model_builder

print('register custom network')
model_builder.register_network('connect4net', ConnectBuilder)