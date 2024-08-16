from rl_games.networks.tcnn_mlp import TcnnNetBuilder
#from rl_games.networks.vision_networks import A2CVisionBuilder, A2CVisionBackboneBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)
# model_builder.register_network('vision_actor_critic', A2CVisionBuilder)
# model_builder.register_network('e2e_vision_actor_critic', A2CVisionBackboneBuilder)