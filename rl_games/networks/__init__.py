from rl_games.networks.tcnn_mlp import TcnnNetBuilder
from rl_games.networks.vision_networks import VisionImpalaBuilder, VisionBackboneBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)
model_builder.register_network('vision_actor_critic', VisionImpalaBuilder)
model_builder.register_network('e2e_vision_actor_critic', VisionBackboneBuilder)