from rl_games.networks.tcnn_mlp import TcnnNetBuilder
from rl_games.networks.ig_networks import EncoderMLPBuilder, TransformerBuilder, TorchTransformerBuilder
from rl_games.algos_torch import model_builder


model_builder.register_network('tcnnnet', TcnnNetBuilder)
model_builder.register_network('enc_mlp', lambda **kwargs : EncoderMLPBuilder())
model_builder.register_network('transformer', lambda **kwargs : TransformerBuilder())
model_builder.register_network('torch_transformer', lambda **kwargs : TorchTransformerBuilder())