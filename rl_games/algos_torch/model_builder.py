from rl_games.common import object_factory
import rl_games.algos_torch
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import models

NETWORK_REGISTRY = {}
MODEL_REGISTRY = {}

MODEL_V2_REGISTRY = {}

def register_network(name, target_class):
    NETWORK_REGISTRY[name] = lambda **kwargs: target_class(**kwargs)

def register_model(name, target_class):
    MODEL_REGISTRY[name] = lambda  network, **kwargs: target_class(network)


def register_model(name, target_class):
    MODEL_REGISTRY[name] = lambda  network_config, env_config, **kwargs: target_class(network_config, env_config)


class NetworkBuilder:
    def __init__(self):
        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.set_builders(NETWORK_REGISTRY)
        self.network_factory.register_builder('actor_critic', lambda **kwargs: network_builder.A2CBuilder(**kwargs))
        self.network_factory.register_builder('resnet_actor_critic',
                                              lambda **kwargs: network_builder.A2CResnetBuilder(**kwargs))
        self.network_factory.register_builder('soft_actor_critic', lambda **kwargs: network_builder.SACBuilder(**kwargs))

    def load(self, params):
        network_name = params['name']
        network = self.network_factory.create(network_name)
        network.load(params)

        return network


class ModelBuilder:
    def __init__(self):
        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.set_builders(MODEL_REGISTRY)
        self.model_factory.register_builder('discrete_a2c', lambda network, **kwargs: models.ModelA2C(network))
        self.model_factory.register_builder('multi_discrete_a2c',
                                            lambda network, **kwargs: models.ModelA2CMultiDiscrete(network))
        self.model_factory.register_builder('continuous_a2c',
                                            lambda network, **kwargs: models.ModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd',
                                            lambda network, **kwargs: models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('soft_actor_critic',
                                            lambda network, **kwargs: models.ModelSACContinuous(network))
        self.model_factory.register_builder('central_value',
                                            lambda network, **kwargs: models.ModelCentralValue(network))
        self.network_builder = NetworkBuilder()

        self.model__v2_factory = object_factory.ObjectFactory()
        self.model_factory.set_builders(MODEL_V2_REGISTRY)

    def get_network_builder(self):
        return self.network_builder

    def load(self, params):
        model_name = params['model']['name']
        network = self.network_builder.load(params['network'])
        model = self.model_factory.create(model_name, network=network)
        return model

    def load_and_build(self, params, env_config):
        model_name = params['model']['name']
        if self.model_factory.find(model_name):
            network = self.network_builder.load(params['network'])
            model = self.model_factory.create(model_name, network=network)
        else:

        return model.build(params, env_config)