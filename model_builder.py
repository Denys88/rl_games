import object_factory
import models
import network_builder

class ModelBuilder:
    def __init__(self):

        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.register_builder('discrete_a2c', lambda network, **kwargs : return models.ModelA2C(network))
        self.model_factory.register_builder('discreate_a2c_lstm', lambda network, **kwargs : return models.LSTMModelA2C(network))
        self.model_factory.register_builder('continuous_a2c', lambda network, **kwargs : return models.ModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd', lambda network, **kwargs : return models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('LSTMModelA2CContinuous', lambda network, **kwargs : return models.LSTMModelA2CContinuous(network))

        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.register_builder('actor_critic_continuous', lambda **kwargs : return A2CContinousMlpBuilder)

    def _create_model(self, name, network, **kwargs):
        return self.factory.create(name, network, **kwargs)

    def load(params):
        self.model_name = params['model']['name']
        self.network_name = params['network']['name']

        network_loader = self.network_factory.create(self.network_name)
        network = network_loader.load(params['network'])
        model = self.model_factory.create(self.model_name, network)

        return model


