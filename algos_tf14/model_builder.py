import common.object_factory as object_factory
import algos_tf14
import algos_tf14.network_builder as network_builder
import algos_tf14.models as models
class ModelBuilder:
    def __init__(self):

        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.register_builder('discrete_a2c', lambda network, **kwargs : models.ModelA2C(network))
        self.model_factory.register_builder('discrete_a2c_lstm', lambda network, **kwargs : models.LSTMModelA2C(network))
        self.model_factory.register_builder('continuous_a2c', lambda network, **kwargs : models.ModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd', lambda network, **kwargs : models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('continuous_a2c_lstm', lambda network, **kwargs : models.LSTMModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_lstm_logstd', lambda network, **kwargs : models.LSTMModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('dqn', lambda network, **kwargs : models.AtariDQN(network))
        self.model_factory.register_builder('vdn', lambda network, **kwargs : models.VDN_DQN(network))
        self.model_factory.register_builder('iql', lambda network, **kwargs : models.IQL_DQN(network))


        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.register_builder('actor_critic', lambda **kwargs : network_builder.A2CBuilder())
        self.network_factory.register_builder('dqn', lambda **kwargs : network_builder.DQNBuilder())

    def load(self, params):
        self.model_name = params['model']['name']
        self.network_name = params['network']['name']

        network = self.network_factory.create(self.network_name)
        network.load(params['network'])
        model = self.model_factory.create(self.model_name, network=network)

        return model


