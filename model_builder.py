import object_factory
import models
import network_builder

class ModelBuilder:
    def __init__(self):
        self.activations_factory = object_factory.ObjectFactory()
        self.activations_factory.register_builder('relu', lambda **kwargs : return tf.nn.relu)
        self.activations_factory.register_builder('tanh', lambda **kwargs : return tf.nn.tanh)
        self.activations_factory.register_builder('sigmoid', lambda **kwargs : return tf.nn.sigmoid)
        self.activations_factory.register_builder('elu', lambda  **kwargs : return tf.nn.elu)
        self.activations_factory.register_builder('selu', lambda **kwargs : return tf.nn.selu)
        self.activations_factory.register_builder('none', lambda **kwargs : return None)


        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.register_builder('ModelA2C', lambda network, **kwargs : return models.ModelA2C(network))
        self.model_factory.register_builder('LSTMModelA2C', lambda network, **kwargs : return models.LSTMModelA2C(network))
        self.model_factory.register_builder('ModelA2CContinuous', lambda network, **kwargs : return models.ModelA2CContinuous(network))
        self.model_factory.register_builder('ModelA2CContinuousLogStd', lambda network, **kwargs : return models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('LSTMModelA2CContinuous', lambda network, **kwargs : return models.LSTMModelA2CContinuous(network))

        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.register_builder('actor_critic_merged', lambda **kwargs : return tf.nn.relu)
        self.network_factory.register_builder('actor_critic_separate', lambda **kwargs : return tf.nn.tanh)
        self.network_factory.register_builder('q_network', lambda **kwargs : return tf.nn.sigmoid)
        self.network_factory.register_builder('ddpg', lambda  **kwargs : return tf.nn.elu)
        self.network_factory.register_builder('soft_actor_critic', lambda **kwargs : return tf.nn.selu)

    def create_model(self, name, network, **kwargs):
        return self.factory.create(name, network, **kwargs)