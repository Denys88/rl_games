import object_factory
import network_builder
import model_builder
import a2c_continuous
import a2c_discrete
import dqnagent


class Runner:
    def __init__(self):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : return a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : return a2c_discrete.A2CAgent(**kwargs)) 
        self.algo_factory.register_builder('dqn', lambda **kwargs : return dqnagent.A2CAgent(**kwargs))  