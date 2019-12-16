import object_factory
import network_builder
import model_builder
import a2c_continuous
import a2c_discrete
import dqnagent
import tr_helpers


class Runner:
    def __init__(self):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : return a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : return a2c_discrete.A2CAgent(**kwargs)) 
        self.algo_factory.register_builder('dqn', lambda **kwargs : return dqnagent.DQNAgent(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()
    def load(self, params):
        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        if self.load_check_point:
            self.load_path = params['load_path']

        self.model_builder.load(params[''])
        self.model = self.model_builder.build()
        self.config = params['config']
        
        self.config['REWARD_SHAPER'] = tr_helpers.DefaultRewardsShaper(scale_value = self.config['reward_scale_value'], shift_value = self.config['reward_shift_value']),

        obs_space, action_space = env_configurations.get_obs_and_action_spaces(env_name)
        






if __name__ == '__main__':
    import sys
    arguments = len(sys.argv) - 1
    assert arguments > 0

    config_name = sys.argv[0]
        
