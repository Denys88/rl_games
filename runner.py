import tensorflow as tf
import object_factory
import env_configurations
import network_builder
import model_builder
import a2c_continuous
import a2c_discrete
import dqnagent
import tr_helpers
import yaml
import ray

class Runner:
    def __init__(self):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.A2CAgent(**kwargs)) 
        self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()
    def load(self, params):
        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        if self.load_check_point:
            self.load_path = params['load_path']

        self.model = self.model_builder.load(params)
        self.config = params['config']
        
        self.config['REWARD_SHAPER'] = tr_helpers.DefaultRewardsShaper(scale_value = self.config['REWARD_SCALE'], shift_value = self.config['REWARD_SHIFT'])
        self.config['NETWORK'] = self.model

    def run(self):
        ray.init(redis_max_memory=1024*1024*100, object_store_memory=1024*1024*100)
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)

        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.InteractiveSession(config=config)
        obs_space, action_space = env_configurations.get_obs_and_action_spaces(self.config['ENV_NAME'])
        agent = self.algo_factory.create(self.algo_name, sess=sess, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)
        if self.load_check_point:
            agent.restore(self.load_path)

        agent.train()
        ray.shutdown()

if __name__ == '__main__':
    import sys
    arguments = len(sys.argv) - 1
    if arguments == 0:
        config_name = 'configs/example_ppo_continuous.yaml'
    else:
        config_name = sys.argv[1]

    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)
    runner.run()
        
