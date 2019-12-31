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
import players
import argparse

class Runner:
    def __init__(self):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.A2CAgent(**kwargs)) 
        self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs)) 
        self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

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
        
        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(scale_value = self.config['reward_scale'], shift_value = self.config['reward_shift'])
        self.config['network'] = self.model

    def get_prebuilt_config(self):
        return self.config

    def run(self, args):
        ray.init(redis_max_memory=1024*1024*100, object_store_memory=1024*1024*100)
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)

        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.InteractiveSession(config=config)

        if 'checkpoint' in args:
            self.load_path = args['checkpoint']

        if args['train']:
            print('Started to train')
            obs_space, action_space = env_configurations.get_obs_and_action_spaces(self.config['env_name'])
            agent = self.algo_factory.create(self.algo_name, sess=sess, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)

            if self.load_check_point or (self.load_path is not None):
                agent.restore(self.load_path)

            agent.train()
        elif args['play']:
            print('Started to play')

            player = self.player_factory.create(self.algo_name, sess=sess, config=self.config)
            player.restore(self.load_path)
            player.run()
        
        ray.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")

    args = vars(ap.parse_args())
    config_name = args['file']
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)
    runner.run(args)
        
