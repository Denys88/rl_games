import algos_torch
import numpy as np
import common.object_factory
import common.env_configurations as env_configurations

import algos_torch.network_builder as network_builder
import algos_torch.model_builder as model_builder
import algos_torch.a2c_continuous as a2c_continuous
import algos_torch.a2c_discrete as a2c_discrete
#import algos_torch.dqnagent as dqnagent

import common.tr_helpers as tr_helpers
import yaml
import ray
#import algos_tf14.players
import argparse
import common.experiment as experiment
import copy
import torch

class Runner:
    def __init__(self):
        self.algo_factory = common.object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = common.object_factory.ObjectFactory()
        #self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        #self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs)) 
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            self.load_path = params['load_path']

        self.model = self.model_builder.load(params)
        self.config = copy.deepcopy(params['config'])
        
        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**self.config['reward_shaper'])
        self.config['network'] = self.model

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(copy.deepcopy(self.default_config))

        if 'experiment_config' in yaml_conf:
            self.exp_config = yaml_conf['experiment_config']

    def get_prebuilt_config(self):
        return self.config

    def run_train(self):
        print('Started to train')
        ray.init(redis_max_memory=1024*1024*1000, object_store_memory=1024*1024*1000)
        obs_space, action_space = env_configurations.get_obs_and_action_spaces_from_config(self.config)
        if len(obs_space.shape) == 3:
            shape = list(obs_space.shape)
            obs_space.shape = [shape[-1]] + shape[:-1]
        print('obs_space:', obs_space)
        print('action_space:', action_space)
        if self.exp_config:
            self.experiment =  experiment.Experiment(self.default_config, self.exp_config)
            exp_num = 0
            exp = self.experiment.get_next_config()
            while exp is not None:
                exp_num += 1
                print('Starting experiment number: ' + str(exp_num))
                self.reset()
                self.load_config(exp)
                agent = self.algo_factory.create(self.algo_name, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)  
                self.experiment.set_results(*agent.train())
                exp = self.experiment.get_next_config()
        else:
            self.reset()
            self.load_config(self.default_config)
            agent = self.algo_factory.create(self.algo_name, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)  
            if self.load_check_point or (self.load_path is not None):
                agent.restore(self.load_path)
            agent.train()
            
    def create_player(self):
        return self.player_factory.create(self.algo_name, config=self.config)

    def create_agent(self, obs_space, action_space):
        return self.algo_factory.create(self.algo_name, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)

    def run(self, args):
        if 'checkpoint' in args:
            self.load_path = args['checkpoint']

        if args['train']:
            self.run_train()
        elif args['play']:
            print('Started to play')

            player = self.player_factory.create(self.algo_name, config=self.config)
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
    
    runner.reset()
    runner.run(args)
        
