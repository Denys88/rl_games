import tensorflow as tf
import algos_tf14
import numpy as np
import common.object_factory
import common.env_configurations as env_configurations

import algos_tf14.network_builder as network_builder
import algos_tf14.model_builder as model_builder
import algos_tf14.a2c_continuous as a2c_continuous
import algos_tf14.a2c_discrete as a2c_discrete
import algos_tf14.dqnagent as dqnagent

import common.tr_helpers as tr_helpers
import yaml
import ray
import algos_tf14.players
import argparse
import common.experiment as experiment
import copy

import torch

from sacred import Experiment
import numpy as np
import os
import collections
from os.path import dirname, abspath
import pymongo
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from utils.logging import get_logger, Logger

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

mongo_client = None

class Runner:
    def __init__(self, logger):
        self.algo_factory = common.object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.A2CAgent(**kwargs)) 
        self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = common.object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs)) 
        self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()
        self.sess = None

        self.logger = logger

    def reset(self):
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)

        config = tf.ConfigProto(gpu_options=gpu_options)
        tf.reset_default_graph()
        if self.sess:
            self.sess.close()
        self.sess = tf.InteractiveSession(config=config)

    def load_config(self, params):
        self.seed = params.get('seed', None)

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        self.exp_config = None

        if self.seed:
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            self.load_path = params['load_path']
        else:
            self.load_path = None

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
        # print('Started to train')
        self.logger.console_logger.info('Started to train')
        ray.init(redis_max_memory=1024*1024*1000, object_store_memory=1024*1024*1000)
        obs_space, action_space = env_configurations.get_obs_and_action_spaces_from_config(self.config)
        # print('obs_space:', obs_space)
        # print('action_space:', action_space)
        self.logger.console_logger.info('obs_space: {}'.format(obs_space))
        self.logger.console_logger.info('action_space: {}'.format(action_space))
        if self.exp_config:
            self.experiment =  experiment.Experiment(self.default_config, self.exp_config)
            exp_num = 0
            exp = self.experiment.get_next_config()
            while exp is not None:
                exp_num += 1
                # print('Starting experiment number: ' + str(exp_num))
                self.logger.console_logger.info('Starting experiment number: ' + str(exp_num))
                self.reset()
                self.load_config(exp)
                agent = self.algo_factory.create(self.algo_name, sess=self.sess, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config, logger=self.logger)
                self.experiment.set_results(*agent.train())
                exp = self.experiment.get_next_config()
        else:
            self.reset()
            self.load_config(self.default_config)
            agent = self.algo_factory.create(self.algo_name, sess=self.sess, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config, logger=self.logger)
            if self.load_check_point or (self.load_path is not None):
                agent.restore(self.load_path)
            agent.train()
            
    def create_player(self):
        return self.player_factory.create(self.algo_name, sess=self.sess, config=self.config)

    def create_agent(self, obs_space, action_space):
        return self.algo_factory.create(self.algo_name, sess=self.sess, base_name='run', observation_space=obs_space, action_space=action_space, config=self.config)

    def run(self, args):
        if 'checkpoint' in args:
            self.load_path = args['checkpoint']

        if args['train']:
            self.run_train()
        elif args['play']:
            # print('Started to play')
            logger.console_logger.info('Started to play')
            player = self.player_factory.create(self.algo_name, sess=self.sess, config=self.config)
            player.restore(self.load_path)
            player.run()
        
        ray.shutdown()

# Function to connect to a mongodb and add a Sacred MongoObserver
def setup_mongodb(db_url, db_name):
    client = None
    mongodb_fail = True

    # Try 5 times to connect to the mongodb
    for tries in range(5):
        # First try to connect to the central server. If that doesn't work then just save locally
        maxSevSelDelay = 10000  # Assume 10s maximum server selection delay
        try:
            # Check whether server is accessible
            logger.info("Trying to connect to mongoDB '{}'".format(db_url))
            client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
            client.server_info()
            # If this hasn't raised an exception, we can add the observer
            ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True))  # db_name=db_name,
            logger.info("Added MongoDB observer on {}.".format(db_url))
            mongodb_fail = False
            break
        except pymongo.errors.ServerSelectionTimeoutError:
            logger.warning("Couldn't connect to MongoDB on try {}".format(tries + 1))

    if mongodb_fail:
        logger.error("Couldn't connect to MongoDB after 5 tries!")
        # TODO: Maybe we want to end the script here sometimes?

    return client

@ex.main
def my_main(_run, _config, _log):
    global mongo_client

    import datetime

    # arglist = parse_args()
    # unique_token = "{}__{}".format(arglist.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # run the framework
    # run(_run, _config, _log, mongo_client, unique_token)

    logger = Logger(_log)

    # configure tensorboard logger
    unique_token = "{}__{}".format(_config["label"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    use_tensorboard = False
    if use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    logger.setup_sacred(_run)

    _log.info("Experiment Parameters:")
    import pprint
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # START THE TRAINING PROCESS
    runner = Runner(logger)
    runner.load(_config)
    runner.reset()
    # args = vars(arglist)
    runner.run(_config)

    # runner.run(args)

    # train(arglist, logger, _config)
    # arglist = convert(_config)
    # train(arglist)

    # force exit
    os._exit(0)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "configs", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-e", "--exp-name", required=True, help="experiment name")
    ap.add_argument("-n", "--no-mongo", required=False, default=False, help="experiment name")
    return ap.parse_args()


if __name__ == '__main__':
    import os

    import sys
    from copy import deepcopy

    params = deepcopy(sys.argv)

    # args = vars(ap.parse_args())
    # config_name = args['file']
    # print('Loading config: ', config_name)
    # with open(config_name, 'r') as stream:
    #     config = yaml.safe_load(stream)
    #     runner = Runner()
    #     try:
    #         runner.load(config)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #
    # # Load algorithm and env base configs
    # #file_config = _get_config(params, "--file", "envs")
    #
    # # Load into official sacred configs
    # if config_name is not None:
    #     with open(os.path.join(os.path.dirname(__file__), "configs", subfolder, "{}.yaml".format(config_name)), "r") as f:
    #         try:
    #             file_config = yaml.load(f)
    #         except yaml.YAMLError as exc:
    #             assert False, "{}.yaml error: {}".format(config_name, exc)

    config_dict = {"train": True,
                   "load_checkpoint": False,
                   "load_path": None}
    file_config = _get_config(params, "--file", "")
    config_dict = recursive_dict_update(config_dict, file_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # arglist = ap.parse_args()

    # from copy import deepcopy
    # ex.add_config({"name":arglist.exp_name})

    # Check if we don't want to save to sacred mongodb
    no_mongodb = False

    for _i, _v in enumerate(params):
        if "no-mongo" in _v:
            if "--no-mongo" == _v:
                del params[_i]
                no_mongodb = True
                break

    config_dict = {"train": True}
    db_config_path = "./db_config.private.yaml"
    with open(db_config_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # If there is no url set for the mongodb, we cannot use it
    if not no_mongodb and "db_url" not in config_dict:
        no_mongodb = True
        logger.error("No 'db_url' to use for Sacred MongoDB")

    if not no_mongodb:
        db_url = config_dict["db_url"]
        db_name = config_dict["db_name"]
        mongo_client = setup_mongodb(db_url, db_name)

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

# if __name__ == '__main__':
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
#     ap.add_argument("-p", "--play", required=False, help="play network", action='store_true')
#     ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
#     ap.add_argument("-f", "--file", required=True, help="path to config")
#
#     args = vars(ap.parse_args())
#     config_name = args['file']
#     print('Loading config: ', config_name)
#     with open(config_name, 'r') as stream:
#         config = yaml.safe_load(stream)
#         runner = Runner()
#         try:
#             runner.load(config)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     main()