import numpy as np
import argparse
import copy
import yaml
import ray
import signal
from rl_games.distributed.dd_ppo import DDPpoRunner

def exit_gracefully(signum, frame):
    ray.shutdown()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-d", "--d_file", required=True, help="path to distributed config")
    args = vars(ap.parse_args())
    config_name = args['file']
    dd_config_name = args['d_file']
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        ppo_config = yaml.safe_load(stream)
    with open(dd_config_name, 'r') as stream:
        dd_config = yaml.safe_load(stream)

    ray.init(num_gpus=4)
    signal.signal(signal.SIGINT, exit_gracefully)

    runner = DDPpoRunner(ppo_config, dd_config)
    runner.init_workers()
    runner.train()
    ray.shutdown()