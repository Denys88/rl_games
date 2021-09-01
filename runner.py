import numpy as np
import argparse, copy, os, yaml
import ray, signal

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#import warnings
#warnings.filterwarnings("error")
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
    ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
    ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
    ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
    ap.add_argument("-f", "--file", required=True, help="path to config")
    ap.add_argument("-na", "--num_actors", type=int, default=0, required=False, 
                    help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")

    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(ap.parse_args())
    config_name = args['file']
    
    print('Loading config: ', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)

        if args['num_actors'] > 0:
            config['params']['config']['num_actors'] = args['num_actors']

        if args['tf']:
            from rl_games.tf14_runner import Runner
        else:
            from rl_games.torch_runner import Runner

        ray.init(object_store_memory=1024*1024*1000)
        #signal.signal(signal.SIGINT, exit_gracefully)

        runner = Runner()
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    runner.reset()
    runner.run(args)

    ray.shutdown()
        
