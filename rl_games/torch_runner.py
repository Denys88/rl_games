import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent

# Limit tensor printouts to 3 decimal places globally
torch.set_printoptions(precision=3, sci_mode=False)


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        if args['train'] and args.get('load_critic_only', False):
            if not getattr(agent, 'has_central_value', False):
                raise ValueError('Loading critic only works only for asymmetric actor critic')
            agent.restore_central_value_function(args['checkpoint'])
            return
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Cannot set new sigma because fixed_sigma is False')


class Runner:
    """Runs training/inference (playing) procedures as per a given configuration.

    The Runner class provides a high-level API for instantiating agents for either training or playing
    with an RL algorithm. It further logs training metrics.

    """

    def __init__(self, algo_observer=None):
        """Initialise the runner instance with algorithms and observers.

        Initialises runners and players for all algorithms available in the library using `rl_games.common.object_factory.ObjectFactory`

        Args:
            algo_observer (:obj:`rl_games.common.algo_observer.AlgoObserver`, optional): Algorithm observer that logs training metrics.
                Defaults to `rl_games.common.algo_observer.DefaultAlgoObserver`

        """

        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()

        # Enable TensorFloat32 (TF32) for faster matrix multiplications on NVIDIA GPUs
        # For maximum perfromance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass

    def load_config(self, params):
        """Loads passed config params.

        Args:
            params (:obj:`dict`): Parameters passed in as a dict obtained from a yaml file or some other config format.

        """

        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        """Run the training procedure from the algorithm passed in.

        Args:
            args (:obj:`dict`): Train specific args passed in as a dict obtained from a yaml file or some other config format.

        """
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)

        # Restore weights (if any) BEFORE compiling the model.  Compiling first
        # wraps the model in an `OptimizedModule`, which changes parameter
        # names (adds the `_orig_mod.` prefix) and breaks `load_state_dict`
        # when loading checkpoints that were saved from an *un‑compiled*
        # model.

        _restore(agent, args)
        _override_sigma(agent, args)

        # Now compile the (already restored) model. Doing it after the restore
        # keeps parameter names consistent with the checkpoint.

        # mode="max-autotune" would be faster at runtime, but it has a much
        # longer compilation time. "reduce-overhead" gives a good trade‑off.
        agent.model = torch.compile(agent.model, mode="reduce-overhead")

        agent.train()

    def run_play(self, args):
        """Run the inference procedure from the algorithm passed in.

        Args:
            args (:obj:`dict`): Playing specific args passed in as a dict obtained from a yaml file or some other config format.

        """
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        """Run either train/play depending on the args.

        Args:
            args (:obj:`dict`):  Args passed in as a dict obtained from a yaml file or some other config format.

        """
        if args['train']:
            self.run_train(args)
        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)
