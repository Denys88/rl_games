# Introduction to [rl_games](https://github.com/Denys88/rl_games/)  - new envs, and new algorithms built on rl_games
This write-up describes some elements of the general functioning of the [rl_games](https://github.com/Denys88/rl_games/) reinforcement learning library. It also provides a guide on extending rl_games with new environments and algorithms using a structure similar to the [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) package. Topics covered in this write-up are
1. The various components of rl_games (runner, algorthms, environments ...)
2. Using rl_games for your own work
    - Adding new gym-like environments to rl_games 
    - Using non-gym environments and simulators with the algorithms in rl_games (refer to [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for examples)
    - Adding new algorithms to rl_games

## General setup in rl_games
rl_games uses the main python script called `runner.py` along with flags for either training (`--train`) or executing policies (`--play`) and a mandatory argument for passing training/playing configurations (`--file`). A basic example of training and then playing for PPO in Pong can be executed with the following. You can also checkout the PPO config file at `rl_games/configs/atari/ppo_pong.yaml`.

```
python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml
python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth
```

rl_games uses the following base classes to define algorithms, instantiate environments, and log metrics.

1. `rl_games.torch_runner.Runner` 
    - Main class that instantiates the algorithm as per the given configuration and executes either training or playing 
    - When instantiated, algorithm instances for all algos in rl_games are automatically added using `rl_games.common.Objectfactory()`'s `register_builder()` method. The same is also done for the player instances for all algos. 
    - Depending on the args given, either `self.run_train()` or `self.run_play()` is executed 
    - The Runner sets up the algorithm observer that logs training metrics. If one is not provided, it automatically uses the `DefaultAlgoObserver()` which logs metrics available to the algo using the tensorboard summarywriter. 
    - Logs and checkpoints are automatically created in a directory called nn (by default).
    - Custom algorithms and observers can also be provided based on your requirements (more on this below).


2. `rl_games.common.Objectfactory()`
    - Creates algorithms or players. Has the `register_builder(self, name, builder)` method that adds a function that returns whatever is being built (note that the name argument is of type string). For example the following line adds the name a2c_continuous to a lambda function that returns the A2CAgent
        ```
        register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        ```
    - Also has a `create(self, name, **kwargs)` method that simply returns one of the registered builders by name

3. `RL Algorithms`
    - rl_games has several reinforcement learning algorithms. Most of these inherit from some sort of base algorithm class, for example, `rl_games.algos_torch.A2CBase`. In rl_games environments are instantiated by the algorithm. 

4. `Environments (rl_games.common.vecenv & rl_games.common.env_configurations)`
    - The vecenv script holds classes to instantiate different environments based on their type. Since rl_games is quite a broad library, it supports multiple environment types (such as openAI gym envs, brax envs, cule envs etc). These environment types and their base classes are stored in the `rl_games.common.vecenv.vecenv_config` dictionary. The environment class enables stuff like running multiple parallel environments, or running multi-agent environments. By default, all available environments are already added. Adding new environments is explained below.

    - `rl_games.common.env_configurations` is another dictionary that stores `env_name: {'vecenv_type', 'env_creator}` information. For example, the following stores the environment name "CartPole-v1" with a value for its type and a lambda function that instantiates the respective gym env.
        ```    
        'CartPole-v1' : {
            'vecenv_type' : 'RAY',
            'env_creator' : lambda **kwargs : gym.make('CartPole-v1'),}
        ```
    - The general idea here is that the algorithm base class instantiates a new environment by looking at the env_name (for example 'CartPole-v1') in the config file. Internally, the name 'CartPole-v1' is used to get the env type from `rl_games.common.env_configurations`. The type then goes into the `vecenv.vecenv_config` dict which returns the actual environment class (such as RayVecEnv). 

    - Note, the env class (such as RayVecEnv) then internally uses the 'env_creator' key to instantiate the environment using whatever function was given to it (for example, `lambda **kwargs : gym.make('CartPole-v1')`)



## Extending rl_games for your own work
While rl_games provides a great baseline implementation of several environments and algorithms, it is also a great starting point for your own work. The rest of this write-up explains how new environments or algorithms can be added. It is based on the setup from [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), the NVIDIA repository for RL simulations and training. We use [hydra](https://hydra.cc/docs/intro/) for easier configuration management. Further, instead of directly using `runner.py` we use another similar script called `train.py` which allows us to dynamically add new environments and insert out own algorithms. 

With this considered, our final file structure is something like this.

```
project dir
│   train.py    
│
└───tasks dir (sometimes also called envs dir)
│   │   customenv.py
│   │   rl_games_env_utils.py
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
|
└───cfg dir (main hydra configs)
│   │   config.yaml (main config for the setting up simulators etc. if needed)
│   │
│   └─── task dir (configs for the env)
│       │   customenv.yaml
│       │   otherenv.yaml
│       │   ...
|   
│   └─── train dir (configs for training the algorithm)
│       │   customenvAlgo.yaml
│       │   otherenvPPO.yaml
│       │   ...
|
└───algos dir (custom wrappers for training algorithms in rl_games)
|   │   custom_network_builder.py
|   │   custom_algo.py
|   | ...
|
└───runs dir (generated automatically on running train.py)
│   └─── env_name_alg_name_datetime dir (train logs)
│       └─── nn
|           |   checkpoints.pth
│       └─── summaries
            |   events.out...
```

## Adding new gym-like environments
New environments can be used with the rl_games setup by first defining the TYPE of the new env. A new environment TYPE can be added by calling the `vecenv.register(config_name, func)` function that simply adds the `config_name:func` pair to the dictionary. For example the following line adds a 'RAY' type env with a lambda function that then instantiates the RayVecEnv class. The "RayVecEnv" holds "RayWorkers" that internally store the environment. This automatically allows for multi-env training.

```
register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))
```

For gym-like envs (that inherit from the gym base class), the TYPE can simply be `RayVecEnv` from rl_games. Adding a gym-like environment essentially translates to creating a gym-like env class (that inherits from gym.Env) and adding this under the type 'RAY' to `rl_games.common.env_configurations`. Ideally, this needs to be done by adding the key value pair `env_name: {'vecenv_type', 'env_creator}` to `env_configurations.configurations`. However, this requires modifying the rl_games library. If you do not wish to do that then you can instead use the register method to add your new env to the dictionary, then make a copy of the RayVecEnv and RayWorked classes and change the __init__ method to instead take in the modified env configurations dict. For example

```python
def create_new_env(**kwargs):
    # Instantiate new env
    env =  newEnv()

    #For example, env = gym.make('LunarLanderContinuous-v2')
    return env

from rl_games.common import env_configurations, vecenv

env_configurations.register('customenv', {
    'vecenv_type': 'COPYRAY',
    'env_creator': lambda **kwargs: create_custom_env(**kwargs),
})

vecenv.register('COPYRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))

------------

# Make a copy of RayVecEnv

class CustomRayVecEnv(IVecEnv):
    import ray

    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        ### ADDED CHANGE ###
        # Explicityly passing in the dictionary containing env_name: {vecenv_type, env_creator}
        self.config_dict = config_dict

        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)

        
        self.remote_worker = self.ray.remote(CustomRayWorker)
        self.workers = [self.remote_worker.remote(self.config_dict, self.config_name, kwargs) for i in range(self.num_actors)]

        ...
        ...

# Make a copy of RayWorker

class CustomRayWorker:
    ## ADDED CHANGE ###
    # Add config_dict to init
    def __init__(self, config_dict, config_name, config):
        self.env = config_dict[config_name]['env_creator'](**config)

        ...
        ...
```


With the new TYPE class set-up, the custom environment (called pushT here) can be added within the hydra-decorated main functions as

```python
@hydra.main(version_base="1.1", config_name="custom_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    from custom_envs.pusht_single_env import PushTEnv
    from custom_envs.customenv_utils import CustomRayVecEnv, PushTAlgoObserver

    def create_pusht_env(**kwargs):
        env =  PushTEnv()
        return env

    env_configurations.register('pushT', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    })

    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))
```
## Adding non-gym environments & simulators
Non-gym environments can be added in the same way. However, now you also need to create your own TYPE class. [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) does this by defining a new RLGPU type that uses the IsaacGym simulation environment. An example of this can be found in the IsaacGymEnvs library.


## New algorithms and observers within rl_games

Adding a custom algorithm essentially translates to registering your own builder and player within the `rl_games.torch_runner.Runner`. IsaacGymEnvs does this by adding the following within the dydra-decorated main function (their algo is called AMP).

```python
# register new AMP network builder and agent
def build_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    return runner
```

As you might have noticed from above, you can also add a custom observer to log whatever data you need. You can make your own by inheriting from `rl_games.common.algo_observer.AlgoObserver`. If you wish to log scores, your custom environment must have a "scores" key in the info dictionary (the info dict is returned when the environment is stepped). 