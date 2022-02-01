# RL Games: High performance RL library  

## Papers and related links

* Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning: https://arxiv.org/abs/2108.10470
* Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger: https://s2r2-ig.github.io/ https://arxiv.org/abs/2108.09779
* Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>

## Some results on interesting environments  

* [NVIDIA Isaac Gym](docs/ISAAC_GYM.md)

![Ant_running](https://user-images.githubusercontent.com/463063/125260924-a5969800-e2b5-11eb-931c-116cc90d4bbe.gif)
![Humanoid_running](https://user-images.githubusercontent.com/463063/125266095-4edf8d00-e2ba-11eb-9c1a-4dc1524adf71.gif)

![Allegro_Hand_400](https://user-images.githubusercontent.com/463063/125261559-38373700-e2b6-11eb-80eb-b250a0693f0b.gif)
![Shadow_Hand_OpenAI](https://user-images.githubusercontent.com/463063/125262637-328e2100-e2b7-11eb-99af-ea546a53f66a.gif)

* [Starcraft 2 Multi Agents](docs/SMAC.md)  
* [BRAX](docs/BRAX.md)  
* [Random Envs](docs/OTHER.md)  


Implemented in Pytorch:

* PPO with the support of asymmetric actor-critic variant
* Support of end-to-end GPU accelerated training pipeline with Isaac Gym and Brax
* Masked actions support
* Multi-agent training, decentralized and centralized critic variants
* Self-play 

 Implemented in Tensorflow 1.x (was removed in this version):

* Rainbow DQN
* A2C
* PPO

# Installation

For maximum training performance a preliminary installation of Pytorch 1.9+ with CUDA 11.1 is highly recommended:

```conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia``` or:
```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.htm```

Then:

```pip install rl-games```

To run Atari games or Box2d based environments training they need to be additionally installed with ```pip install gym[atari]``` or ```pip install gym[box2d]``` respectively.


# Training
**NVIDIA Isaac Gym**

Download and follow the installation instructions from https://developer.nvidia.com/isaac-gym  
Run from ```python/rlgpu``` directory:

Ant  
```python rlg_train.py --task Ant --headless```  
```python rlg_train.py --task Ant --play --checkpoint nn/Ant.pth --num_envs 100``` 

Humanoid  
```python rlg_train.py --task Humanoid --headless```  
```python rlg_train.py --task Humanoid --play --checkpoint nn/Humanoid.pth --num_envs 100``` 

Shadow Hand block orientation task  
```python rlg_train.py --task ShadowHand --headless```  
```python rlg_train.py --task ShadowHand --play --checkpoint nn/ShadowHand.pth --num_envs 100``` 


**Atari Pong**    
```python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml```  
```python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth```  


**Brax Ant**  
```python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml```  
```python runner.py --play --file rl_games/configs/brax/ppo_ant.yaml --checkpoint nn/Ant_brax.pth``` 

# Config Parameters

| Field                  | Example Value                             | Default  | Description                                                                                            |
|------------------------|-------------------------------------------|----------|--------------------------------------------------------------------------------------------------------|
| seed                   | 8                                         |  None    | Seed for pytorch, numpy etc.                                                            |
| algo                   |                                           |          | Algorithm block.                                             |
|   name                 | a2c_continuous                            |  None    | Algorithm name. Possible values are: sac, a2c_discrete, a2c_continuous                          |
| model                  |                                           |          | Model block.                                                                                        |
|   name                 | continuous_a2c_logstd                     |  None    | Possible values: continuous_a2c ( expects sigma to be (0, +inf), continuous_a2c_logstd  ( expects sigma to be (-inf, +inf), a2c_discrete, a2c_multi_discrete                      |
| network                |                                           |          | Network description.                                                                            |
|   name                 | actor_critic                              |          | Possible values: actor_critic or soft_actor_critic.                                                                           |
|   separate             | False                                     |          | Whether use or not separate network with same same architecture for critic. In almost all cases if you normalize value it is better to have it False                                                                                           |
|   space                |                                           |          | Network space                                                  |
|     continuous         |                                           |          | continuous or discrete                                |
|       mu_activation    | None                                      |          | Activation for mu. In almost all cases None works the best, but we may try tanh.                             |
|       sigma_activation | None                                      |          | Activation for sigma. Will be threated as log(sigma) or sigma depending on model.                                                                                    |
|       mu_init          |                                           |          | Initializer for mu.                                                   |
|         name           | default                                   |          |                                                                                     |
|       sigma_init       |                                           |          | Initializer for sigma. if you are using logstd model good value is 0.                          |
|         name           | const_initializer                         |          |                                                    |
|         val            | 0                                         |          |                  |
|       fixed_sigma      | True                                      |          | If true then sigma vector doesn't depend on input.                                                   |
|   cnn                  |                                           |          | Convolution block.                    |
|     type               | conv2d                                    |          | Type: right now two types supported: conv2d or conv1d                                               |
|     activation         | elu                                       |          | activation between conv layers.                                  |
|     initializer        |                                           |          | Initialier. I took some names from the tensorflow.                                                             |
|       name             | glorot_normal_initializer                 |          | initializer name                                                                                         |
|       gain             | 1.4142                                    |          | Additional parameter.                                                                  |
|     convs              |                                           |          | Convolution layers. Same parameters as we have in torch.                                                                                        |
|         filters        | 32                                        |          | Number of filters.                                                                                                  |
|         kernel_size    | 8                                         |          | Kernel size.                                                                                                    |
|         strides        | 4                                         |          | Strides                                                                  |
|         padding        | 0                                         |          | Padding                                                                                          |
|         filters        | 64                                        |          | Next convolution layer info.                                                                  |
|         kernel_size    | 4                                         |          |                                                                                                          |
|         strides        | 2                                         |          |                                                                                                |
|         padding        | 0                                         |          |                                                              |
|         filters        | 64                                        |          |                                           |
|         kernel_size    | 3                                         |          |                                                                                                         |
|         strides        | 1                                         |          |                                                |
|         padding        | 0                                         |          |                       
|   mlp                  |                                           |          | MLP Block. Convolution is supported too. See other config examples.                                                                                           |
|     units              |                                           |          | Lorem ipsum dolor sit amet, consecteteur adipiscing elit.                                              |
|     d2rl               | False                                     |          | Use d2rl architecture from https://arxiv.org/abs/2010.09163.                                                                                     |
|     activation         | elu                                       |          | Activations between dense layers.                                |
|     initializer        |                                           |          | Lorem ipsum dolor sit amet, consecteteur adipiscing elit b'duis'.                                      |
|       name             | default                                   |          | Lorem ipsum dolor sit amet, consecteteur adipiscing elit b'urna' b'mi'.                                |
|   rnn                  |                                           |          | RNN block.                                 |
|     name               | lstm                                      |          | RNN Layer name. lstm and gru are supported.                                                                                          |
|     units              | 256                                       |          | Number of units.                                             |
|     layers             | 1                                         |          | Number of layers                                                                                                  |
|     before_mlp         | False                                     | False    | Apply rnn before mlp block or not.                                                                                                  |
| config                 |                                           |          | RL Config block.                               |
|   reward_shaper        |                                           |          | Reward Shaper. Can apply simple transformations.                                              |
|     min_val            | -1                                        |          | You can apply min_val, max_val, scale and shift.                  |
|     scale_value        | 0.1                                       | 1        |  |
|   normalize_advantage  | True                                      | True     | Normalize Advantage.                                                              |
|   gamma                | 0.995                                     |          | Reward Discount                                                              |
|   tau                  | 0.95                                      |          | Lambda for GAE. Called tau by mistake long time ago because lambda is keyword in python :(         |
|   learning_rate        | 3e-4                                      |          | Learning rate.                                                   |
|   name                 | walker                                    |          | Name which will be used in tensorboard.                  |
|   save_best_after      | 10                                        |          | How many epochs to wait before start saving checkpoint with best score.                                                                                    |
|   score_to_win         | 300                                       |          | If score is >=value then this value training will stop.        |
|   grad_norm            | 1.5                                       |          | Grad norm. Applied if truncate_grads is True. Good value is in (1.0, 10.0)                                             |
|   entropy_coef         | 0                                         |          | Entropy coefficient. Good value for continuous space is 0. For discrete is 0.02                                              |
|   truncate_grads       | True                                      |          | Apply truncate grads or not. It stabilizes training.                                                  |
|   env_name             | BipedalWalker-v3                          |          | Envinronment name.            |
|   ppo                  | True                                      | True     | Use ppo loss or actor critic. Should be always true.                                    |
|   e_clip               | 0.2                                       |          | clip parameter for ppo loss.                                                                                 |
|   clip_value           | False                                     |          | Apply clip to the value loss. If you are using normalize_value you don't need it.                                                                                 |
|   num_actors           | 16                                        |          | Number of running actors.                           |
|   horizon_length       | 4096                                      |          | Horizon length per each actor. Total number of steps will be num_actors*horizon_length * num_agents (if env is not MA num_agents==1).                          |
|   minibatch_size       | 8192                                      |          | Minibatch size. total number number of steps must be divisible by minibatch size.                                                           |
|   mini_epochs          | 4                                         |          | Number of miniepochs. Good value is in [1,10]                                                                            |
|   critic_coef          | 2                                         |          | Critic coef. by default critic_loss= critic_coef * 1/2 * MSE.                                                                                    |
|   lr_schedule          | adaptive                                  | None     | Scheduler type. Could be None, linear or adaptive. Adaptive is the best for continuous.                                     |
|   schedule_type        | standard                                  |          | if schedule is adaptive there are a few places where we can change LR based on KL. If you standard it will be changed every miniepoch.                                                                                          |
|   kl_threshold         | 0.008                                     |          | KL threshould for adaptive schedule. if KL < kl_threshold/2 lr = lr * 1.5 and opposite.                                            |
|   normalize_input      | True                                      |          | Apply running mean std for input.                                                                           |
|   bounds_loss_coef     | 0.0                                       |          | Coefficient to the auxiary loss for continuous space.    |
|   max_epochs           | 10000                                     |          | Maximum number of epochs to run.                     |
|   normalize_value      | True                                      |          | Use value running mean std normalization.                                                                                          |
|   use_diagnostics      | True                                      |          | Adds more information into the tensorboard.                                              |
|   value_bootstrap      | True                                      |          | Bootstraping value when episode is finished. Very useful for different locomotion envs.               |
|   bound_loss_type      | 'regularisation'                          | None     | Adds aux loss for continuous case. 'regularisation' is the sum of sqaured actions. 'bound' is the sam of actions higher than 1.1.                                              |
|   bounds_loss_coef     | 0.0005                                    | 0        | Regularisation coefficient               |
|   player               |                                           |          | Player configuration block.                                                                                |
|     render             | True                                      | False    | Render environment                                                                            |
|     determenistic      | True                                      | True     | Use deterministic policy ( argmax or mu) or stochastic.                                                                                |
|     games_num          | 200                                       |          | Number of games to run in the player mode.                                             |
|   env_config           |                                           |          | Env configuration block. It goes directly to the environment. This example was take for my atari wrapper.                                                                                |
|     skip               | 4                                         |          | Number of frames to skip                                                                           |
|     name               | 'BreakoutNoFrameskip-v4'                  |          | Name of exact atari env. Of course depending on your env this parameters may be different.                                                                                |

## Custom network example: 
[simple test network](rl_games/envs/test_network.py)  
This network takes dictionary observation.
To register it you can add code in your __init__.py

```
from rl_games.envs.test_network import TestNetBuilder 
from rl_games.algos_torch import model_builder
model_builder.register_network('testnet', TestNetBuilder)
```
[simple test environment](rl_games/envs/test/rnn_env.py)
[example environment](rl_games/envs/test/example_env.py)  

Additional environment supported properties and functions  

| Field                       | Default Value   | Description                         |
|-----------------------------|-----------------|-------------------------------------|
| use_central_value           | 200             | If true than returned obs is expected to be dict with 'obs' and 'state'                                    |
| value_size                  | 1               | Shape of the returned rewards. Network wil support multihead value automatically.                                    |
| concat_infos                | False           | Should default vecenv convert list of dicts to the dicts of lists. Very usefull if you want to use value_boostrapping. in this case you need to always return 'time_outs' : True or False, from the env.                                    |
| get_number_of_agents(self)  | 1               | Returns number of agents in the environment                                    |
| has_action_mask(self)       | False           | Returns True if environment has invalid actions mask.                                    |
| get_action_mask(self)       | None            | Returns action masks if  has_action_mask is true.  Good example is [SMAC Env](rl_games/envs/test/smac_env.py)                                 |


## Release Notes


1.3.0

* Simplified rnn implementation. Works a little bit slower but much more stable. 
* Now central value can be non-rnn if policy is rnn.
* Removed load_checkpoint from the yaml file. now --checkpoint works for both train and play.

1.2.0

* Added Swish (SILU) and GELU activations, it can improve Isaac Gym results for some of the envs.
* Removed tensorflow and made initial cleanup of the old/unused code.
* Simplified runner.
* Now networks are created in the algos with load_network method.

1.1.4

* Fixed crash in a play (test) mode in player, when simulation and rl_devices are not the same.
* Fixed variuos multi gpu errors.

1.1.3

* Fixed crash when running single Isaac Gym environment in a play (test) mode.
* Added config parameter ```clip_actions``` for switching off internal action clipping and rescaling

1.1.0

* Added to pypi: ```pip install rl-games```
* Added reporting env (sim) step fps, without policy inference. Improved naming.
* Renames in yaml config for better readability: steps_num to horizon_length amd lr_threshold to kl_threshold




# Troubleshouting

* Some of the supported envs are not installed with setup.py, you need to manually install them
* Starting from rl-games 1.1.0 old yaml configs won't be compatible with the new version: 
    * ```steps_num``` should be changed to ```horizon_length``` amd ```lr_threshold``` to ```kl_threshold```

# Known issues

* Running a single environment with Isaac Gym can cause crash, if it happens switch to at least 2 environments simulated in parallel
    

