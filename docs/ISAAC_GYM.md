## Isaac Gym Results
https://developer.nvidia.com/isaac-gym

## What's Written Below

Content below is written is written to complement `HOW_TO_RL_GAMES.md` in the same directory, while focusing more on **explaining the implementations** in the classes and how to **customize the training (testing) loops, models and networks**. Since the AMP implementation in `IsaacGymEnvs` is used as the example, so you are reading me here under this file.

## Program Entry Point

The primary entry point for both training and testing within `IsaacGymEnvs` is the `train.py` script. This file initializes an instance of the `rl_games.torch_runner.Runner` class, and depending on the mode selected, either the `run_train` or `run_play` function is executed. Additionally, `train.py` allows for custom implementations of training and testing loops, as well as the integration of custom neural networks and models into the library through the `build_runner` function, a process referred to as "registering." By registering custom code, the library can be configured to execute the user-defined code by specifying the appropriate names within the configuration file.

In RL Games, the training algorithms are referred to as "agents," while their counterparts for testing are known as "players." In the `run_train` function, an agent is instantiated, and training is initiated through the `agent.train` call. Similarly, in the `run_play` function, a player is created, and testing begins by invoking `player.run`. Thus, the core entry points for training and testing in RL Games are the `train` function for agents and the `run` function for players.

```python
def run_train(self, args):
    """Run the training procedure from the algorithm passed in."""
    
    print('Started to train')
    agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
    _restore(agent, args)
    _override_sigma(agent, args)
    agent.train()
    
def run_play(self, args):
    """Run the inference procedure from the algorithm passed in."""
    
    print('Started to play')
    player = self.create_player()
    _restore(player, args)
    _override_sigma(player, args)
    player.run()
```

## Training Algorithms

The creation of an agent is handled by the `algo_factory`, as demonstrated in the code above. By default, the `algo_factory` is registered with continuous-action A2C, discrete-action A2C, and SAC. This default registration is found within the constructor of the `Runner` class, and its implementation is shown below. Our primary focus will be on understanding `A2CAgent`, as it is the primary algorithm used for most continuous-control tasks in `IsaacGymEnvs`.

```python
self.algo_factory.register_builder(
    'a2c_continuous',
    lambda **kwargs: a2c_continuous.A2CAgent(**kwargs)
)
self.algo_factory.register_builder(
    'a2c_discrete',
    lambda **kwargs: a2c_discrete.DiscreteA2CAgent(**kwargs)
) 
self.algo_factory.register_builder(
    'sac',
    lambda **kwargs: sac_agent.SACAgent(**kwargs)
)
```

At the base of all RL Games algorithms is the `BaseAlgorithm` class, an abstract class that defines several essential methods, including `train` and `train_epoch`, which are critical for training. The `A2CBase` class inherits from `BaseAlgorithm` and provides many shared functionalities for both continuous and discrete A2C agents. These include methods such as `play_steps` and `play_steps_rnn` for gathering rollout data, and `env_step` and `env_reset` for interacting with the environment. However, functions directly related to training—like `train`, `train_epoch`, `update_epoch`, `prepare_dataset`, `train_actor_critic`, and `calc_gradients`—are left unimplemented at this level. These functions are implemented in `ContinuousA2CBase`, a subclass of `A2CBase`, and further in `A2CAgent`, a subclass of `ContinuousA2CBase`.

The `ContinuousA2CBase` class is responsible for the core logic of agent training, specifically in the methods `train`, `train_epoch`, and `prepare_dataset`. In the `train` function, the environment is reset once before entering the main training loop. This loop consists of three primary stages: 

1. Calling `update_epoch`.
2. Running `train_epoch`.
3. Logging key information, such as episode length, rewards, and losses.

The `update_epoch` function, which increments the epoch count, is implemented in `A2CAgent`. The heart of the training process is the `train_epoch` function, which operates as follows:

1. `play_steps` or `play_steps_rnn` is called to generate rollout data in the form of a dictionary of tensors, `batch_dict`. The number of environment steps collected equals the configured `horizon_length`.
2. `prepare_dataset` modifies the tensors in `batch_dict`, which may include normalizing values and advantages, depending on the configuration.
3. Multiple mini-epochs are executed. In each mini-epoch, the dataset is divided into mini-batches, which are sequentially fed into `train_actor_critic`. Function `train_actor_critic`, implemented in `A2CAgent`, internally calls `calc_grad`, also found in `A2CAgent`.

The `A2CAgent` class, which inherits from `ContinuousA2CBase`, handles the crucial task of gradient calculation and model parameter optimization in its `calc_grad` function. Specifically, `calc_grad` first performs a forward pass of the policy model with PyTorch’s gradients and computational graph enabled. It then calculates the individual loss terms as well as the total scalar loss, runs the backward pass to compute gradients, truncates gradients if necessary, updates model parameters via the optimizer, and finally logs the relevant training metrics such as loss terms and learning rates.

With an understanding of the default functions, it becomes straightforward to customize agents by inheriting from `A2CAgent` and overriding specific methods to suit particular needs. A good example of this is the implementation of the AMP algorithm in `IsaacGymEnvs`, where the `AMPAgent` class is created and registered in `train.py`, as shown below.

```python
_runner.algo_factory.register_builder(
    'amp_continuous',
    lambda **kwargs: amp_continuous.AMPAgent(**kwargs)
)
```

## Players

Similar to training algorithms, default players are registered with `player_factory` in the `Runner` class. These include `PPOPlayerContinuous`, `PPOPlayerDiscrete`, and `SACPlayer`. Each of these player classes inherits from the `BasePlayer` class, which provides a common `run` function. The derived player classes implement specific methods for restoring from model checkpoints (`restore`), initializing the RNN (`reset`), and generating actions based on observations through `get_action` and `get_masked_action`.

The testing loop is simpler compared to the training loop. It starts by resetting the environment to obtain the initial observation. Then, for `max_steps` iterations, the loop feeds the observation into the model to generate an action, which is applied to the environment to retrieve the next observation, reward, and other necessary data. This process is repeated for `n_games` episodes, after which the average reward and episode lengths are calculated and displayed.

Customizing the testing loop is as straightforward as customizing the training loop. By inheriting from a default player class, one can override specific functions as needed. As with custom training algorithms, customized players must also be registered with `player_factory` in `train.py`, as demonstrated below.

```python
self.player_factory.register_builder(
    'a2c_continuous',
    lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
)
self.player_factory.register_builder(
    'a2c_discrete',
    lambda **kwargs: players.PpoPlayerDiscrete(**kwargs)
)
self.player_factory.register_builder(
    'sac',
    lambda **kwargs: players.SACPlayer(**kwargs)
)

_runner.player_factory.register_builder(
    'amp_continuous',
    lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
)
```

## Models and Networks

The terminology and implementation of models and networks in RL Games version `1.6.1` can be confusing for new users. Below is a high-level overview of their functionality and relationships:

- **Network Builder:** Network builder classes, such as `A2CBuilder` and `SACBuilder`, are subclasses of `NetworkBuilder` and can be found in `algos_torch.network_builder`. The core component of a network builder is the nested `Network` class (the "inner network" class), which is typically derived from `torch.nn.Module`. This class receives a dictionary of tensors, such as observations and other necessary inputs, and outputs a tuple of tensors from which actions can be generated. The `forward` function of the `Network` class handles this transformation.

- **Model:** Model classes, like `ModelA2C` and `ModelSACContinuous`, inherit from `BaseModel` in `algos_torch.models`. They are similar to network builders, as each contains a nested `Network` class (the "model network" class) and a `build` function to construct an instance of this network.

- **Model & Network in Algorithm:** In a default agent or player algorithm, `self.model` refers to an instance of the model network class, while `self.network` refers to an instance of the model class.

- **Model Builder:** The `ModelBuilder` class, located in `algos_torch.model_builder`, is responsible for loading and managing models. It provides a `load` function, which creates a model instance based on the specified name.

Customizing models requires implementing a custom network builder and model class. These custom classes should be registered in the `Runner` class within `train.py`. A good reference example is the AMP implementation, as shown below.

```python
# algos_torch.model_builder.NetworkBuilder.__init__
self.network_factory.register_builder(
    'actor_critic',
    lambda **kwargs: network_builder.A2CBuilder()
)
self.network_factory.register_builder(
    'resnet_actor_critic',
    lambda **kwargs: network_builder.A2CResnetBuilder()
)
self.network_factory.register_builder(
    'rnd_curiosity',
    lambda **kwargs: network_builder.RNDCuriosityBuilder()
)
self.network_factory.register_builder(
    'soft_actor_critic',
    lambda **kwargs: network_builder.SACBuilder()
)

# algos_torch.model_builder.ModelBuilder.__init__
self.model_factory.register_builder(
    'discrete_a2c',
    lambda network, **kwargs: models.ModelA2C(network)
)
self.model_factory.register_builder(
    'multi_discrete_a2c',
    lambda network, **kwargs: models.ModelA2CMultiDiscrete(network)
)
self.model_factory.register_builder(
    'continuous_a2c',
    lambda network, **kwargs: models.ModelA2CContinuous(network)
)
self.model_factory.register_builder(
    'continuous_a2c_logstd',
    lambda network, **kwargs: models.ModelA2CContinuousLogStd(network)
)
self.model_factory.register_builder(
    'soft_actor_critic',
    lambda network, **kwargs: models.ModelSACContinuous(network)
)
self.model_factory.register_builder(
    'central_value',
    lambda network, **kwargs: models.ModelCentralValue(network)
)

# isaacgymenvs.train.launch_rlg_hydra.build_runner
model_builder.register_model(
    'continuous_amp',
    lambda network, **kwargs: amp_models.ModelAMPContinuous(network),
)
model_builder.register_network(
    'amp',
    lambda **kwargs: amp_network_builder.AMPBuilder()
)
```
