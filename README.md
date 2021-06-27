# RL Games: High performance RL library  

## Papers and related links

* <https://arxiv.org/abs/2011.09533>

## Some results on interesting environments  

* [Starcraft 2 Multi Agents](docs/SMAC.md)  
* [NVIDIA Isaac Gym](docs/ISAAC_GYM.md)  
* [BRAX](docs/BRAX.md)  
* [Old TF1.x results](docs/BRAX.md)  

Implemented in Pytorch:

* PPO with the support of asymmetric actor-critic variant
* Support of end-to-end GPU training pipeline
* Masked actions support

 Implemented in Tensorflow 1.x (not updates now):

* Rainbow DQN
* A2C
* PPO

# Installation
TODO

# Training
How to run and play simple atari pong:  
```python runner.py --train --file 'rl_games/configs/atari/ppo_pong.yaml' ```
```python runner.py --play --file 'rl_games/configs/atari/ppo_pong.yaml' --checkpoint 'nn/PongNoFrameskip.pth'```  

# Troubleshouting

* Some of the supported envs are not installed with setup.py, you need to manually install them
