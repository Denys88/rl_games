# RL Games: High performance RL library  

## Papers and related links

* <https://arxiv.org/abs/2011.09533>

## Some results on interesting environments  

* [NVIDIA Isaac Gym](docs/ISAAC_GYM.md)
* [Starcraft 2 Multi Agents](docs/SMAC.md)  
* [BRAX](docs/BRAX.md)  
* [Old TF1.x results](docs/BRAX.md)  

## Config file  

* [Configuration](docs/CONFIG_PARAMS.md)  

Implemented in Pytorch:

* PPO with the support of asymmetric actor-critic variant
* Support of end-to-end GPU accelerated training pipeline with Isaac Gym and Brax
* Masked actions support
* Multi-agent training, decentralized and centralized critic variants
* Self-play 

 Implemented in Tensorflow 1.x (not updates now):

* Rainbow DQN
* A2C
* PPO

# Installation
Clone repo and run:
```pip install -e .```

Or:
```pip install git+https://github.com/Denys88/rl_games.git```

# Training
NVIDIA Isaac Gym

Download and follow the installation instructions from https://developer.nvidia.com/isaac-gym  
Run from ```python/rlgpu``` directory:

Ant:  
```python rlg_train.py --task Ant --headless```  
```python rlg_train.py --task Ant --play --checkpoint nn/Ant.pth --num_envs 100``` 

Humanoid:  
```python rlg_train.py --task Humanoid --headless```  
```python rlg_train.py --task Humanoid --play --checkpoint nn/Humanoid.pth --num_envs 100``` 

Shadow Hand block orientation task:  
```python rlg_train.py --task ShadowHand --headless```  
```python rlg_train.py --task ShadowHand --play --checkpoint nn/ShadowHand.pth --num_envs 100``` 

How to run and play simple atari pong:  
```python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml```  
```python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth```  

Brax Ant env:  
```python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml```  
```python runner.py --play --file rl_games/configs/atari/ppo_ant.yaml --checkpoint nn/Ant_brax.pth``` 

# Troubleshouting

* Some of the supported envs are not installed with setup.py, you need to manually install them
