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

```pip install rl-games```

Or clone repo and run:
```pip install -e .```

Or:
```pip install git+https://github.com/Denys88/rl_games.git```

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
```python runner.py --play --file rl_games/configs/atari/ppo_ant.yaml --checkpoint nn/Ant_brax.pth``` 


# Release Notes

1.1.0

* Added to pypi: ```pip install rl-games```
* Added reporting env (sim) step fps, without policy inference. Improved naming.
* Renames in yaml config for better readability: steps_num to horizon_length amd lr_threshold to kl_threshold

# Troubleshouting

* Some of the supported envs are not installed with setup.py, you need to manually install them
* Starting from rl-games 1.1.0 old yaml configs won't be compatible with the new version: 
    * ```steps_num``` should be changed to ```horizon_length``` amd ```lr_threshold``` to ```kl_threshold```


