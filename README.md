# RL Games: High performance RL library  

## Some results on interest
[Starcraft 2 Multi Agents](docs/SMAC.md)  
[Isaac Gym](docs/ISAAC_GYM.md)  
[BRAX](docs/BRAX.md)  
[Old TF1.x results](docs/BRAX.md)  

Implemented in Pytorch:
* PPO
* Asymmetric PPO
* Only GPU pipeline
* Masked actions supported

 Implemented in Tensorflow 1.x (not updates now):
* Rainbow DQN
* A2C
* PPO

How to run:
python runner.py --train --file 'rl_games/configs/atari/ppo_pong.yaml'



# Troubleshouting:
* Some of the supported envs are not installed with setup.py, you need to manually install them
