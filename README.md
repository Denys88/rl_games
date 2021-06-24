# Basic RL Algorithms Implementations

[Starcraft 2 Multi Agents](https://github.com/Denys88/rl_games/blob/master/docs/SMAC.md)
[Isaac Gym](https://github.com/Denys88/rl_games/blob/master/docs/ISAAC_GYM.md)
[BRAX](https://github.com/Denys88/rl_games/blob/master/docs/BRAX.md)
[Old TF1.x results](https://github.com/Denys88/rl_games/blob/master/docs/BRAX.md)

Implemented in Pytorch:
* PPO
* Asymmetric PPO
* Only GPU pipeline
* Masked actions supported
* Multidiscrete 

 Implemented in Tensorflow 1.x (not supported now):
* Rainbow DQN
* A2C
* PPO

How to run:
python runner.py --train --file 'rl_games/configs/atari/ppo_pong.yaml'



#Troubleshouting:
* Some of the supported envs are not installed with setup.py, you need to manually install them
