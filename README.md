# Basic RL Algorithms Implementations
* Starcraft 2 Multiple Agents Results with PPO (https://github.com/oxwhirl/smac)
* Every agent was controlled independently and has restricted information
* All the environments were trained with a default difficulty level 7
* No curriculum, just baseline PPO
* Full state information wasn't used for critic, actor and critic recieved the same agent observations
* Most results are significantly better by win rate and were trained on a single PC much faster than QMIX (https://arxiv.org/pdf/1902.04043.pdf), MAVEN (https://arxiv.org/pdf/1910.07483.pdf) or QTRAN
* No hyperparameter search
* 4 frames + conv1d actor-critic network
* Miniepoch num was set to 1, higher numbers didn't work
* Simple MLP networks didnot work good on hard envs

[![Watch the video](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/mmm2.gif)](https://www.youtube.com/watch?v=F_IfFz-s-iQ)

# How to run configs:
# Pytorch
* python runner.py --train --file rl_games/configs/smac/3m_torch.yaml
* python runner.py --play --file rl_games/configs/smac/3m_torch.yaml --checkpoint 'nn/3m_cnn'
# Tensorflow
* python runner.py --tf --train --file rl_games/configs/smac/3m_torch.yaml
* python runner.py --tf --play --file rl_games/configs/smac/3m_torch.yaml --checkpoint 'nn/3m_cnn'
* tensorboard --logdir runs
# Results on some environments:
* 2m_vs_1z took near 2 minutes to achive 100% WR
* corridor took near 2 hours for 95+% WR
* MMM2 4 hours for 90+% WR
* 6h_vs_8z got 82% WR after 8 hours of training
* 5m_vs_6m got 72% WR after 8 hours of training

# Plots:
FPS in these plots is calculated on per env basis except MMM2 (it was scaled by number of agents which is 10), to get a win rate per number of environmental steps info, the same as used in plots in QMIX, MAVEN, QTRAN or Deep Coordination Graphs (https://arxiv.org/pdf/1910.00091.pdf) papers FPS numbers under the horizontal axis should be devided by number of agents in player's team.

* 2m_vs_1z:
![2m_vs_1z](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/2m_vs_1z.png)
* 3s5z_vs_3s6z:
![3s5z_vs_3s6z](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/3s5z_vs_3s6z.png)
* 3s_vs_5z:
![3s_vs_5z](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/3s_vs_5z.png)
* corridor:
![corridor](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/corridor.png)
* 5m_vs_6m:
![5m_vs_6m](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/5m_vs_6m.png)
* MMM2:
![MMM2](https://github.com/Denys88/dqn_atari/blob/master/pictures/smac/MMM2.png)


[Link to the continuous results](https://github.com/Denys88/rl_games/blob/master/CONTINUOUS_RESULTS.md)

Currently Implemented:
* DQN
* Double DQN
* Dueling DQN
* Noisy DQN
* N-Step DQN
* Categorical
* Rainbow DQN
* A2C
* PPO

Tensorflow implementations of the DQN atari.

* Double dueling DQN vs DQN with the same parameters

![alt text](https://github.com/Denys88/dqn_atari/blob/master/pictures/dqn_vs_dddqn.png)
Near 90 minutes to learn with this setup.

* Different DQN Configurations tests

Light grey is noisy 1-step dddqn.
Noisy 3-step dddqn was even faster.
Best network (configuration 5) needs near 20 minutes to learn, on NVIDIA 1080.
Currently the best setup for pong is noisy 3-step double dueling network.
In pong_runs.py different experiments could be found.
Less then 200k frames to take score > 18.
![alt text](https://github.com/Denys88/dqn_atari/blob/master/pictures/pong_dqn.png)
DQN has more optimistic Q value estimations.

# Other Games Results
This results are not stable. Just best games, for good average results you need to train network more then 10 million steps.
Some games need 50m steps.

* 5 million frames two step noisy double dueling dqn:

[![Watch the video](https://j.gifs.com/K1OL6r.gif)](https://youtu.be/Lu9Cm9K_6ms)

* Random lucky game in Space Invaders after less then one hour learning:

[![Watch the video](https://j.gifs.com/D1RQE5.gif)](https://www.youtube.com/watch?v=LO0RL437rh4)


# A2C and PPO Results
* More than 2 hours for Pong to achieve 20 score with one actor playing. 
* 8 Hours for Supermario lvl1

[![Watch the video](https://j.gifs.com/nxOYyp.gif)](https://www.youtube.com/watch?v=T9ujS3HIvMY)

* PPO with LSTM layers

[![Watch the video](https://j.gifs.com/YWV9W0.gif)](https://www.youtube.com/watch?v=fjY4AWbmhHg)


![alt text](https://github.com/Denys88/dqn_atari/blob/master/pictures/mario_random_stages.png)
