# Basic RL Algorithms Implementations

Currently Implemented:
* DQN
* Double DQN
* Dueling DQN
* Noisy DQN
* N-Step DQN

TODO:
* Rainbow DQN
* A3C, PPO, TRPO, etc

Future Plans:
* Play Sonic
* Reproduce Monte Carlo Tree Search from AlphaGo.

Tensorflow implementations of the DQN atari.
In pong_runs.py there are some setups. Prioritized Replay doesn't work as good as I expected. Probably there is mistake in implementation.

Currently the best setup for pong is noisy 3-step double dueling network.
In test_dqn.ipynb different experiments could be found.
Less then 400 frames to take score > 18.

Double dueling DQN vs DQN with the same parameters:
![alt text](https://github.com/Denys88/dqn_atari/blob/master/pictures/dqn_vs_dddqn.png)
As it was very simple game we get almost the same results,  90 minutes to learn to the 20 reward.
DQN has more optimistic Q value estimations.

# Other Games Results
This results are not stable. Just best games, for good average results you need to train network more then 10 million steps.
Some games need 50m steps.

* 5 million frames two step noisy double dueling dqn:
Near 8 hours to learn.
[![Watch the video](https://j.gifs.com/K1OL6r.gif)](https://youtu.be/f0sy4Fb3ZrQ)

* Random lucky game in Space Invaders after less then one hour learning:
[![Watch the video](https://j.gifs.com/D1RQE5.gif)](https://www.youtube.com/watch?v=LO0RL437rh4)


Thanks this article https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55 and Max's book. Helped me a lot 
