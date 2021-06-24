## Old Tensorflow results

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