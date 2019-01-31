# dqn_atari
Tensorflow implementations of the DQN atari.

Double dueling DQN vs DQN with the same parameters:
![alt text](https://github.com/Denys88/dqn_atari/blob/master/pictures/dqn_vs_dddqn.png)
As it was very simple game we get almost the same results,  90 minutes to learn to the 20 reward.
DQN has more optimistic Q value estimations.

3 million frames noisy double dueling dqn:
Near 10 hours to learn.
breakout_noisy_dddqn_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-4,
    'STEPS_PER_EPOCH' : 8,
    'BATCH_SIZE' : 32 *2,
    'EPSILON' : 0.0,
    'MIN_EPSILON' : 0.00,
    'EPSILON_DECAY_FRAMES' : 100000,
    'NUM_EPOCHS_TO_COPY' : 1000,
    'EPS_DECAY_RATE' : 0.0,
    'NAME' : 'NDDDQN',
    'IS_DOUBLE' : True,
    'DUELING_TYPE' : 'AVERAGE',
    'SCORE_TO_WIN' : 400,
    'NUM_STEPS_FILL_BUFFER' : 10000,
    'NETWORK' : networks.AtariNoisyDuelingDQN()
    }
    

[![Watch the video](https://j.gifs.com/oVRzRz.gif)](https://youtu.be/f0sy4Fb3ZrQ)
