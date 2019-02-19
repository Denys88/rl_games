import networks
import tr_helpers
import experience
import tensorflow as tf
import numpy as np
import collections
import time
from collections import deque
from tensorboardX import SummaryWriter


default_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-3,
    'BATCH_SIZE' : 64,
    'EPSILON' : 0.8,
    'EPSILON_DECAY_FRAMES' : 1e5,
    'NAME' : 'A2C',
    'SCORE_TO_WIN' : 20,
    'NETWORK' : networks.CartPoleA2C(),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5,
    'STEPS_NUM' : 1
}



class DQNAgent:
    def __init__(self, env, sess, env_name, config = default_config):
        observation_shape = env.observation_space.shape
        actions_num = env.action_space.n
        
        self.network = config['NETWORK']
        self.config = config
        self.state_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter()
        self.epsilon = self.config['EPSILON']
        self.rewards_shaper = self.config['REWARD_SHAPER']
        self.env = env
        self.sess = sess
        self.steps_num = self.config['STEPS_NUM']
        self.states = deque([], maxlen=self.steps_num)
    
    

    def get_action_distribution(self, state):
        return self.sess.run(self.soft_max_actions, {self.obs_ph: state})


    def get_action(self, state, is_determenistic = False):
 
        actions = self.get_action_distribution([state])
            action = np.argmax(qvals)
        return action      

    def play_steps(self, steps, epsilon=0.0):
        done_reward = None
        done_shaped_reward = None
        done_steps = None
        steps_rewards = 0
        cur_gamma = 1
        cur_states_len = len(self.states)
        # always break after one
        while True:
            if cur_states_len > 0:
                state = self.states[-1][0]
            else:
                state = self.state
            action = self.get_action(state, epsilon)
            new_state, reward, is_done, _ = self.env.step(action)
            reward = reward * (1 - is_done)
 
            self.step_count += 1
            self.total_reward += reward
            shaped_reward = self.rewards_shaper(reward)
            self.total_shaped_reward += shaped_reward
            self.states.append([new_state, action, shaped_reward])

            if len(self.states) < steps:
                break

            for i in range(steps):
                sreward = self.states[i][2]
                steps_rewards += sreward * cur_gamma
                cur_gamma = cur_gamma * self.gamma

            next_state, current_action, _ = self.states[0]
            self.exp_buffer.add(self.state, current_action, steps_rewards, new_state, is_done)
            self.state = next_state
            break

        if is_done:
            done_reward = self.total_reward
            done_steps = self.step_count
            done_shaped_reward = self.total_shaped_reward
            self._reset()
        return done_reward, done_shaped_reward, done_steps