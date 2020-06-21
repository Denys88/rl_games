import tensorflow as tf
import algos_tf14.models
from common import tr_helpers, experience, env_configurations
import numpy as np
import collections
import time
from collections import deque
from tensorboardX import SummaryWriter
from datetime import datetime
from algos_tf14.tensorflow_utils import TensorFlowVariables
from common.categorical import CategoricalQ

class VDNAgent:
    def __init__(self, sess, base_name, observation_space, action_space, config, logger):
        observation_shape = observation_space.shape
        actions_num = action_space.n
        self.config = config
        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.is_polynom_decay_lr = config['lr_schedule'] == 'polynom_decay'
        self.is_exp_decay_lr = config['lr_schedule'] == 'exp_decay'
        self.lr_multiplier = tf.constant(1, shape=(), dtype=tf.float32)
        self.learning_rate_ph = tf.placeholder('float32', (), name = 'lr_ph')
        self.games_to_track = tr_helpers.get_or_default(config, 'games_to_track', 100)
        self.max_epochs = tr_helpers.get_or_default(self.config, 'max_epochs', 1e6)

        self.game_rewards = deque([], maxlen=self.games_to_track)
        self.game_lengths = deque([], maxlen=self.games_to_track)

        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)
        self.update_epoch_op = self.epoch_num.assign(self.epoch_num + 1)
        self.current_lr = self.learning_rate_ph

        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']
        if self.is_polynom_decay_lr:
            self.lr_multiplier = tf.train.polynomial_decay(1.0, global_step=self.epoch_num, decay_steps=self.max_epochs, end_learning_rate=0.001, power=tr_helpers.get_or_default(config, 'decay_power', 1.0))
        if self.is_exp_decay_lr:
            self.lr_multiplier = tf.train.exponential_decay(1.0, global_step=self.epoch_num, decay_steps=self.max_epochs,  decay_rate = config['decay_rate'])
            
        self.env_name = config['env_name']
        self.network = config['network']
        self.obs_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))
        self.epsilon = self.config['epsilon']
        self.rewards_shaper = self.config['reward_shaper']
        self.epsilon_processor = tr_helpers.LinearValueProcessor(self.config['epsilon'], self.config['min_epsilon'], self.config['epsilon_decay_frames'])
        self.beta_processor = tr_helpers.LinearValueProcessor(self.config['priority_beta'], self.config['max_beta'], self.config['beta_decay_frames'])
        if self.env_name:
            self.env = env_configurations.configurations[self.env_name]['env_creator'](name=config['name'])
        self.sess = sess
        self.steps_num = self.config['steps_num']
        
        self.obs_act_rew = deque([], maxlen=self.steps_num)
        
        self.is_prioritized = config['replay_buffer_type'] != 'normal'
        self.atoms_num = self.config['atoms_num']
        assert self.atoms_num == 1
        
        self.state_shape = (self.env.env_info['state_shape'],)
        self.n_agents = self.env.env_info['n_agents']
        
        if not self.is_prioritized:
            self.exp_buffer = experience.ReplayBuffer(config['replay_buffer_size'])
        else: 
            self.exp_buffer = experience.PrioritizedReplayBuffer(config['replay_buffer_size'], config['priority_alpha'])
            self.sample_weights_ph = tf.placeholder(tf.float32, shape= [None,] , name='sample_weights')
        
        self.obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.obs_shape , name = 'obs_ph')
        self.state_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.state_shape , name = 'state_ph')
        self.actions_ph = tf.placeholder(tf.int32, shape=[None,], name = 'actions_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None,], name = 'rewards_ph')
        self.next_obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.obs_shape , name = 'next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None,], name = 'is_done_ph')
        self.is_not_done = 1 - self.is_done_ph
        self.name = base_name
        
        self.gamma = self.config['gamma']
        self.gamma_step = self.gamma**self.steps_num
        self.grad_norm = config['grad_norm']
        self.input_obs = self.obs_ph
        self.input_next_obs = self.next_obs_ph
        if observation_space.dtype == np.uint8:
            print('scaling obs')
            self.input_obs = tf.to_float(self.input_obs) / 255.0
            self.input_next_obs = tf.to_float(self.input_next_obs) / 255.0
        self.setup_qvalues(actions_num)
        
        if self.env_name:
            self.sess.run(tf.global_variables_initializer())
#         self.reg_loss = tf.losses.get_regularization_loss()
#         self.td_loss_mean += self.reg_loss
#         self.learning_rate = self.config['learning_rate']
#         self.train_step = tf.train.AdamOptimizer(self.learning_rate * self.lr_multiplier).minimize(self.td_loss_mean, var_list=self.weights)        

#         self.saver = tf.train.Saver()
#         self.assigns_op = [tf.assign(w_target, w_self, validate_shape=True) for w_self, w_target in zip(self.weights, self.target_weights)]
#         self.variables = TensorFlowVariables(self.qvalues, self.sess)
        if self.env_name:
            sess.run(tf.global_variables_initializer())
        self._reset()
    
    def setup_qvalues(self, actions_num):
        config = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'actions_num' : actions_num,
        }
        #(n_agents, n_actions)
        self.qvalues = self.network(config, reuse=False)
        config = {
            'name' : 'target',
            'inputs' : self.input_next_obs,
            'actions_num' : actions_num,
        }
        self.target_qvalues = tf.stop_gradient(self.network(config, reuse=False))
        
        if self.config['is_double'] == True:
            config = {
                'name' : 'agent',
                'inputs' : self.input_next_obs,
                'actions_num' : actions_num,
            }
            self.next_qvalues = tf.stop_gradient(self.network(config, reuse=True))

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        
        #(n_agents, 1)
        self.current_action_qvalues = tf.reduce_sum(tf.one_hot(self.actions_ph, actions_num) * self.qvalues, reduction_indices = 1)
        
        if self.config['is_double'] == True:
            self.next_selected_actions = tf.argmax(self.next_qvalues, axis = 1)
            self.next_selected_actions_onehot = tf.one_hot(self.next_selected_actions, actions_num)
            self.next_state_values_target = tf.stop_gradient( tf.reduce_sum( self.target_qvalues * self.next_selected_actions_onehot , reduction_indices=[1,] ))
        else:
            self.next_state_values_target = tf.stop_gradient(tf.reduce_max(self.target_qvalues, reduction_indices=1))
        
    def play_steps(self, steps, epsilon=0.0):
        done_reward = None
        done_shaped_reward = None
        done_steps = None
        steps_rewards = 0
        cur_gamma = 1
        cur_obs_act_rew_len = len(self.obs_act_rew)

        # always break after one
        while True:
            if cur_obs_act_rew_len > 0:
                obs = self.obs_act_rew[-1][0]
            else:
                obs = self.current_obs
            obs = np.reshape(obs, ((self.n_agents,) + self.obs_shape))

            action = self.get_action(obs, self.env.get_action_mask(), epsilon)
            print(action)
            print(self.sess.run(self.qvalues, {self.obs_ph: obs}))
            print(self.sess.run(self.next_state_values_target, {self.obs_ph: obs, self.actions_ph: action}))
            new_obs, reward, is_done, _ = self.env.step(action)
            #reward = reward * (1 - is_done)
 
            self.step_count += 1
            self.total_reward += reward
            shaped_reward = self.rewards_shaper(reward)
            self.total_shaped_reward += shaped_reward
            self.obs_act_rew.append([new_obs, action, shaped_reward])

            if len(self.obs_act_rew) < steps:
                break

            for i in range(steps):
                sreward = self.obs_act_rew[i][2]
                steps_rewards += sreward * cur_gamma
                cur_gamma = cur_gamma * self.gamma

            next_obs, current_action, _ = self.obs_act_rew[0]
            self.exp_buffer.add(self.current_obs, current_action, steps_rewards, new_obs, is_done)
            self.current_obs = next_obs
            break
            
        if all(is_done):
            done_reward = self.total_reward
            done_steps = self.step_count
            done_shaped_reward = self.total_shaped_reward
            self._reset()
        return done_reward, done_shaped_reward, done_steps
                
    def _reset(self):
        self.obs_act_rew.clear()
        if self.env_name:
            self.current_obs = self.env.reset()
        self.total_reward = 0.0
        self.total_shaped_reward = 0.0
        self.step_count = 0
        
    def get_action(self, obs, avail_acts, epsilon=0.0):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self.get_qvalues(obs)
            qvals[avail_acts == False] = -9999999
            action = np.argmax(qvals, axis=1)
        return action  
    
    def get_qvalues(self, obs):
        return self.sess.run(self.qvalues, {self.obs_ph: obs})
        
    def train(self):
        for _ in range(5):
            self.play_steps(steps=3)
            
        
        
        
        
        
        
        
        

