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
        self.states = deque([], maxlen=self.steps_num)
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
        self.sess.run(tf.global_variables_initializer())
    
    def setup_qvalues(self, actions_num):
        config = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'actions_num' : actions_num,
        }
        self.qvalues = self.network(config, reuse=False)
        config = {
            'name' : 'target',
            'inputs' : self.input_next_obs,
            'actions_num' : actions_num,
        }
        self.target_qvalues = tf.stop_gradient(self.network(config, reuse=False))
        
    def play_episode(self, epsilon=0.0):
        mb_obs = []
        mb_rewards = []
        mb_actions = []
        mb_avail_actions = []
        mb_dones = []
        mb_states = []
        step_count = 0
        
        obs = self.env.reset()
        obs = np.reshape(obs, ((self.n_agents,) + self.obs_shape))
        mb_obs.append(obs)
        mb_states.append(self.env.get_state())
        avail_acts = self.env.get_action_mask()
        mb_avail_actions.append(avail_acts)
        while True:
            step_count += 1
            step_act = self.get_action(obs, avail_acts, epsilon)
            next_obs, rewards, dones, _ = self.env.step(step_act)
            mb_actions.append(step_act)
            mb_obs.append(next_obs)
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_states.append(self.env.get_state())
            
            obs = next_obs
            obs = np.reshape(obs, ((self.n_agents,) + self.obs_shape))
            avail_acts = self.env.get_action_mask()
            mb_avail_actions.append(avail_acts)
            
            if all(dones) or self.steps_num < step_count:
                break
            
            
    def get_action(self, obs, avail_acts, epsilon=0.0):
        print(obs.shape)
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
        self.play_episode()
        
        
        
        
        
        
        
        

