import models
import tr_helpers
import experience
import tensorflow as tf
import numpy as np
import collections
import time
import env_configurations
from collections import deque
from tensorboardX import SummaryWriter
import networks
from datetime import datetime
from tensorflow_utils import TensorFlowVariables

default_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-3,
    'STEPS_PER_EPOCH' : 20,
    'BATCH_SIZE' : 64,
    'EPSILON' : 0.8,
    'EPSILON_DECAY_FRAMES' : 1e5,
    'MIN_EPSILON' : 0.02,
    'NUM_EPOCHS_TO_COPY' : 1000,
    'NUM_STEPS_FILL_BUFFER' : 10000,
    'NAME' : 'DQN',
    'IS_DOUBLE' : False,
    'SCORE_TO_WIN' : 20,
    'REPLAY_BUFFER_TYPE' : 'normal', # 'prioritized'
    'REPLAY_BUFFER_SIZE' :100000,
    'PRIORITY_BETA' : 0.4,
    'PRIORITY_ALPHA' : 0.6,
    'BETA_DECAY_FRAMES' : 1e5,
    'MAX_BETA' : 1.0,
    'NETWORK' : models.AtariDQN(networks.dqn_network),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5, # 5 it is divider to calculate rewards, 5 lifes is one episode in breakout
    'STEPS_NUM' : 1,
    'ATOMS_NUM' : 51, # if atoms_num > 1 then it is distributional dqn https://arxiv.org/abs/1510.09142
    'V_MAX' : 10,
    'V_MIN' : -10,
    }



def kl_loss_log(y_True_log, y_pred_log):
    return tf.exp(y_True_log) * (y_True_log - y_pred_log)

class DQNAgent:
    def __init__(self, sess, base_name, observation_space, action_space, config):
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
        self.state_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))
        self.epsilon = self.config['epsilon']
        self.rewards_shaper = self.config['reward_shaper']
        self.epsilon_processor = tr_helpers.LinearValueProcessor(self.config['epsilon'], self.config['min_epsilon'], self.config['epsilon_decay_frames'])
        self.beta_processor = tr_helpers.LinearValueProcessor(self.config['priority_beta'], self.config['max_beta'], self.config['beta_decay_frames'])
        if self.env_name:
            self.env = env_configurations.configurations[self.env_name]['env_creator'](sess=sess)
        self.sess = sess
        self.steps_num = self.config['steps_num']
        self.states = deque([], maxlen=self.steps_num)
        self.is_prioritized = config['replay_buffer_type'] != 'normal'
        self.atoms_num = self.config['atoms_num']
        self.is_distributional = self.atoms_num > 1
        
        self.self_play = config.get('self_play', False)

        if self.is_distributional:
            self.v_min = self.config['v_min']
            self.v_max = self.config['v_max']
            self.delta_z = (self.v_max - self.v_min) /(self.atoms_num - 1)
            self.all_z = tf.range(self.v_min, self.v_max + self.delta_z, self.delta_z)

        if not self.is_prioritized:
            self.exp_buffer = experience.ReplayBuffer(config['replay_buffer_size'])
        else: 
            self.exp_buffer = experience.prioritizedReplayBuffer(config['replay_buffer_size'], config['priority_alpha'])
            self.sample_weights_ph = tf.placeholder(tf.float32, shape= [None,] , name='sample_weights')
        
        self.obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.state_shape , name = 'obs_ph')
        self.actions_ph = tf.placeholder(tf.int32, shape=[None,], name = 'actions_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None,], name = 'rewards_ph')
        self.next_obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.state_shape , name = 'next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None,], name = 'is_done_ph')
        self.is_not_done = 1 - self.is_done_ph
        self.name = base_name
        
        self.gamma = self.config['gamma']

        self.input_obs = self.obs_ph
        self.input_next_obs = self.next_obs_ph
 
        if observation_space.dtype == np.uint8:
            print('scaling obs')
            self.input_obs = tf.to_float(self.input_obs) / 255.0
            self.input_next_obs = tf.to_float(self.input_next_obs) / 255.0


        if self.atoms_num == 1:
            self.setup_qvalues(actions_num)
        else:
            self.setup_c51_qvalues(actions_num)
        
        self.saver = tf.train.Saver()
        self.assigns_op = [tf.assign(w_target, w_self, validate_shape=True) for w_self, w_target in zip(self.weights, self.target_weights)]
        self.variables = TensorFlowVariables(self.qvalues, self.sess)
        if self.env_name:
            sess.run(tf.global_variables_initializer())
        self._reset()

    def get_weights(self):
        return self.variables.get_flat()
    
    def set_weights(self, weights):
        return self.variables.set_flat(weights)

    def update_epoch(self):
        return self.sess.run([self.update_epoch_op])[0]

    def setup_c51_qvalues(self, actions_num):
        self.qvalues_c51 = self.network('agent', self.obs_ph, actions_num)

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

        if self.config['is_double'] == True:
            config = {
                'name' : 'agent',
                'inputs' : self.input_next_obs,
                'actions_num' : actions_num,
            }
            self.next_qvalues = tf.stop_gradient(self.network(config, reuse=True))

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')


        self.current_action_qvalues = tf.reduce_sum(tf.one_hot(self.actions_ph, actions_num) * self.qvalues, reduction_indices = 1)

        if self.config['is_double'] == True:
            self.next_selected_actions = tf.argmax(self.next_qvalues, axis = 1)
            self.next_selected_actions_onehot = tf.one_hot(self.next_selected_actions, actions_num)
            self.next_state_values_target = tf.stop_gradient( tf.reduce_sum( self.target_qvalues * self.next_selected_actions_onehot , reduction_indices=[1,] ))
        else:
            self.next_state_values_target = tf.stop_gradient(tf.reduce_max(self.target_qvalues, reduction_indices=1))

        self.gamma_step = tf.constant(self.gamma**self.steps_num, dtype=tf.float32)
        self.reference_qvalues = self.rewards_ph + self.gamma_step *self.is_not_done * self.next_state_values_target

        if self.is_prioritized:
            # we need to return l1 loss to update priority buffer
            self.abs_errors = tf.abs(self.current_action_qvalues - self.reference_qvalues) + 1e-5
            # the same as multiply gradients later (other way is used in different examples over internet) 
            self.td_loss = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues, reduction=tf.losses.reduction.NONE) * self.sample_weights_ph
            self.td_loss_mean = tf.reduce_mean(self.td_loss) 
        else:
            self.td_loss_mean = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues, reduction=tf.losses.Reduction.MEAN)

        self.reg_loss = tf.losses.get_regularization_loss()
        self.td_loss_mean += self.reg_loss
        self.learning_rate = self.config['learning_rate']
        if self.env_name:
            self.train_step = tf.train.AdamOptimizer(self.learning_rate * self.lr_multiplier).minimize(self.td_loss_mean, var_list=self.weights)

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def _reset(self):
        self.states.clear()
        if self.env_name:
            self.state = self.env.reset()
        self.total_reward = 0.0
        self.total_shaped_reward = 0.0
        self.step_count = 0

    def get_qvalues(self, state):
        return self.sess.run(self.qvalues, {self.obs_ph: state})


    def get_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self.get_qvalues([state])
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
            reward = reward# * (1 - is_done)
 
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


    def load_weigths_into_target_network(self):
        self.sess.run(self.assigns_op)

    def sample_batch(self, exp_replay, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch  = exp_replay.sample(batch_size)
        return {
        self.obs_ph:obs_batch, self.actions_ph:act_batch, self.rewards_ph:reward_batch, 
        self.is_done_ph:is_done_batch, self.next_obs_ph:next_obs_batch
        }

    def sample_prioritized_batch(self, exp_replay, batch_size, beta):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,  sample_weights, sample_idxes = exp_replay.sample(batch_size, beta)
        batch = { self.obs_ph:obs_batch, self.actions_ph:act_batch, self.rewards_ph:reward_batch, 
        self.is_done_ph:is_done_batch, self.next_obs_ph:next_obs_batch, self.sample_weights_ph: sample_weights }
        return [batch , sample_idxes]

    def train(self):
        last_mean_rewards = -100500
        epoch_num = 0
        frame = 0
        update_time = 0
        play_time = 0

        start_time = time.time()
        total_time = 0
        self.load_weigths_into_target_network()
        for _ in range(0, self.config['num_steps_fill_buffer']):
            self.play_steps(self.steps_num, self.epsilon)

        steps_per_epoch = self.config['steps_per_epoch']
        num_epochs_to_copy = self.config['num_epochs_to_copy']
        batch_size = self.config['batch_size']
        lives_reward = self.config['lives_reward']
        episodes_to_log = self.config['episodes_to_log']
        frame = 0
        play_time = 0
        update_time = 0
        rewards = []
        shaped_rewards = []
        steps = []
        while True:
            epoch_num = self.update_epoch()
            t_play_start = time.time()
            self.epsilon = self.epsilon_processor(frame)
            self.beta = self.beta_processor(frame)

            for _ in range(0, steps_per_epoch):
                reward, shaped_reward, step = self.play_steps(self.steps_num, self.epsilon)
                if reward != None:
                    self.game_lengths.append(step)
                    self.game_rewards.append(reward)
                    #shaped_rewards.append(shaped_reward)

            t_play_end = time.time()
            play_time += t_play_end - t_play_start
            
            # train
            frame = frame + steps_per_epoch
            
            t_start = time.time()
            if self.is_prioritized:
                batch, idxes = self.sample_prioritized_batch(self.exp_buffer, batch_size=batch_size, beta = self.beta)
                _, loss_t, errors_update, lr_mul = self.sess.run([self.train_step, self.td_loss_mean, self.abs_errors], batch)
                self.exp_buffer.update_priorities(idxes, errors_update)
            else:
                batch = self.sample_batch(self.exp_buffer, batch_size=batch_size)
                _, loss_t, lr_mul = self.sess.run([self.train_step, self.td_loss_mean, self.lr_multiplier], batch)
            t_end = time.time()
            update_time += t_end - t_start
            total_time += update_time
            if frame % 1000 == 0:
                sum_time = update_time + play_time
                print('frames per seconds: ', 1000 / (sum_time))
                self.writer.add_scalar('performance/fps', 1000 / sum_time, frame)
                self.writer.add_scalar('performance/upd_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/td_loss', loss_t, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/lr', self.learning_rate*lr_mul, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self.writer.add_scalar('info/epsilon', self.epsilon, frame)
                if self.is_prioritized:
                    self.writer.add_scalar('beta', self.beta, frame)
                    
                update_time = 0
                play_time = 0
                num_games = len(self.game_rewards)
                if num_games > 10:
                    d = num_games / lives_reward
                    mean_rewards = np.sum(self.game_rewards) / d 
                    mean_lengths = np.sum(self.game_lengths) / d
                    self.writer.add_scalar('rewards/mean', mean_rewards, frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/mean', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if mean_rewards > last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                        if last_mean_rewards > self.config['score_to_win']:
                            print('network won!')
                            return last_mean_rewards, epoch_num   

                    if self.self_play and num_games == self.games_to_track:
                        if mean_rewards > self.self_play['score_to_update']:
                            print('updating weights')
                            self.game_rewards.clear()
                            self.game_lengths.clear()
                            last_mean_rewards = -100500
                            self.env.update_weights(self.get_weights())
                
                
                #clear_output(True)
            # adjust agent parameters
            if frame % num_epochs_to_copy == 0:
                self.load_weigths_into_target_network()
            
            if epoch_num >= self.max_epochs:
                print('Max epochs reached')
                self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(np.sum(self.game_rewards) *  lives_reward / len(self.game_rewards)))
                return last_mean_rewards, epoch_num            

