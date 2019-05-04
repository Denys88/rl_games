import networks
import tr_helpers
import experience
import tensorflow as tf
import numpy as np
import collections
import time
from collections import deque
from tensorboardX import SummaryWriter
import ray


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
    'NETWORK' : networks.AtariDQN(),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5, # 5 it is divider to calculate rewards, 5 lifes is one episode in breakout
    'STEPS_NUM' : 1,
    'ATOMS_NUM' : 51, # if atoms_num > 1 then it is distributional dqn https://arxiv.org/abs/1510.09142
    'V_MAX' : 10,
    'V_MIN' : -10,
    }





class DQNAgent:
    def __init__(self, env, sess, env_name, config = default_config):
        observation_shape = env.observation_space.shape
        
        actions_num = env.action_space.n
        self.network = config['NETWORK']
        self.config = config
        self.state_shape = observation_shape
        self.actions_num = actions_num
        print(self.state_shape)
        self.writer = SummaryWriter()
        self.epsilon = self.config['EPSILON']
        self.rewards_shaper = self.config['REWARD_SHAPER']
        self.epsilon_processor = tr_helpers.LinearValueProcessor(self.config['EPSILON'], self.config['MIN_EPSILON'], self.config['EPSILON_DECAY_FRAMES'])
        self.beta_processor = tr_helpers.LinearValueProcessor(self.config['PRIORITY_BETA'], self.config['MAX_BETA'], self.config['BETA_DECAY_FRAMES'])
        self.env = env
        self.sess = sess
        self.steps_num = self.config['STEPS_NUM']
        self.states = deque([], maxlen=self.steps_num)
        self.is_prioritized = config['REPLAY_BUFFER_TYPE'] != 'normal'
        self.atoms_num = self.config['ATOMS_NUM']
        self.is_distributional = self.atoms_num > 1
    
        if self.is_distributional:
            self.v_min = self.config['V_MIN']
            self.v_max = self.config['V_MAX']
            self.delta_z = (self.v_max - self.v_min) /(self.atoms_num - 1)
            self.all_z = tf.range(self.v_min, self.v_max + self.delta_z, self.delta_z)

        if not self.is_prioritized:
            self.exp_buffer = experience.ReplayBuffer(config['REPLAY_BUFFER_SIZE'])
        else: 
            self.exp_buffer = experience.PrioritizedReplayBuffer(config['REPLAY_BUFFER_SIZE'], config['PRIORITY_ALPHA'])
            self.sample_weights_ph = tf.placeholder(tf.float32, shape= [None,] , name='sample_weights')
        
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'obs_ph')
        self.actions_ph = tf.placeholder(tf.int32, shape=[None,], name = 'actions_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None,], name = 'rewards_ph')
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None,], name = 'is_done_ph')
        self.is_not_done = 1 - self.is_done_ph
        self.env_name = env_name
        self._reset()
        self.gamma = self.config['GAMMA']
        if self.atoms_num == 1:
            self.setup_qvalues(actions_num)
        else:
            self.setup_c51_qvalues(actions_num)
        
        self.saver = tf.train.Saver()
        self.assigns_op = [tf.assign(w_target, w_self, validate_shape=True) for w_self, w_target in zip(self.weights, self.target_weights)]

        sess.run(tf.global_variables_initializer())

    def setup_c51_qvalues(self, actions_num):
        self.qvalues_c51 = self.network('agent', self.obs_ph, actions_num)


    def setup_qvalues(self, actions_num):
        self.qvalues = self.network('agent', self.obs_ph, actions_num)
        self.target_qvalues = tf.stop_gradient(self.network('target', self.next_obs_ph, actions_num))

        if self.config['IS_DOUBLE'] == True:
            self.next_qvalues = tf.stop_gradient(self.network('agent', self.next_obs_ph, actions_num, reuse=True))

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')


        self.current_action_qvalues = tf.reduce_sum(tf.one_hot(self.actions_ph, actions_num) * self.qvalues, reduction_indices = 1)

        if self.config['IS_DOUBLE'] == True:
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
            self.td_loss = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues, reduction=tf.losses.Reduction.NONE) * self.sample_weights_ph
            self.td_loss_mean = tf.reduce_mean(self.td_loss) 
        else:
            self.td_loss_mean = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues, reduction=tf.losses.Reduction.MEAN)

        self.learning_rate = self.config['LEARNING_RATE']
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.td_loss_mean, var_list=self.weights)

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def _reset(self):
        self.states.clear()
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
        self.load_weigths_into_target_network()
        for _ in range(0, self.config['NUM_STEPS_FILL_BUFFER']):
            self.play_steps(self.steps_num, self.epsilon)

        STEPS_PER_EPOCH = self.config['STEPS_PER_EPOCH']
        NUM_EPOCHS_TO_COPY = self.config['NUM_EPOCHS_TO_COPY']
        BATCH_SIZE = self.config['BATCH_SIZE']
        LIVES_REWARD = self.config['LIVES_REWARD']
        EPISODES_TO_LOG = self.config['EPISODES_TO_LOG']
        frame = 0
        play_time = 0
        update_time = 0
        rewards = []
        shaped_rewards = []
        steps = []
        while True:
            t_play_start = time.time()
            self.epsilon = self.epsilon_processor(frame)
            self.beta = self.beta_processor(frame)

            for _ in range(0, STEPS_PER_EPOCH):
                reward, shaped_reward, step = self.play_steps(self.steps_num, self.epsilon)
                if reward != None:
                    steps.append(step)
                    rewards.append(reward)
                    shaped_rewards.append(shaped_reward)

            t_play_end = time.time()
            play_time += t_play_end - t_play_start
            
            # train
            frame = frame + STEPS_PER_EPOCH
            
            t_start = time.time()
            if self.is_prioritized:
                batch, idxes = self.sample_prioritized_batch(self.exp_buffer, batch_size=BATCH_SIZE, beta = self.beta)
                _, loss_t, errors_update = self.sess.run([self.train_step, self.td_loss_mean, self.abs_errors], batch)
                self.exp_buffer.update_priorities(idxes, errors_update)
            else:
                batch = self.sample_batch(self.exp_buffer, batch_size=BATCH_SIZE)
                _, loss_t = self.sess.run([self.train_step, self.td_loss_mean], batch)
            t_end = time.time()
            update_time += t_end - t_start

            if frame % 1000 == 0:
                print('Frames per seconds: ', 1000 / (update_time + play_time))
                self.writer.add_scalar('Frames per seconds: ', 1000 / (update_time + play_time), frame)
                self.writer.add_scalar('upd_time', update_time, frame)
                self.writer.add_scalar('play_time', play_time, frame)
                self.writer.add_scalar('loss', loss_t, frame)
                self.writer.add_scalar('epsilon', self.epsilon, frame)
                if self.is_prioritized:
                    self.writer.add_scalar('beta', self.beta, frame)
                    
                update_time = 0
                play_time = 0
            
            if len(rewards) == EPISODES_TO_LOG:
                d = EPISODES_TO_LOG / LIVES_REWARD
                mean_reward = np.sum(rewards) / d
                mean_shaped_reward = np.sum(shaped_rewards) / d
                mean_steps = np.sum(steps) / d 
                rewards = []
                shaped_rewards = []
                steps = []
                if mean_reward > last_mean_rewards:
                    print('saving next best rewards: ', mean_reward)
                    last_mean_rewards = mean_reward
                    self.save("./nn/" + self.config['NAME'] + self.env_name)
                    if last_mean_rewards > self.config['SCORE_TO_WIN']:
                        print('Network won!')
                        return

                self.writer.add_scalar('steps', mean_steps, frame)
                self.writer.add_scalar('reward', mean_reward, frame)
                self.writer.add_scalar('shaped_reward', mean_shaped_reward, frame)
                
                
                #clear_output(True)
            # adjust agent parameters
            if frame % NUM_EPOCHS_TO_COPY == 0:
                self.load_weigths_into_target_network()

