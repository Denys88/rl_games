import networks
import tensorflow as tf
import numpy as np
import collections
import time
from tensorboardX import SummaryWriter

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

default_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-3,
    'STEPS_PER_EPOCH' : 20,
    'BATCH_SIZE' : 64,
    'EPSILON' : 0.8,
    'MIN_EPSILON' : 0.02,
    'NUM_EPOCHS_TO_COPY' : 1000,
    'NUM_STEPS_FILL_BUFFER' : 10000,
    'EPS_DECAY_RATE' : 0.99,
    'NAME' : 'DQN',
    'IS_DOUBLE' : False,
    'SCORE_TO_WIN' : 20,
    'NETWORK' : networks.AtariDQN()
    }


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)



class DQNAgent:
    def __init__(self, env, sess, exp_buffer, env_name, config = default_config):
        observation_shape = env.observation_space.shape
        actions_num = env.action_space.n
        self.network = config['NETWORK']
        self.config = config
        self.state_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter()
        self.epsilon = self.config['EPSILON']
        self.env = env
        self.sess = sess
        self.exp_buffer = exp_buffer
        self._reset()
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'obs_ph')
        self.actions_ph = tf.placeholder(tf.int32, shape=[None], name = 'actions_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None], name = 'rewards_ph')
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None], name = 'is_done_ph')
        self.is_not_done = 1 - self.is_done_ph
        self.env_name = env_name
        self.step_count = 0
  
        self.qvalues = self.network('agent', self.obs_ph, actions_num)
        self.target_qvalues = self.network('target', self.next_obs_ph, actions_num)
        if self.config['IS_DOUBLE'] == True:
            self.next_qvalues = self.network('agent', self.next_obs_ph, actions_num, reuse=True)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')


        self.current_action_qvalues = tf.reduce_sum(tf.one_hot(self.actions_ph, actions_num) * self.qvalues, axis=1)
        
        if self.config['IS_DOUBLE'] == True:
            self.next_selected_actions = tf.argmax(self.next_qvalues, axis=1)
            self.next_selected_actions_onehot = tf.one_hot(self.next_selected_actions, actions_num)
            self.next_state_values_target = tf.stop_gradient( tf.reduce_sum( self.target_qvalues * self.next_selected_actions_onehot , reduction_indices=[1,] ))
        else:
            self.next_state_values_target = tf.reduce_max(self.target_qvalues, axis=-1)


        GAMMA = self.config['GAMMA']
        
        LEARNING_RATE = self.config['LEARNING_RATE']
        self.reference_qvalues = self.rewards_ph + self.is_not_done * GAMMA * self.next_state_values_target
        self.td_loss = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues)
        #td_loss = (self.current_action_qvalues - self.reference_qvalues) ** 2
        self.td_loss_mean = tf.reduce_mean(self.td_loss)

        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.td_loss_mean, var_list=self.weights)
        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.step_count = 0

    def get_qvalues(self, state):
        return self.sess.run(self.qvalues, {self.obs_ph: state})

    def play_step(self, epsilon=0.0):
        done_reward = None
        done_steps = None
        action = 0
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self.get_qvalues([self.state])
            action = np.argmax(qvals)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            done_steps = self.step_count
            self._reset()
        return done_reward, done_steps

    def load_weigths_into_target_network(self):
        assigns = []
        for w_self, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, w_self, validate_shape=True))
        self.sess.run(assigns)

    def sample_batch(self, exp_replay, batch_size):
        obs_batch, act_batch, reward_batch, is_done_batch, next_obs_batch = exp_replay.sample(batch_size)
        return {
        self.obs_ph:obs_batch, self.actions_ph:act_batch, self.rewards_ph:reward_batch, 
        self.is_done_ph:is_done_batch, self.next_obs_ph:next_obs_batch
        }

    def evaluate(self, env,  n_games=3, t_max=10000):
        rewards = []
        steps = []
        max_qvals = []
        for _ in range(n_games):
            s = env.reset()
            reward = 0
            for step in range(t_max):
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    qvalues = self.get_qvalues([s])
                    max_qvals = np.max(qvalues)
                    action = np.argmax(qvalues)
                s, r, done, _ = env.step(action)
                reward += r
                if done: 
                    steps.append(step)
                    break
                
            rewards.append(reward)
        return np.mean(rewards), np.mean(steps), np.mean(max_qvals)

    def train(self):
        last_mean_rewards = -100500
        self.load_weigths_into_target_network()
        for k in range(0, self.config['NUM_STEPS_FILL_BUFFER']):
            self.play_step(self.epsilon)

        STEPS_PER_EPOCH = self.config['STEPS_PER_EPOCH']
        MIN_EPSILON = self.config['MIN_EPSILON']
        EPS_DECAY_RATE = self.config['EPS_DECAY_RATE']
        NUM_EPOCHS_TO_COPY = self.config['NUM_EPOCHS_TO_COPY']
        BATCH_SIZE = self.config['BATCH_SIZE']
        frame = 0
        play_time = 0
        update_time = 0
        rewards = []
        steps = []
        while True:
            
            t_play_start = time.time()
            for k in range(0, STEPS_PER_EPOCH):
                reward, step = self.play_step(self.epsilon)
                if reward != None:
                    steps.append(step)
                    rewards.append(reward)
            t_play_end = time.time()
            play_time += t_play_end - t_play_start
            t_start = time.time()
            frame += STEPS_PER_EPOCH
            _, loss_t = self.sess.run([self.train_step, self.td_loss], self.sample_batch(self.exp_buffer, batch_size=BATCH_SIZE))
            t_end = time.time()
            update_time += t_end - t_start

            if frame % 1000 == 0:
                print('Frames per seconds: ', 1000 / update_time)
                self.writer.add_scalar('Frames per seconds: ', 1000 / update_time, frame)
                self.writer.add_scalar('upd_time', update_time, frame)
                self.writer.add_scalar('play_time', play_time, frame)
                update_time = 0
                play_time = 0

            if len(rewards) == 10:
                print(frame)
                mean_reward = np.mean(rewards)
                mean_steps = np.mean(steps)
                rewards = []
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
                self.writer.add_scalar('loss', loss_t, frame)
                self.writer.add_scalar('epsilon', self.epsilon, frame)
                
                
                #clear_output(True)
            # adjust agent parameters
            if frame % NUM_EPOCHS_TO_COPY == 0:
                self.epsilon = max(self.epsilon * EPS_DECAY_RATE, MIN_EPSILON)
                self.load_weigths_into_target_network()

