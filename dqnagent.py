from networks import dqn_network
import tensorflow as tf
import numpy as np
import collections
from tensorboardX import SummaryWriter
from wrappers import make_env
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

default_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-3,
    'STEPS_PER_EPOCH' : 20,
    'EPSILON' : 0.8,
    'MIN_EPSILON' : 0.02,
    'NUM_EPOCHS' : 3 * 10**5,
    'EPS_DECAY_RATE' : 0.99,
    'NAME' : 'DQN',
    'IS_DDQN' : False
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
        self.qvalues = dqn_network('agent', self.obs_ph, actions_num)
        self.target_qvalues = dqn_network('target', self.next_obs_ph, actions_num)
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')


        self.current_action_qvalues = tf.reduce_sum(tf.one_hot(self.actions_ph, actions_num) * self.qvalues, axis=1)

        if self.config['IS_DDQN'] == True:
            self.next_q_values_agent = tf.stop_gradient(dqn_network('agent', self.next_obs_ph, actions_num, reuse=True))
            self.next_selected_actions = tf.argmax(self.next_q_values_agent, dimension=1)
            self.next_selected_actions_onehot = tf.one_hot(self.next_selected_actions, actions_num)
            self.next_state_values_target = tf.stop_gradient( tf.reduce_sum( self.target_qvalues * self.next_selected_actions_onehot , reduction_indices=[1,] ))
        else:
            self.next_state_values_target = tf.reduce_max(self.target_qvalues, axis=-1)


        GAMMA = self.config['GAMMA']
        
        LEARNING_RATE = self.config['LEARNING_RATE']
        self.reference_qvalues = self.rewards_ph + self.is_not_done * GAMMA * self.next_state_values_target

        td_loss = (self.current_action_qvalues - self.reference_qvalues) ** 2
        self.td_loss = tf.reduce_mean(td_loss)

        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.td_loss, var_list=self.weights)
        self.saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def get_qvalues(self, state):
        return self.sess.run(self.qvalues, {self.obs_ph: state})

    def play_step(self, epsilon=0.0):
        done_reward = None
        action = 0
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self.get_qvalues([self.state])
            action = np.argmax(qvals)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

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
                    max_qvals = np.max(qvalues);
                    action = np.argmax(qvalues)
                s, r, done, _ = env.step(action)
                reward += r
                if done: 
                    steps.append(step)
                    break
                
            rewards.append(reward)
        return np.mean(rewards), np.mean(steps), np.mean(max_qvals)

    def train(self):
        self.load_weigths_into_target_network()
        for k in range(0, 10000):
            self.play_step(self.epsilon)

        NUM_EPOCHS = self.config['NUM_EPOCHS']
        STEPS_PER_EPOCH = self.config['STEPS_PER_EPOCH']
        MIN_EPSILON = self.config['MIN_EPSILON']
        EPS_DECAY_RATE = self.config['EPS_DECAY_RATE']
        for i in range(NUM_EPOCHS):
            for k in range(0, STEPS_PER_EPOCH):
                self.play_step(self.epsilon)
            # train

            _, loss_t = self.sess.run([self.train_step, self.td_loss], self.sample_batch(self.exp_buffer, batch_size=64))
            if i % 500 == 0:
                print(i)
                mean_reward, mean_steps, mean_qvals = self.evaluate(make_env(self.env_name))
                print(mean_reward) 
                print(mean_steps)
                self.writer.add_scalar('steps', mean_steps, i)
                self.writer.add_scalar('reward', mean_reward, i)
                self.writer.add_scalar('mean_qvals', mean_qvals, i)
                self.writer.add_scalar('loss', loss_t, i)
                
                self.load_weigths_into_target_network()
                self.writer.add_scalar('epsilon', self.epsilon, i)
                self.epsilon = max(self.epsilon * EPS_DECAY_RATE, MIN_EPSILON)
                #clear_output(True)
            # adjust agent parameters
            if i % 5000 == 0:
                self.save("./nn/" + self.config['NAME'] + self.env_name)

