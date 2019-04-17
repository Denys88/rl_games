import networks
import tr_helpers
import experience
import tensorflow as tf
import numpy as np
import collections
import time
import ray
from collections import deque, OrderedDict
from tensorboardX import SummaryWriter
from tensorflow_utils import TensorFlowVariables
import gym


default_config = {
    'GAMMA' : 0.99,
    'LEARNING_RATE' : 1e-4,
    'EPSILON_DECAY_FRAMES' : 1e5,
    'NAME' : 'A2C',
    'SCORE_TO_WIN' : 20,
    'NETWORK' : networks.CartPoleA2C(),
    'ENV' : lambda : None, #gym.make('CartPole-v1')
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5,
    'STEPS_NUM' : 1,
    'ENTROPY_COEF' : 0.001,
    'ACTOR_STEPS_PER_UPDATE' : 10,
    'NUM_ACTORS' : 8
}

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]



class Agent:
    def __init__(self, sess, actions_num, observation_shape, network, determenistic = False):
        self.sess = sess
        self.determenistic = determenistic
        self.obs_ph = tf.placeholder('float32', (None, ) + observation_shape)   
        self.actions , self.values = network('agent', self.obs_ph, actions_num, reuse=False)
        self.softmax_probs = tf.nn.softmax(self.actions)
        self.variables = TensorFlowVariables(self.softmax_probs, self.sess)
        self.sess.run(tf.global_variables_initializer())

    def get_action_distribution(self, state):
        return self.sess.run([self.softmax_probs, self.values], {self.obs_ph: state})

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def get_action(self, state):
        policy, value = self.get_action_distribution([state])
        if self.determenistic:
            action = np.argmax(policy[0])
        else:
            action = np.random.choice(len(policy[0]), p=policy[0])
        return action, value[0][0]

    

class NStepBuffer:
    def __init__(self, steps_num, env, agent, gamma, rewards_shaper):
        self.steps_num = steps_num
        self.env = env
        self.agent = agent
        self.rewards_shaper = rewards_shaper
        self.is_done = True
        self.done_reward = []
        self.done_shaped_reward = []
        self.done_steps = []
        self.gamma = gamma
        self._reset()
    
    def _reset(self):
        self.current_state = self.env.reset()
        self.total_reward = 0.0
        self.total_shaped_reward = 0.0
        self.step_count = 0
        self.is_done = False

    def get_logs(self):
        res = [self.done_reward, self.done_shaped_reward, self.done_steps]
        self.done_reward = []
        self.done_shaped_reward = []
        self.done_steps = []
        return res

    def set_weights(self, weights):
        self.agent.set_weights(weights)

    def play_steps(self):

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        is_done = self.is_done
        for _ in range(self.steps_num):
            mb_dones.append(is_done)
            state = self.current_state
            action, value = self.agent.get_action(state)
            new_state, reward, is_done, _ = self.env.step(action)
            mb_obs.append(np.copy(state))
            mb_actions.append(action)
            mb_values.append(value)
            
            self.step_count += 1
            self.total_reward += reward
            shaped_reward = self.rewards_shaper(reward)
            self.total_shaped_reward += shaped_reward
            self.current_state = new_state
            mb_rewards.append(shaped_reward)
            self.is_done = is_done
            if is_done:
                self.done_reward.append(self.total_reward)
                self.done_steps.append(self.step_count)
                self.done_shaped_reward.append(self.total_shaped_reward)
                self._reset()

        
        mb_dones.append(is_done)

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        #mb_masks = mb_dones[:-1]
        mb_dones = mb_dones[1:]
            
        if mb_dones[-1] == 0:
            _, next_value = self.agent.get_action(self.current_state)
            rewards = discount_with_dones(mb_rewards + [next_value], mb_dones + [0], self.gamma)[:-1]
        else:
            rewards = discount_with_dones(mb_rewards, mb_dones, self.gamma)
            
        mb_rewards = np.asarray(rewards, dtype=np.float32)
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones

class Worker:
    def __init__(self, actions_num, observation_shape, steps_num, gamma):
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        agent = Agent(sess, actions_num, observation_shape, networks.CartPoleA2C(), determenistic = False)
        env = gym.make('CartPole-v1')
        buffer = NStepBuffer(steps_num, env, agent, gamma, tr_helpers.DefaultRewardsShaper())
        self.buffer = buffer

    def set_weights(self, weights):
        self.buffer.set_weights(weights)

    def step(self):
        return self.buffer.play_steps()

class A2CAgent:
    def __init__(self, sess, env_name, observation_shape, actions_num, config = default_config):
        self.network = config['NETWORK']
        self.env_creator = config['ENV']
        self.num_actors = config['NUM_ACTORS']
        self.steps_num = config['STEPS_NUM']
        self.config = config
        self.state_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter()
        self.sess = sess
        self.grad_norm = config['GRAD_NORM']
        self.gamma = self.config['GAMMA']
        self.obs_ph = tf.placeholder('float32', (None, ) + observation_shape)    
        self.actions_ph = tf.placeholder('int32', (None,))
        self.rewards_ph = tf.placeholder('float32', (None,))
        self.advantages_ph = tf.placeholder('float32', (None,))

        self.actions, self.state_values = self.network('agent', self.obs_ph, actions_num, reuse=False)
        self.probs = tf.nn.softmax(self.actions)
        self.logp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actions, labels=self.actions_ph)
        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actions, labels=self.probs, name="entropy"))
        self.actor_loss = tf.reduce_mean(self.logp_actions * self.advantages_ph) 
        self.critic_loss = tf.reduce_mean((tf.squeeze(self.state_values) - self.rewards_ph)**2 ) # TODO use huber loss too
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.config['ENTROPY_COEF'] * self.entropy
        self.train_step = tf.train.AdamOptimizer(self.config['LEARNING_RATE'])
        self.variables = TensorFlowVariables([self.actions, self.state_values], self.sess)
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        grads = tf.gradients(self.loss, self.weights)
        #grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        grads = list(zip(grads, self.weights))

        # 4. Backpropagation
        #self.train_op = self.train_step.apply_gradients(grads)
        self.train_op = self.train_step.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.remote_worker = ray.remote(Worker)
        self.actor_list = [self.remote_worker.remote(self.actions_num, self.state_shape, self.steps_num, self.gamma) for i in range(self.num_actors)]

    def get_action_distribution(self, state):
        return self.sess.run(self.probs, {self.obs_ph: state})


    def get_action(self, state, is_determenistic = False):
        policy = self.get_action_distribution([state])[0]
        if is_determenistic:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)
        return action      


    def train(self):
        
        ind = 0
        steps_per_update = self.steps_num
        scores = []
        while True:
            
            weights = self.variables.get_weights()
            weights_id = ray.put(weights)
            [actor.set_weights.remote(weights_id) for actor in self.actor_list]    
            
            obses = []
            actions = []
            rewards = []
            advantages = []
            ind += steps_per_update
            info = [actor.step.remote() for actor in self.actor_list]
            unpacked_info = ray.get(info)
            for k in range(self.num_actors):
                actor_info = unpacked_info[k]
                obses.append(actor_info[0])
                rewards.append(actor_info[1])
                actions.append(actor_info[2])
                value = actor_info[3]
                reward = actor_info[1]
                advantages.append(reward - value)
            
            obses = np.asarray(obses, dtype=np.float32)[0,:]
            rewards = np.asarray(rewards, dtype=np.float32)[0,:]
            actions = np.asarray(actions, dtype=np.int32)[0,:]
            advantages = np.asarray(advantages, dtype=np.float32)[0,:]
            dict = {self.obs_ph: obses, self.actions_ph : actions, self.rewards_ph : rewards, self.advantages_ph : advantages}
            a_loss, c_loss, entropy, _ = self.sess.run([self.actor_loss, self.critic_loss, self.entropy, self.train_op], dict)

            if ind % 1000 == 0:
                print("a_loss", a_loss)
                print("c_loss", c_loss)
                print("entropy", entropy)
                if (len(scores)) > 0:
                    print("scores", np.mean(scores))
                    scores = []

            
        