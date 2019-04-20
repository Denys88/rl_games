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
    'GAMMA' : 0.99, #discount value
    'TAU' : 0.5, #for gae
    'LEARNING_RATE' : 1e-4,
    'EPSILON_DECAY_FRAMES' : 1e5,
    'NAME' : 'A2C',
    'SCORE_TO_WIN' : 20,
    'ENV_NAME' : 'CartPole-v1'
    'REWARD_SHAPER',
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5,
    'STEPS_NUM' : 1,
    'ENTROPY_COEF' : 0.001,
    'ACTOR_STEPS_PER_UPDATE' : 10,
    'NUM_ACTORS' : 8,
    'PPO' : True,
    'E_CLIP' : 0.1
}

a2c_configurations = {
    'CartPole-v1' : {
        'NETWORK' : networks.CartPoleA2C(),
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
        'ENV_CREATOR' : lambda : gym.make('CartPole-v1')
    }
}

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def compute_gae(rewards, dones, values, gamma, tau):
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * tau * (1 -dones[step]) * gae
        returns.append(gae + values[step])
    return returns[::-1]


def flatten_first_two_dims(arr):
    if arr.ndim > 2:
        return arr.reshape(-1, *arr.shape[-(arr.ndim-2):])
    else:
        return arr.reshape(-1)

class Agent:
    def __init__(self, sess, actions_num, observation_shape, network, determenistic = False):
        self.sess = sess
        self.determenistic = determenistic
        self.obs_ph = tf.placeholder('float32', (None, ) + observation_shape)   
        self.actions , self.values = network('agent', self.obs_ph, actions_num, reuse=False)
        self.softmax_probs = tf.nn.softmax(self.actions)
        self.variables = TensorFlowVariables(self.softmax_probs, self.sess)
        self.sess.run(tf.global_variables_initializer())

    def get_network_output(self, state):
        return self.sess.run([self.softmax_probs, self.values, self.actions], {self.obs_ph: state})

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def get_action_and_value(self, state):
        policy, value, log_policy = self.get_network_output([state])
        if self.determenistic:
            action = np.argmax(policy[0])
        else:
            action = np.random.choice(len(policy[0]), p=policy[0])
        return action, value[0][0], log_policy[0]

    

class NStepBuffer:
    def __init__(self, steps_num, env, agent, gamma, tau, rewards_shaper):
        self.steps_num = steps_num
        self.env = env
        self.agent = agent
        self.rewards_shaper = rewards_shaper
        self.is_done = True
        self.done_reward = []
        self.done_shaped_reward = []
        self.done_steps = []
        self.gamma = gamma
        self.tau = tau
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

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logits = [],[],[],[],[],[]
        is_done = self.is_done
        for _ in range(self.steps_num):
            mb_dones.append(is_done)
            state = self.current_state
            action, value, mb_logit = self.agent.get_action_and_value(state)
            new_state, reward, is_done, _ = self.env.step(action)
            mb_obs.append(np.copy(state))
            mb_actions.append(action)
            mb_values.append(value)
            mb_logits.append(mb_logit)
            
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
        mb_logits = np.asarray(mb_logits, dtype=np.int32)
        mb_dones = mb_dones[1:]
    
        if mb_dones[-1] == 0:
            _, next_value, _ = self.agent.get_action_and_value(self.current_state)
            
            rewards = compute_gae(mb_rewards, mb_dones, mb_values + [next_value], self.gamma, self.tau)
        else:
            rewards = compute_gae(mb_rewards, mb_dones, mb_values + [0], self.gamma, self.tau)
        '''
        if mb_dones[-1] == 0:
            rewards2 = discount_with_dones(mb_rewards + [next_value], mb_dones + [0], self.gamma)[:-1]
        else:
            rewards2 = discount_with_dones(mb_rewards, mb_dones, self.gamma)
        '''
        mb_rewards = np.asarray(rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_logits

class Worker:
    def __init__(self, env_name, actions_num, observation_shape, steps_num, gamma, tau):
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        agent = Agent(sess, actions_num, observation_shape, a2c_configurations['CartPole-v1']['NETWORK'], determenistic = False)
        env = a2c_configurations[env_name]['ENV_CREATOR']()
        buffer = NStepBuffer(steps_num, env, agent, gamma, tau, a2c_configurations['CartPole-v1']['REWARD_SHAPER'])
        self.buffer = buffer

    def set_weights(self, weights):
        self.buffer.set_weights(weights)

    def step(self):
        return self.buffer.play_steps()

class A2CAgent:
    def __init__(self, sess, env_name, observation_shape, actions_num, action_shape, config = default_config):    
        self.env_name = config['ENV_NAME']
        self.ppo = config['PPO']
        self.e_clip = config['E_CLIP']
        self.network = a2c_configurations[self.env_name]['NETWORK']
        self.num_actors = config['NUM_ACTORS']
        self.steps_num = config['STEPS_NUM']
        self.config = config
        self.state_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter()
        self.sess = sess
        self.writer = SummaryWriter()
        self.grad_norm = config['GRAD_NORM']
        self.gamma = self.config['GAMMA']
        self.tau = self.config['TAU']
        self.obs_ph = tf.placeholder('float32', (None, ) + observation_shape, name = "obs")    
        self.actions_ph = tf.placeholder('int32', (None,), name = "actions")
        self.rewards_ph = tf.placeholder('float32', (None,), name = "rewards")
        self.prev_logits_ph = tf.placeholder('float32', (None, ) + (actions_num,), name = "prev_logs")
        self.advantages_ph = tf.placeholder('float32', (None,), name = "adv")

        self.actions, self.state_values = self.network('agent', self.obs_ph, actions_num, reuse=False)
        self.probs = tf.nn.softmax(self.actions)
        self.logp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actions, labels=tf.squeeze(self.actions_ph))
        self.old_logp_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prev_logits_ph, labels=tf.squeeze(self.actions_ph))
        if (self.ppo):
            self.prob_ratio = tf.exp(self.old_logp_actions - self.logp_actions)
            self.pg_loss_unclipped = -self.advantages_ph * self.prob_ratio
            self.pg_loss_clipped = -self.advantages_ph * tf.clip_by_value(self.prob_ratio, 1.- self.e_clip, 1.+ self.e_clip)
            self.actor_loss = tf.reduce_mean(tf.maximum(self.pg_loss_unclipped, self.pg_loss_clipped))
        else:
            self.actor_loss = tf.reduce_mean(self.logp_actions * self.advantages_ph) 

        self.entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actions, labels=self.probs, name="entropy"))
        self.critic_loss = tf.reduce_mean((tf.squeeze(self.state_values) - self.rewards_ph)**2 ) # TODO use huber loss too
        self.loss = self.actor_loss + 0.5 * self.critic_loss - self.config['ENTROPY_COEF'] * self.entropy
        self.train_step = tf.train.AdamOptimizer(self.config['LEARNING_RATE'])


        self.variables = TensorFlowVariables([self.actions, self.state_values], self.sess)
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        grads = tf.gradients(self.loss, self.weights)
        if self.config['TRUNCATE_GRADS']:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        grads = list(zip(grads, self.weights))
        self.train_op = self.train_step.apply_gradients(grads)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.remote_worker = ray.remote(Worker)
        self.actor_list = [self.remote_worker.remote(self.env_name, self.actions_num, self.state_shape, self.steps_num, self.gamma, self.tau) for i in range(self.num_actors)]

    def get_action_distribution(self, state):
        return self.sess.run(self.probs, {self.obs_ph: state})


    def get_action(self, state, is_determenistic = False):
        policy = self.get_action_distribution([state])[0]
        if is_determenistic:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)
        return action      

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

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
            policies = []
            ind += steps_per_update
            info = [actor.step.remote() for actor in self.actor_list]
            unpacked_info = ray.get(info)
            for k in range(self.num_actors):
                actor_info = unpacked_info[k]
                obses.append(actor_info[0])
                rewards.append(actor_info[1])
                actions.append(actor_info[2])
                policies.append(actor_info[4])
                value = actor_info[3]
                reward = actor_info[1]
                advantages.append(reward - value)
            
            obses = np.asarray(obses, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int32)
            advantages = np.asarray(advantages, dtype=np.float32)
            policies = np.asarray(policies, dtype=np.float32)
            obses = flatten_first_two_dims(obses)
            rewards = flatten_first_two_dims(rewards)
            actions = flatten_first_two_dims(actions)
            advantages = flatten_first_two_dims(advantages)
            policies = flatten_first_two_dims(policies)

            dict = {self.obs_ph: obses, self.actions_ph : actions, self.rewards_ph : rewards, self.advantages_ph : advantages, self.prev_logits_ph : policies}
            a_loss, c_loss, entropy, _ = self.sess.run([self.actor_loss, self.critic_loss, self.entropy, self.train_op], dict)

            if ind % 1000 == 0:
                print("a_loss", a_loss)
                print("c_loss", c_loss)
                print("entropy", entropy)
                if (len(scores)) > 0:
                    print("scores", np.mean(scores))
                    scores = []

            
        