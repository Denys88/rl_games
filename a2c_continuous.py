import networks
import tr_helpers
import experience
import wrappers
import tensorflow as tf
import numpy as np
import collections
import time
from env_configurations import a2c_configurations
from collections import deque, OrderedDict
from tensorboardX import SummaryWriter
from tensorflow_utils import TensorFlowVariables
import gym
import vecenv

config_example = {
    'NETWORK' : networks.ModelA2C(networks.default_a2c_network),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'VECENV_TYPE' : 'RAY',
    'GAMMA' : 0.99,
    'TAU' : 0.9,
    'LEARNING_RATE' : 1e-4,
    'NAME' : 'robo1',
    'SCORE_TO_WIN' : 5000,
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 5,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.000,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'RoboschoolAnt-v1',
    'PPO' : True,
    'E_CLIP' : 0.2,
    'NUM_ACTORS' : 16,
    'STEPS_NUM' : 128,
    'MINIBATCH_SIZE' : 512,
    'MINI_EPOCHS' : 4,
    'CRITIC_COEF' : 1,
    'CLIP_VALUE' : True,
    'IS_ADAPTIVE_LR' : False,
    'LR_THRESHOLD' : 0.5
}

def swap_and_flatten01(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

#(-1, 1) -> (low, high)
def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):


class A2CAgent:
    def __init__(self, sess, name, observation_space, is_discrete, action_space, config):
        assert not is_discrete  

        self.name = name
        self.actions_low = action_space.low
        self.actions_high = action_space.high
        self.env_name = config['ENV_NAME']
        self.ppo = config['PPO']
        self.is_adaptive_lr = config['IS_ADAPTIVE_LR']
        if self.is_adaptive_lr:
            self.lr_threshold = config['LR_THRESHOLD']
        self.e_clip = config['E_CLIP']
        self.clip_value = config['CLIP_VALUE']
        self.is_discrete = is_discrete
        self.network = config['NETWORK']
        self.rewards_shaper = config['REWARD_SHAPER']
        self.num_actors = config['NUM_ACTORS']
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors)
        self.steps_num = config['STEPS_NUM']
        self.normalize_advantage = config['NORMALIZE_ADVANTAGE']
        self.config = config
        self.state_shape = observation_space.shape
        self.critic_coef = config['CRITIC_COEF']
        self.writer = SummaryWriter()
        self.sess = sess
        self.grad_norm = config['GRAD_NORM']
        self.gamma = self.config['GAMMA']
        self.tau = self.config['TAU']
        self.dones = np.asarray([False]*self.num_actors, dtype=np.bool)
        self.current_rewards = np.asarray([0]*self.num_actors, dtype=np.float32)  
        self.game_rewards = deque([], maxlen=100)
        self.obs_ph = tf.placeholder('float32', (None, ) + self.state_shape, name = 'obs')
        self.target_obs_ph = tf.placeholder('float32', (None, ) + self.state_shape, name = 'target_obs')
        self.actions_num = action_space.shape[0]   
        self.actions_ph = tf.placeholder('float32', (None,) + action_space.shape, name = 'actions')

        self.old_logp_actions_ph = tf.placeholder('float32', (None, ), name = 'old_logpactions')
        self.rewards_ph = tf.placeholder('float32', (None,), name = 'rewards')
        self.old_values_ph = tf.placeholder('float32', (None,), name = 'old_values')
        self.advantages_ph = tf.placeholder('float32', (None,), name = 'advantages')
        self.learning_rate_ph = tf.placeholder('float32', (), name = 'lr_ph')
        self.epoch_num_ph = tf.placeholder('int32', (), name = 'epoch_num_ph')
        self.current_lr = self.learning_rate_ph

        self.logp_actions ,self.state_values, self.action, self.entropy, self.mu, self.sigma  = self.network('agent', self.obs_ph, self.actions_num, self.actions_ph, reuse=False)
        self.target_neglogp, self.target_state_values, self.target_action, _ , self.target_mu, self.target_sigma = self.network('agent',  self.target_obs_ph, self.actions_num, None, reuse=True)
        

        if (self.ppo):
            self.prob_ratio = tf.exp(self.old_logp_actions_ph - self.logp_actions)
            self.pg_loss_unclipped = -tf.multiply(self.advantages_ph, self.prob_ratio)
            self.pg_loss_clipped = -tf.multiply(self.advantages_ph, tf.clip_by_value(self.prob_ratio, 1.- self.e_clip, 1.+ self.e_clip))
            self.actor_loss = tf.reduce_mean(tf.maximum(self.pg_loss_unclipped, self.pg_loss_clipped))
        else:
            self.actor_loss = tf.reduce_mean(self.logp_actions * self.advantages_ph)


        self.c_loss = (tf.squeeze(self.state_values) - self.rewards_ph)**2
        if self.clip_value:
            self.cliped_values = self.old_values_ph + tf.clip_by_value(tf.squeeze(self.state_values) - self.old_values_ph, - self.e_clip, self.e_clip)
            self.c_loss_clipped = tf.square(self.cliped_values - self.rewards_ph)
            self.critic_loss = tf.reduce_mean(tf.maximum(self.c_loss, self.c_loss_clipped))
        else:
            self.critic_loss = tf.reduce_mean(self.c_loss)
        
        self.kl_approx = 0.5 * tf.stop_gradient(tf.reduce_mean((self.old_logp_actions_ph - self.logp_actions)**2))
        if self.is_adaptive_lr:
            self.current_lr = tf.where(self.kl_approx > (2.0 * self.lr_threshold), tf.maximum(self.current_lr / 1.5, 1e-6), self.current_lr)
            self.current_lr = tf.where(self.kl_approx < (0.5 * self.lr_threshold), tf.minimum(self.current_lr * 1.5, 1e-2), self.current_lr)
            self.current_lr = tf.where(self.epoch_num_ph > 20, self.current_lr, self.learning_rate_ph)
        self.loss = self.actor_loss + 0.5 * self.critic_coef * self.critic_loss - self.config['ENTROPY_COEF'] * self.entropy
        
        self.train_step = tf.train.AdamOptimizer(self.current_lr)
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')

        grads = tf.gradients(self.loss, self.weights)
        if self.config['TRUNCATE_GRADS']:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        grads = list(zip(grads, self.weights))
        self.train_op = self.train_step.apply_gradients(grads)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action_values(self, obs):
        return self.sess.run([self.target_action, self.target_state_values, self.target_neglogp, self.target_mu, self.target_sigma], {self.target_obs_ph : obs})

    def get_values(self, obs):
        return self.sess.run([self.target_state_values], {self.target_obs_ph : obs})

    def play_steps(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mus, mb_sigmas = [],[],[],[],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.steps_num):
            actions, values, neglogpacs, sigma, mu = self.get_action_values(self.obs)
            values = np.squeeze(values)
            #actions = np.squeeze(actions)
            neglogpacs = np.squeeze(neglogpacs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_mus.append(mu)
            mb_sigmas.append(sigma)
            self.obs[:], rewards, self.dones, infos = self.vec_env.step(rescale_actions(self.actions_low, self.actions_high, actions))
            self.current_rewards += rewards

            for reward, done in zip(self.current_rewards, self.dones):
                if done:
                    self.game_rewards.append(reward)

            self.current_rewards = self.current_rewards * (1.0 -self.dones)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)
            mb_rewards.append(shaped_rewards)

        #using openai baseline approach
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.get_values(self.obs)
        last_values = np.squeeze(last_values)
   

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), epinfos)
        return result

    def get_action(self, state, det = False):
        action = self.sess.run(self.action, {self.obs_ph: state})
        return rescale_actions(self.actions_low, self.actions_high, action)


    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def train(self):
        self.obs = self.vec_env.reset()
        batch_size = self.steps_num * self.num_actors
        minibatch_size = self.config['MINIBATCH_SIZE']
        mini_epochs_num = self.config['MINI_EPOCHS']
        num_minibatches = batch_size // minibatch_size
        last_lr = self.config['LEARNING_RATE']
        frame = 0
        update_time = 0
        last_mean_rewards = -100500
        play_time = 0
        epoch_num = 0
        while True:
            play_time_start = time.time()
            epoch_num += 1
            frame += batch_size
            obses, returns, dones, actions, values, neglogpacs, infos = self.play_steps()
            advantages = returns - values
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            a_losses = []
            c_losses = []
            entropies = []
            kls = []
            play_time_end = time.time()
            play_time = play_time_end - play_time_start
            update_time_start = time.time()
            for _ in range(0, mini_epochs_num):
                
                permutation = np.random.permutation(batch_size)
                obses = obses[permutation]
                returns = returns[permutation]
                dones = dones[permutation]
                actions = actions[permutation]
                values = values[permutation]
                neglogpacs = neglogpacs[permutation]
                advantages = advantages[permutation]
                 
                for i in range(0, num_minibatches):
                    batch = range(i * minibatch_size, (i + 1) * minibatch_size)
                    #std_advs = advantages[batch]
                    dict = {self.obs_ph: obses[batch], self.actions_ph : actions[batch], self.rewards_ph : returns[batch], 
                            self.advantages_ph : advantages[batch], self.old_logp_actions_ph : neglogpacs[batch], self.old_values_ph : values[batch]}
                    dict[self.learning_rate_ph] = last_lr
                    dict[self.epoch_num_ph] = epoch_num
                    a_loss, c_loss, entropy, kl, last_lr, _ = self.sess.run([self.actor_loss, self.critic_loss, self.entropy, self.kl_approx, self.current_lr, self.train_op], dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)
            update_time_end = time.time()
            update_time = update_time_end - update_time_start
            sum_time = update_time + play_time
            if True:
                print('Frames per seconds: ', batch_size / sum_time)
                self.writer.add_scalar('Frames per seconds: ', batch_size / sum_time, frame)
                self.writer.add_scalar('upd_time', update_time, frame)
                self.writer.add_scalar('play_time', play_time, frame)
                self.writer.add_scalar('a_loss', np.mean(a_losses), frame)
                self.writer.add_scalar('c_loss', np.mean(c_losses), frame)
                self.writer.add_scalar('entropy', np.mean(entropies), frame)
                self.writer.add_scalar('entropy', entropy, frame)
                self.writer.add_scalar('kl', np.mean(kls), frame)
                self.writer.add_scalar('last_lr', last_lr, frame)
                if len(self.game_rewards) > 0:
                    mean_rewards = np.mean(self.game_rewards)
                    self.writer.add_scalar('mean_rewards', mean_rewards, frame)
                    if mean_rewards > last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.name + self.env_name)
                        if last_mean_rewards > self.config['SCORE_TO_WIN']:
                            print('Network won!')
                            return

                update_time = 0

            
        