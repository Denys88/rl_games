from rl_games.common import tr_helpers, vecenv
#from rl_games.algos_tf14 import networks
from rl_games.algos_tf14.tensorflow_utils import TensorFlowVariables
from rl_games.algos_tf14.tf_moving_mean_std import MovingMeanStd

import tensorflow as tf
import numpy as np
import collections
import time
from collections import deque, OrderedDict
from tensorboardX import SummaryWriter

import gym

from datetime import datetime


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class A2CAgent:
    def __init__(self, sess, base_name, observation_space, action_space, config):
        observation_shape = observation_space.shape
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)
        self.self_play = config.get('self_play', False)
        self.name = base_name
        self.config = config
        self.env_name = config['env_name']
        self.ppo = config['ppo']
        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.is_polynom_decay_lr = config['lr_schedule'] == 'polynom_decay'
        self.is_exp_decay_lr = config['lr_schedule'] == 'exp_decay'
        self.lr_multiplier = tf.constant(1, shape=(), dtype=tf.float32)
        self.epoch_num = tf.Variable( tf.constant(0, shape=(), dtype=tf.float32), trainable=False)

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_actors = config['num_actors']
        self.env_config = self.config.get('env_config', {})
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.num_agents = self.vec_env.get_number_of_agents()
        self.steps_num = config['steps_num']
        self.seq_len = self.config['seq_len']
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = self.config['normalize_input']
       
        self.state_shape = observation_shape
        self.critic_coef = config['critic_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))
        self.sess = sess
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.ignore_dead_batches = self.config.get('ignore_dead_batches', False)

        self.dones = np.asarray([False]*self.num_actors *self.num_agents, dtype=np.bool)
        self.current_rewards = np.asarray([0]*self.num_actors *self.num_agents, dtype=np.float32)
        self.current_lengths = np.asarray([0]*self.num_actors *self.num_agents, dtype=np.float32)
        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = deque([], maxlen=self.games_to_track)
        self.game_lengths = deque([], maxlen=self.games_to_track)
        self.game_scores = deque([], maxlen=self.games_to_track)
        self.obs_ph = tf.placeholder(observation_space.dtype, (None, ) + observation_shape, name = 'obs')
        self.target_obs_ph = tf.placeholder(observation_space.dtype, (None, ) + observation_shape, name = 'target_obs') 
        self.actions_num = action_space.n   
        self.actions_ph = tf.placeholder('int32', (None,), name = 'actions')       
        if self.use_action_masks:
            self.action_mask_ph = tf.placeholder('int32', (None, self.actions_num), name = 'actions_mask')       
        else:
            self.action_mask_ph = None

        self.old_logp_actions_ph = tf.placeholder('float32', (None, ), name = 'old_logpactions')
        self.rewards_ph = tf.placeholder('float32', (None,), name = 'rewards')
        self.old_values_ph = tf.placeholder('float32', (None,), name = 'old_values')
        self.advantages_ph = tf.placeholder('float32', (None,), name = 'advantages')
        self.learning_rate_ph = tf.placeholder('float32', (), name = 'lr_ph')

        self.update_epoch_op = self.epoch_num.assign(self.epoch_num + 1)
        self.current_lr = self.learning_rate_ph

        self.input_obs = self.obs_ph
        self.input_target_obs = self.target_obs_ph
 
        if observation_space.dtype == np.uint8:
            self.input_obs = tf.to_float(self.input_obs) / 255.0
            self.input_target_obs = tf.to_float(self.input_target_obs) / 255.0

        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']
        if self.is_polynom_decay_lr:
            self.lr_multiplier = tf.train.polynomial_decay(1.0, self.epoch_num, config['max_epochs'], end_learning_rate=0.001, power=tr_helpers.get_or_default(config, 'decay_power', 1.0))
        if self.is_exp_decay_lr:
            self.lr_multiplier = tf.train.exponential_decay(1.0, self.epoch_num,config['max_epochs'],  decay_rate = config['decay_rate'])
        if self.normalize_input:
            self.moving_mean_std = MovingMeanStd(shape = observation_space.shape, epsilon = 1e-5, decay = 0.99)
            self.input_obs = self.moving_mean_std.normalize(self.input_obs, train=True)
            self.input_target_obs = self.moving_mean_std.normalize(self.input_target_obs, train=False)

        games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation

        self.train_dict = {
            'name' : 'agent',
            'inputs' : self.input_obs,
            'batch_num' : self.config['minibatch_size'],
            'games_num' : games_num,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : self.actions_ph,
            'action_mask_ph' : None
        }

        self.run_dict = {
            'name' : 'agent',
            'inputs' : self.input_target_obs,
            'batch_num' : self.num_actors * self.num_agents,
            'games_num' : self.num_actors * self.num_agents,
            'actions_num' : self.actions_num,
            'prev_actions_ph' : None,
            'action_mask_ph' : self.action_mask_ph
        }

        self.states = None
        if self.network.is_rnn():
            self.logp_actions ,self.state_values, self.action, self.entropy, self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = self.network(self.train_dict, reuse=False)
            self.target_neglogp, self.target_state_values, self.target_action, _,  self.target_states_ph, self.target_masks_ph, self.target_lstm_state, self.target_initial_state, self.logits = self.network(self.run_dict, reuse=True)
            self.states = self.target_initial_state
        else:
            self.logp_actions ,self.state_values, self.action, self.entropy = self.network(self.train_dict, reuse=False)
            self.target_neglogp, self.target_state_values, self.target_action, _, self.logits = self.network(self.run_dict, reuse=True)

        self.saver = tf.train.Saver()
        self.variables = TensorFlowVariables([self.target_action, self.target_state_values, self.target_neglogp], self.sess)
        
        if self.is_train:
            self.setup_losses()

        self.sess.run(tf.global_variables_initializer())

    def setup_losses(self):
        curr_e_clip = self.e_clip * self.lr_multiplier
        if (self.ppo):
            self.prob_ratio = tf.exp(self.old_logp_actions_ph - self.logp_actions)
            
            self.pg_loss_unclipped = -tf.multiply(self.advantages_ph, self.prob_ratio)
            self.pg_loss_clipped = -tf.multiply(self.advantages_ph, tf.clip_by_value(self.prob_ratio, 1.- curr_e_clip, 1.+ curr_e_clip))
            self.actor_loss = tf.maximum(self.pg_loss_unclipped, self.pg_loss_clipped)
        else:
            self.actor_loss = self.logp_actions * self.advantages_ph

        self.actor_loss = tf.reduce_mean(self.actor_loss)
        self.c_loss = (tf.squeeze(self.state_values) - self.rewards_ph)**2

        if self.clip_value:
            self.cliped_values = self.old_values_ph + tf.clip_by_value(tf.squeeze(self.state_values) - self.old_values_ph, - curr_e_clip, curr_e_clip)
            self.c_loss_clipped = tf.square(self.cliped_values - self.rewards_ph)
            self.critic_loss = tf.maximum(self.c_loss, self.c_loss_clipped)
        else:
            self.critic_loss = self.c_loss
 
        self.critic_loss = tf.reduce_mean(self.critic_loss)
        
        self.kl_approx = 0.5 * tf.stop_gradient(tf.reduce_mean((self.old_logp_actions_ph - self.logp_actions)**2))
        if self.is_adaptive_lr:
            self.current_lr = tf.where(self.kl_approx > (2.0 * self.lr_threshold), tf.maximum(self.current_lr / 1.5, 1e-6), self.current_lr)
            self.current_lr = tf.where(self.kl_approx < (0.5 * self.lr_threshold), tf.minimum(self.current_lr * 1.5, 1e-2), self.current_lr)

        self.loss = self.actor_loss + 0.5 * self.critic_coef * self.critic_loss - self.config['entropy_coef'] * self.entropy
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss += self.reg_loss
        self.train_step = tf.train.AdamOptimizer(self.current_lr * self.lr_multiplier)
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')

        grads = tf.gradients(self.loss, self.weights)
        if self.config['truncate_grads']:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        grads = list(zip(grads, self.weights))
        self.train_op = self.train_step.apply_gradients(grads)

    def update_epoch(self):
        return self.sess.run([self.update_epoch_op])[0]

    def get_action_values(self, obs):
        run_ops = [self.target_action, self.target_state_values, self.target_neglogp]
        if self.network.is_rnn():
            run_ops.append(self.target_lstm_state)
            return self.sess.run(run_ops, {self.target_obs_ph : obs, self.target_states_ph : self.states, self.target_masks_ph : self.dones})
        else:
            return (*self.sess.run(run_ops, {self.target_obs_ph : obs}), None)

    def get_masked_action_values(self, obs, action_masks):
        run_ops = [self.target_action, self.target_state_values, self.target_neglogp, self.logits]
        if self.network.is_rnn():
            run_ops.append(self.target_lstm_state)
            return self.sess.run(run_ops, {self.action_mask_ph: action_masks, self.target_obs_ph : obs, self.target_states_ph : self.states, self.target_masks_ph : self.dones})
        else:
            return (*self.sess.run(run_ops, {self.action_mask_ph: action_masks, self.target_obs_ph : obs}), None)


    def get_values(self, obs):
        if self.network.is_rnn():
            return self.sess.run([self.target_state_values], {self.target_obs_ph : obs, self.target_states_ph : self.states, self.target_masks_ph : self.dones})
        else:
            return self.sess.run([self.target_state_values], {self.target_obs_ph : obs})

    def get_weights(self):
        return self.variables.get_flat()
    
    def set_weights(self, weights):
        return self.variables.set_flat(weights)



    def play_steps(self):
        # here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        
        mb_states = []
        epinfos = []

        # for n in range number of steps
        for _ in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()

            if self.use_action_masks:
                actions, values, neglogpacs, _, self.states = self.get_masked_action_values(self.obs, masks)
            else:
                actions, values, neglogpacs, self.states = self.get_action_values(self.obs)

            actions = np.squeeze(actions)
            values = np.squeeze(values)
            neglogpacs = np.squeeze(neglogpacs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones.copy())

            self.obs[:], rewards, self.dones, infos = self.vec_env.step(actions)
            self.current_rewards += rewards

            self.current_lengths += 1
            for reward, length, done, info in zip(self.current_rewards[::self.num_agents], self.current_lengths[::self.num_agents], self.dones[::self.num_agents], infos):
                if done:
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)

                    game_res = 1.0
                    if isinstance(info, dict):
                        game_res = info.get('battle_won', 0.5)
                    self.game_scores.append(game_res)

            self.current_rewards = self.current_rewards * (1.0 - self.dones)
            self.current_lengths = self.current_lengths * (1.0 - self.dones)

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
        mb_states = np.asarray(mb_states, dtype=np.float32)
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
        if self.network.is_rnn():
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states  )), epinfos)
        else:
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), None, epinfos)
        return result

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def train(self):
        self.obs = self.vec_env.reset()
        batch_size = self.steps_num * self.num_actors * self.num_agents
        batch_size_envs = self.steps_num * self.num_actors
        minibatch_size = self.config['minibatch_size']
        mini_epochs_num = self.config['mini_epochs']
        num_minibatches = batch_size // minibatch_size
        last_lr = self.config['learning_rate']
        frame = 0
        update_time = 0
        self.last_mean_rewards = -100500
        play_time = 0
        epoch_num = 0
        max_epochs = self.config.get('max_epochs', 1e6)

        start_time = time.time()
        total_time = 0
        rep_count = 0
        while True:
            play_time_start = time.time()
            epoch_num = self.update_epoch()
            frame += batch_size_envs
            obses, returns, dones, actions, values, neglogpacs, lstm_states, _ = self.play_steps()
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

            if self.network.is_rnn():
                total_games = batch_size // self.seq_len
                num_games_batch = minibatch_size // self.seq_len
                game_indexes = np.arange(total_games)
                flat_indexes = np.arange(total_games * self.seq_len).reshape(total_games, self.seq_len)
                lstm_states = lstm_states[::self.seq_len]
                for _ in range(0, mini_epochs_num):
                    np.random.shuffle(game_indexes)

                    for i in range(0, num_minibatches):
                        batch = range(i * num_games_batch, (i + 1) * num_games_batch)
                        mb_indexes = game_indexes[batch]
                        mbatch = flat_indexes[mb_indexes].ravel()                        

                        dict = {}
                        dict[self.old_values_ph] = values[mbatch]
                        dict[self.old_logp_actions_ph] = neglogpacs[mbatch]
                        dict[self.advantages_ph] = advantages[mbatch]
                        dict[self.rewards_ph] = returns[mbatch]
                        dict[self.actions_ph] = actions[mbatch]
                        dict[self.obs_ph] = obses[mbatch]
                        dict[self.masks_ph] = dones[mbatch]

                        dict[self.states_ph] = lstm_states[mb_indexes]

                        dict[self.learning_rate_ph] = last_lr
                        run_ops = [self.actor_loss, self.critic_loss, self.entropy, self.kl_approx, self.current_lr, self.lr_multiplier,  self.train_op]
                        run_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                        a_loss, c_loss, entropy, kl, last_lr, lr_mul,_, _ = self.sess.run(run_ops, dict)
                        a_losses.append(a_loss)
                        c_losses.append(c_loss)
                        kls.append(kl)
                        entropies.append(entropy)
            else:
                for _ in range(0, mini_epochs_num):
                    permutation = np.random.permutation(batch_size)
                    obses = obses[permutation]
                    returns = returns[permutation]
                    
                    actions = actions[permutation]
                    values = values[permutation]
                    neglogpacs = neglogpacs[permutation]
                    advantages = advantages[permutation]
                                                   
                    for i in range(0, num_minibatches):
                        batch = range(i * minibatch_size, (i + 1) * minibatch_size)
                        dict = {self.obs_ph: obses[batch], self.actions_ph : actions[batch], self.rewards_ph : returns[batch], 
                                self.advantages_ph : advantages[batch], self.old_logp_actions_ph : neglogpacs[batch], self.old_values_ph : values[batch]}
                        dict[self.learning_rate_ph] = last_lr
                        run_ops = [self.actor_loss, self.critic_loss, self.entropy, self.kl_approx, self.current_lr, self.lr_multiplier, self.train_op]
                            
                        run_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                        a_loss, c_loss, entropy, kl, last_lr, lr_mul, _, _ = self.sess.run(run_ops, dict)
                        a_losses.append(a_loss)
                        c_losses.append(c_loss)
                        kls.append(kl)
                        entropies.append(entropy)

            update_time_end = time.time()
            update_time = update_time_end - update_time_start
            sum_time = update_time + play_time
            total_time = update_time_end - start_time

            if True:
                scaled_time = self.num_agents * sum_time
                print('frames per seconds: ', batch_size / scaled_time)
                self.writer.add_scalar('performance/fps', batch_size / scaled_time, frame)
                self.writer.add_scalar('performance/upd_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/a_loss', np.mean(a_losses), frame)
                self.writer.add_scalar('losses/c_loss', np.mean(c_losses), frame)
                self.writer.add_scalar('losses/entropy', np.mean(entropies), frame)
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', np.mean(kls), frame)
                self.writer.add_scalar('epochs', epoch_num, frame)
                
                if len(self.game_rewards) > 0:
                    mean_rewards = np.mean(self.game_rewards)
                    mean_lengths = np.mean(self.game_lengths)
                    mean_scores = np.mean(self.game_scores)
                    self.writer.add_scalar('rewards/mean', mean_rewards, frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/mean', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    self.writer.add_scalar('win_rate/mean', mean_scores, frame)
                    self.writer.add_scalar('win_rate/time', mean_scores, total_time)

                    if rep_count % 10 == 0:
                        self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                        rep_count += 1

                    if mean_rewards > self.last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'])
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                            return self.last_mean_rewards, epoch_num

                if epoch_num > max_epochs:
                    self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num                               
                update_time = 0
            
        