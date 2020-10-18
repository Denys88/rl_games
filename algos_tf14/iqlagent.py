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
import tensorflow_probability as tfp


class IQLAgent:
    def __init__(self, sess, base_name, observation_space, action_space, config, logger, central_state_space=None):
        observation_shape = observation_space.shape
        actions_num = action_space.n
        self.config = config
        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.is_polynom_decay_lr = config['lr_schedule'] == 'polynom_decay'
        self.is_exp_decay_lr = config['lr_schedule'] == 'exp_decay'
        self.lr_multiplier = tf.constant(1, shape=(), dtype=tf.float32)
        self.learning_rate_ph = tf.placeholder('float32', (), name='lr_ph')
        self.games_to_track = tr_helpers.get_or_default(config, 'games_to_track', 100)
        self.max_epochs = tr_helpers.get_or_default(self.config, 'max_epochs', 1e6)

        self.games_to_log = self.config.get('games_to_track', 100)
        self.game_rewards = deque([], maxlen=self.games_to_track)
        self.game_lengths = deque([], maxlen=self.games_to_track)
        self.game_scores = deque([], maxlen=self.games_to_log)

        self.epoch_num = tf.Variable(tf.constant(0, shape=(), dtype=tf.float32), trainable=False)
        self.update_epoch_op = self.epoch_num.assign(self.epoch_num + 1)
        self.current_lr = self.learning_rate_ph

        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']
        if self.is_polynom_decay_lr:
            self.lr_multiplier = tf.train.polynomial_decay(1.0, global_step=self.epoch_num, decay_steps=self.max_epochs,
                                                           end_learning_rate=0.001,
                                                           power=tr_helpers.get_or_default(config, 'decay_power', 1.0))
        if self.is_exp_decay_lr:
            self.lr_multiplier = tf.train.exponential_decay(1.0, global_step=self.epoch_num,
                                                            decay_steps=self.max_epochs,
                                                            decay_rate=config['decay_rate'])

        self.env_name = config['env_name']
        self.network = config['network']
        self.batch_size = self.config['batch_size']

        self.obs_shape = observation_shape
        self.actions_num = actions_num
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))
        self.epsilon = self.config['epsilon']
        self.rewards_shaper = self.config['reward_shaper']
        self.epsilon_processor = tr_helpers.LinearValueProcessor(self.config['epsilon'], self.config['min_epsilon'],
                                                                 self.config['epsilon_decay_frames'])
        self.beta_processor = tr_helpers.LinearValueProcessor(self.config['priority_beta'], self.config['max_beta'],
                                                              self.config['beta_decay_frames'])
        if self.env_name:
            self.env_config = config.get('env_config', {})
            self.env = env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)
        self.sess = sess
        self.steps_num = self.config['steps_num']

        self.obs_act_rew = deque([], maxlen=self.steps_num)

        self.is_prioritized = config['replay_buffer_type'] != 'normal'
        self.atoms_num = self.config['atoms_num']
        assert self.atoms_num == 1

        if central_state_space is not None:
            self.state_shape = central_state_space.shape
        else:
            raise NotImplementedError("central_state_space input to IQL is NONE!")
        self.n_agents = self.env.env_info['n_agents']

        if not self.is_prioritized:
            self.exp_buffer = experience.ReplayBuffer(config['replay_buffer_size'], observation_space, self.n_agents)
        else:
            self.exp_buffer = experience.PrioritizedReplayBuffer(config['replay_buffer_size'], config['priority_alpha'], observation_space, self.n_agents)
            self.sample_weights_ph = tf.placeholder(tf.float32, shape=[None, 1], name='sample_weights')

        self.batch_size_ph = tf.placeholder(tf.int32, name='batch_size_ph')
        self.batch_size_ph = tf.placeholder(tf.int32, name='batch_size_ph')
        self.obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.obs_shape, name='obs_ph')
        self.state_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.state_shape, name='state_ph')
        self.actions_ph = tf.placeholder(tf.int32, shape=[None, 1], name='actions_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None, self.n_agents, 1], name='rewards_ph')
        self.next_obs_ph = tf.placeholder(observation_space.dtype, shape=(None,) + self.obs_shape, name='next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None, self.n_agents, 1], name='is_done_ph')
        self.is_not_done = 1 - self.is_done_ph
        self.name = base_name

        self.gamma = self.config['gamma']
        self.gamma_step = self.gamma ** self.steps_num
        self.grad_norm = config['grad_norm']
        self.input_obs = self.obs_ph
        self.input_next_obs = self.next_obs_ph
        if observation_space.dtype == np.uint8:
            print('scaling obs')
            self.input_obs = tf.to_float(self.input_obs) / 255.0
            self.input_next_obs = tf.to_float(self.input_next_obs) / 255.0
        self.setup_qvalues(actions_num)

        self.reg_loss = tf.losses.get_regularization_loss()
        self.td_loss_mean += self.reg_loss
        self.learning_rate = self.config['learning_rate']
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate * self.lr_multiplier)  # .minimize(self.td_loss_mean, var_list=self.weights)
        grads = tf.gradients(self.td_loss_mean, self.weights)
        if self.config['truncate_grads']:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        grads = list(zip(grads, self.weights))
        self.train_op = self.train_step.apply_gradients(grads)

        self.saver = tf.train.Saver()
        self.assigns_op = [tf.assign(w_target, w_self, validate_shape=True) for w_self, w_target in
                           zip(self.weights, self.target_weights)]
        self.variables = TensorFlowVariables(self.qvalues, self.sess)
        if self.env_name:
            sess.run(tf.global_variables_initializer())
        self._reset()

        self.logger = logger
        self.num_env_steps_train = 0

    def get_weights(self):
        return self.variables.get_flat()

    def set_weights(self, weights):
        return self.variables.set_flat(weights)

    def update_epoch(self):
        return self.sess.run([self.update_epoch_op])[0]

    def setup_qvalues(self, actions_num):
        config = {
            'input_obs': self.input_obs,
            'input_next_obs': self.input_next_obs,
            'actions_num': actions_num,
            'is_double': self.config['is_double'],
            'actions_ph': self.actions_ph,
            'batch_size_ph': self.batch_size_ph,
            'n_agents': self.n_agents
        }

        # (bs, n_agents, n_actions), (bs, n_agents, 1), (bs, n_agents, 1)
        self.qvalues, self.current_action_qvalues, self.target_action_qvalues = self.network(config)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        self.reference_qvalues = self.rewards_ph + self.gamma_step * self.is_not_done * self.target_action_qvalues

        if self.is_prioritized:
            # we need to return l1 loss to update priority buffer
            self.abs_errors = tf.abs(self.current_action_qvalues - self.reference_qvalues) + 1e-5
            # the same as multiply gradients later (other way is used in different examples over internet) 
            self.td_loss = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues,
                                                reduction=tf.losses.Reduction.NONE) * self.sample_weights_ph
            self.td_loss_mean = tf.reduce_mean(self.td_loss)
        else:
            self.td_loss_mean = tf.losses.huber_loss(self.current_action_qvalues, self.reference_qvalues,
                                                     reduction=tf.losses.Reduction.MEAN)

        self.reg_loss = tf.losses.get_regularization_loss()
        self.td_loss_mean += self.reg_loss
        self.learning_rate = self.config['learning_rate']
        if self.env_name:
            self.train_step = tf.train.AdamOptimizer(self.learning_rate * self.lr_multiplier).minimize(
                self.td_loss_mean, var_list=self.weights)

    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def _reset(self):
        self.obs_act_rew.clear()
        if self.env_name:
            self.current_obs = self.env.reset()
        self.total_reward = 0.0
        self.total_shaped_reward = 0.0
        self.step_count = 0

    def get_action(self, obs, avail_acts, epsilon=0.0):
        if np.random.random() < epsilon:
            action = tfp.distributions.Categorical(probs=avail_acts.astype(float)).sample().eval(session=self.sess)
        else:
            obs = obs.reshape((self.n_agents,) + self.obs_shape)
            # (n_agents, num_actions)
            qvals = self.get_qvalues(obs).squeeze(0)
            qvals[avail_acts == False] = -9999999
            action = np.argmax(qvals, axis=1)
        return action

    def get_qvalues(self, obs):
        return self.sess.run(self.qvalues, {self.obs_ph: obs, self.batch_size_ph: 1})

    def play_steps(self, steps, epsilon=0.0):
        done_reward = None
        done_shaped_reward = None
        done_steps = None
        done_info = None
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
            state = self.env.get_state()

            action = self.get_action(obs, self.env.get_action_mask(), epsilon)
            new_obs, reward, is_done, info = self.env.step(action)
            # reward = reward * (1 - is_done)

            # Increase step count by 1 - we do not use vec env! (WHIRL)
            self.num_env_steps_train += 1

            # Same reward, done for all agents
            reward = reward[0]
            is_done = all(is_done)
            state = state[0]

            self.step_count += 1
            self.total_reward += reward
            shaped_reward = self.rewards_shaper(reward)
            self.total_shaped_reward += shaped_reward
            self.obs_act_rew.append([new_obs, action, shaped_reward, state])

            if len(self.obs_act_rew) < steps:
                break

            for i in range(int(steps)):
                sreward = self.obs_act_rew[i][2]
                steps_rewards += sreward * cur_gamma
                cur_gamma = cur_gamma * self.gamma

            next_obs, current_action, _, current_st = self.obs_act_rew[0]
            self.exp_buffer.add(self.current_obs, current_action, steps_rewards, new_obs, is_done)
            self.current_obs = next_obs
            break

        if is_done:
            done_reward = self.total_reward
            done_steps = self.step_count
            done_shaped_reward = self.total_shaped_reward
            done_info = info
            self._reset()

        return done_reward, done_shaped_reward, done_steps, done_info

    def load_weights_into_target_network(self):
        self.sess.run(self.assigns_op)

    def sample_batch(self, exp_replay, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
        obs_batch = obs_batch.reshape((batch_size * self.n_agents,) + self.obs_shape)
        act_batch = act_batch.reshape((batch_size * self.n_agents, 1))
        next_obs_batch = next_obs_batch.reshape((batch_size * self.n_agents,) + self.obs_shape)
        reward_batch = reward_batch.reshape((batch_size, 1, 1)).repeat(self.n_agents, axis=1)
        is_done_batch = is_done_batch.reshape((batch_size, 1, 1)).repeat(self.n_agents, axis=1)

        return {
            self.obs_ph: obs_batch, self.actions_ph: act_batch,
            self.rewards_ph: reward_batch, self.is_done_ph: is_done_batch, self.next_obs_ph: next_obs_batch,
            self.batch_size_ph: batch_size
        }

    def sample_prioritized_batch(self, exp_replay, batch_size, beta):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, sample_weights, sample_idxes = exp_replay.sample(
            batch_size, beta)
        obs_batch = obs_batch.reshape((batch_size * self.n_agents,) + self.obs_shape)
        act_batch = act_batch.reshape((batch_size * self.n_agents, 1))
        next_obs_batch = next_obs_batch.reshape((batch_size * self.n_agents,) + self.obs_shape)
        reward_batch = reward_batch.reshape((batch_size, 1, 1)).repeat(self.n_agents, axis=1)
        is_done_batch = is_done_batch.reshape((batch_size, 1, 1)).repeat(self.n_agents, axis=1)
        sample_weights = sample_weights.reshape((batch_size, 1))
        batch = {self.obs_ph: obs_batch, self.actions_ph: act_batch,
                 self.rewards_ph: reward_batch,
                 self.is_done_ph: is_done_batch, self.next_obs_ph: next_obs_batch,
                 self.sample_weights_ph: sample_weights,
                 self.batch_size_ph: batch_size}
        return [batch, sample_idxes]

    def train(self):
        mem_free_steps = 0
        last_mean_rewards = -100500
        epoch_num = 0
        frame = 0
        update_time = 0
        play_time = 0

        start_time = time.time()
        total_time = 0
        self.load_weights_into_target_network()
        for _ in range(0, self.config['num_steps_fill_buffer']):
            self.play_steps(self.steps_num, self.epsilon)
        steps_per_epoch = self.config['steps_per_epoch']
        num_epochs_to_copy = self.config['num_epochs_to_copy']
        frame = 0
        play_time = 0
        update_time = 0
        rewards = []
        shaped_rewards = []
        steps = []
        losses = deque([], maxlen=100)

        while True:
            epoch_num = self.update_epoch()
            t_play_start = time.time()
            self.epsilon = self.epsilon_processor(frame)
            self.beta = self.beta_processor(frame)

            for _ in range(0, steps_per_epoch):
                reward, shaped_reward, step, info = self.play_steps(self.steps_num, self.epsilon)
                if reward != None:
                    self.game_lengths.append(step)
                    self.game_rewards.append(reward)
                    game_res = info.get('battle_won', 0.5)
                    self.game_scores.append(game_res)
                    # shaped_rewards.append(shaped_reward)

            t_play_end = time.time()
            play_time += t_play_end - t_play_start

            # train
            frame = frame + steps_per_epoch
            t_start = time.time()
            if self.is_prioritized:
                batch, idxes = self.sample_prioritized_batch(self.exp_buffer, batch_size=self.batch_size,
                                                             beta=self.beta)
                _, loss_t, errors_update, lr_mul = self.sess.run(
                    [self.train_op, self.td_loss_mean, self.abs_errors, self.lr_multiplier], batch)
                self.exp_buffer.update_priorities(idxes, errors_update)
            else:
                batch = self.sample_batch(self.exp_buffer, batch_size=self.batch_size)
                _, loss_t, lr_mul = self.sess.run(
                    [self.train_op, self.td_loss_mean, self.lr_multiplier], batch)

            losses.append(loss_t)
            t_end = time.time()
            update_time += t_end - t_start
            total_time += update_time
            if frame % 1000 == 0:
                mem_free_steps += 1
                if mem_free_steps == 10:
                    mem_free_steps = 0
                    tr_helpers.free_mem()
                sum_time = update_time + play_time
                print('frames per seconds: ', 1000 / (sum_time))
                self.writer.add_scalar('performance/fps', 1000 / sum_time, frame)
                self.writer.add_scalar('performance/upd_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/td_loss', np.mean(losses), frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/lr', self.learning_rate * lr_mul, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self.writer.add_scalar('info/epsilon', self.epsilon, frame)

                self.logger.log_stat("whirl/performance/fps", 1000 / sum_time, self.num_env_steps_train)
                self.logger.log_stat("whirl/performance/upd_time", update_time, self.num_env_steps_train)
                self.logger.log_stat("whirl/performance/play_time", play_time, self.num_env_steps_train)
                self.logger.log_stat("losses/td_loss", np.mean(losses), self.num_env_steps_train)
                self.logger.log_stat("whirl/info/last_lr", self.learning_rate*lr_mul, self.num_env_steps_train)
                self.logger.log_stat("whirl/info/lr_mul", lr_mul, self.num_env_steps_train)
                self.logger.log_stat("whirl/epochs", epoch_num, self.num_env_steps_train)
                self.logger.log_stat("whirl/epsilon", self.epsilon, self.num_env_steps_train)

                if self.is_prioritized:
                    self.writer.add_scalar('beta', self.beta, frame)

                update_time = 0
                play_time = 0
                num_games = len(self.game_rewards)
                if num_games > 10:
                    mean_rewards = np.sum(self.game_rewards) / num_games
                    mean_lengths = np.sum(self.game_lengths) / num_games
                    mean_scores = np.mean(self.game_scores)
                    self.writer.add_scalar('rewards/mean', mean_rewards, frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/mean', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    self.logger.log_stat("whirl/rewards/mean", np.asscalar(mean_rewards), self.num_env_steps_train)
                    self.logger.log_stat("whirl/rewards/time", mean_rewards, total_time)
                    self.logger.log_stat("whirl/episode_lengths/mean", np.asscalar(mean_lengths), self.num_env_steps_train)
                    self.logger.log_stat("whirl/episode_lengths/time", mean_lengths, total_time)
                    self.logger.log_stat("whirl/win_rate/mean", np.asscalar(mean_scores), self.num_env_steps_train)
                    self.logger.log_stat("whirl/win_rate/time", np.asscalar(mean_scores), total_time)

                    if mean_rewards > last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                        if last_mean_rewards > self.config['score_to_win']:
                            print('network won!')
                            return last_mean_rewards, epoch_num

            if frame % num_epochs_to_copy == 0:
                self.load_weights_into_target_network()

            if epoch_num >= self.max_epochs:
                print('Max epochs reached')
                self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(
                    np.sum(self.game_rewards) / len(self.game_rewards)))
                return last_mean_rewards, epoch_num
