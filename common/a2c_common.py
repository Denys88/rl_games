import common.tr_helpers as tr_helpers
import numpy as np
import collections
import time
from collections import deque, OrderedDict
import gym
import common.vecenv as vecenv
from datetime import datetime
from tensorboardX import SummaryWriter

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

class A2CBase:
    def __init__(self, base_name, observation_space, action_space, config):
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
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.dones = np.asarray([False]*self.num_actors *self.num_agents, dtype=np.bool)
        self.current_rewards = np.asarray([0]*self.num_actors *self.num_agents, dtype=np.float32)
        self.current_lengths = np.asarray([0]*self.num_actors *self.num_agents, dtype=np.float32)
        self.games_to_log = self.config.get('games_to_track', 100)
        self.game_rewards = deque([], maxlen=self.games_to_log)
        self.game_lengths = deque([], maxlen=self.games_to_log)
        self.game_scores = deque([], maxlen=self.games_to_log)   
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.is_rnn = False
        self.states = None
        self.batch_size = self.steps_num * self.num_actors * self.num_agents
        self.batch_size_envs = self.steps_num * self.num_actors
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.num_env_steps_train = 0
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))

        self.curiosity_config = self.config.get('rnd_config', None)
        self.has_curiosity = self.curiosity_config is not None
        if self.has_curiosity:
            self.curiosity_gamma = self.curiosity_config['gamma']
            self.curiosity_lr = self.curiosity_config['lr']
            self.curiosity_rewards = deque([], maxlen=self.games_to_log)
            self.curiosity_mins = deque([], maxlen=self.games_to_log)
            self.curiosity_maxs = deque([], maxlen=self.games_to_log)
            self.rnd_adv_coef = self.curiosity_config.get('adv_coef', 1.0)

        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']

        # WHiRL test reward addition
        self.test_n_epochs = self.config.get('test_n_epochs', 2000)
        self.steps_num_test = self.config.get('steps_num_test', 10*65)


    def calc_returns_with_rnd(self, mb_returns, last_intrinsic_values, mb_intrinsic_values, mb_intrinsic_rewards):
        mb_intrinsic_advs = np.zeros_like(mb_intrinsic_rewards)
        lastgaelam = 0
        self.curiosity_rewards.append(np.sum(np.mean(mb_intrinsic_rewards, axis=1)))
        self.curiosity_mins.append(np.min(mb_intrinsic_rewards, axis=1))
        self.curiosity_maxs.append(np.max(mb_intrinsic_rewards, axis=1))
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextvalues = last_intrinsic_values
            else:
                nextvalues = mb_intrinsic_values[t+1]
            
            delta = mb_intrinsic_rewards[t] + self.curiosity_gamma * nextvalues - mb_intrinsic_values[t]
            mb_intrinsic_advs[t] = lastgaelam = delta + self.curiosity_gamma * self.tau * lastgaelam

        mb_intrinsic_returns = mb_intrinsic_advs + mb_intrinsic_values
        mb_returns = np.concatenate([f[..., np.newaxis] for f in [mb_returns, mb_intrinsic_returns]], axis=-1)
        return mb_returns

    def update_epoch(self):
        pass
    def save(self, fn):
        pass

    def restore(self, fn):
        pass

    def get_action_values(self, obs):
        pass

    def get_masked_action_values(self, obs, action_masks):
        pass

    def get_action_values_test(self, obs):
        pass

    def get_masked_action_values_test(self, obs, action_masks):
        pass

    def get_values(self, obs):
        pass

    def get_weights(self):
        pass
    
    def set_weights(self, weights):
        pass

    def play_steps(self):
        pass

    def train(self):       
        pass

    def train_epoch(self):
        pass

    def train_actor_critic(self, dict):
        pass 

    def get_intrinsic_reward(self, obs):
        pass

    def train_intrinsic_reward(self, dict):
        pass

class DiscreteA2CBase(A2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        A2CBase.__init__(self, base_name, observation_space, action_space, config)
        self.actions_num = action_space.n

    def play_steps(self):
        # here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        if self.has_curiosity:
            mb_intrinsic_rewards = []
        mb_states = []
        epinfos = []

        # for n in range number of steps
        for _ in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()

            if self.use_action_masks:
                actions, values, neglogpacs, logits, self.states = self.get_masked_action_values(self.obs, masks)
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

            # Increase step count by self.num_actors (WHIRL)
            self.num_env_steps_train += self.num_actors

            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs)
                mb_intrinsic_rewards.append(intrinsic_reward)
            self.current_rewards += rewards

            self.current_lengths += 1
            for reward, length, done, info in zip(self.current_rewards[::self.num_agents], self.current_lengths[::self.num_agents], self.dones[::self.num_agents], infos):
                if done:
                    game_res = info.get('battle_won', 0.5)
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)
                    self.game_scores.append(game_res)

            self.current_rewards = self.current_rewards * (1.0 - self.dones)
            self.current_lengths = self.current_lengths * (1.0 - self.dones)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)
            mb_rewards.append(shaped_rewards)

        #using openai baseline approach
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.long)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_states = np.asarray(mb_states, dtype=np.float32)
        last_values = self.get_values(self.obs)
        last_values = np.squeeze(last_values)

        if self.has_curiosity:
            mb_intrinsic_values = mb_values[:,:,1]
            mb_extrinsic_values = mb_values[:,:,0]
            last_intrinsic_values = last_values[:, 1]
            last_extrinsic_values = last_values[:, 0]
        else:
            mb_extrinsic_values = mb_values
            last_extrinsic_values = last_values

        mb_advs = np.zeros_like(mb_rewards)

        lastgaelam = 0
        
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_extrinsic_values

        if self.has_curiosity:
            mb_intrinsic_rewards = np.asarray(mb_intrinsic_rewards, dtype=np.float32)
            mb_returns = self.calc_returns_with_rnd(mb_returns, last_intrinsic_values, mb_intrinsic_values, mb_intrinsic_rewards)

        batch_dict = {
            'obs' : mb_obs,
            'returns' : mb_returns,
            'dones' : mb_dones,
            'actions' : mb_actions,
            'values' : mb_values,
            'neglogpacs' : mb_neglogpacs,
        }
        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}
        if self.network.is_rnn():
            batch_dict['states'] = mb_states

        return batch_dict

    def train_epoch(self):
        play_time_start = time.time()
        batch_dict = self.play_steps() 

        obses = batch_dict['obs']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        actions = batch_dict['actions']
        values = batch_dict['values']
        neglogpacs = batch_dict['neglogpacs']
        lstm_states = batch_dict.get('states', None)

        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values
        if self.has_curiosity:
            self.train_intrinsic_reward(batch_dict)
            advantages[:,1] = advantages[:,1] * self.rnd_adv_coef
            advantages = np.sum(advantages, axis=1)
                      
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        a_losses = []
        c_losses = []
        entropies = []
        kls = []

        if self.network.is_rnn():
            total_games = self.batch_size // self.seq_len
            num_games_batch = self.minibatch_size // self.seq_len
            game_indexes = np.arange(total_games)
            flat_indexes = np.arange(total_games * self.seq_len).reshape(total_games, self.seq_len)
            lstm_states = lstm_states[::self.seq_len]
            for _ in range(0, self.mini_epochs_num):
                np.random.shuffle(game_indexes)

                for i in range(0, self.num_minibatches):
                    batch = range(i * num_games_batch, (i + 1) * num_games_batch)
                    mb_indexes = game_indexes[batch]
                    mbatch = flat_indexes[mb_indexes].ravel()                        

                    input_dict = {}
                    input_dict['old_values'] = values[mbatch]
                    input_dict['old_logp_actions'] = neglogpacs[mbatch]
                    input_dict['advantages'] = advantages[mbatch]
                    input_dict['returns'] = returns[mbatch]
                    input_dict['actions'] = actions[mbatch]
                    input_dict['obs'] = obses[mbatch]
                    input_dict['masks'] = dones[mbatch]
                    input_dict['states'] = lstm_states[batch]
                    input_dict['learning_rate'] = self.last_lr

                    a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(input_dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)    
        else:
            for _ in range(0, self.mini_epochs_num):
                permutation = np.random.permutation(self.batch_size)
                obses = obses[permutation]
                returns = returns[permutation]
                
                actions = actions[permutation]
                values = values[permutation]
                neglogpacs = neglogpacs[permutation]
                advantages = advantages[permutation]

                for i in range(0, self.num_minibatches):
                    batch = range(i * self.minibatch_size, (i + 1) * self.minibatch_size)
                    input_dict = {}
                    input_dict['old_values'] = values[batch]
                    input_dict['old_logp_actions'] = neglogpacs[batch]
                    input_dict['advantages'] = advantages[batch]
                    input_dict['returns'] = returns[batch]
                    input_dict['actions'] = actions[batch]
                    input_dict['obs'] = obses[batch]
                    input_dict['masks'] = dones[batch]
                    input_dict['learning_rate'] = self.last_lr
                    
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(input_dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    # def test(self):
    #
    #     game_scores_test = []
    #     rewards_test = []
    #     self.vec_env.reset()
    #     for _ in range(self.steps_num_test):
    #         #if self.network.is_rnn():
    #         #    mb_states.append(self.states)
    #
    #         if self.use_action_masks:
    #             masks = self.vec_env.get_action_masks()
    #
    #         if self.use_action_masks:
    #             actions, values, neglogpacs, logits, self.states = self.get_masked_action_values_test(self.obs, masks)
    #         else:
    #             actions, values, neglogpacs, self.states = self.get_action_values_test(self.obs)
    #
    #         actions = np.squeeze(actions)
    #         _, rewards, dones, infos = self.vec_env.step(actions)
    #
    #         # Increase step count by self.num_actors (WHIRL)
    #         # self.num_env_steps_test += self.num_actors
    #
    #
    #
    #         self.current_lengths += 1
    #         for rewards, done, info in zip(rewards[::self.num_agents], dones[::self.num_agents], infos):
    #             if done:
    #                 game_res = info.get('battle_won', 0.5)
    #                 game_scores_test.append(game_res)
    #                 rewards_test.append(rewards)
    #
    #     self.vec_env.reset()
    #     return np.mean(rewards_test), np.mean(game_scores_test)

    def train(self):
        last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        frame = 0
        self.obs = self.vec_env.reset()
        while True:
            epoch_num = self.update_epoch()
            frame += self.batch_size_envs

            # if not(epoch_num-1 % self.test_n_epochs):
            #     # WHIRL TEST ROUTINE
            #     rew, wrate = self.test()
            #     self.logger.log_stat("whirl/test_reward/mean", rew, self.num_env_steps_train)
            #     self.logger.log_stat("whirl/test_win_rate/mean", wrate, self.num_env_steps_train)

            play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            if True:
                scaled_time = self.num_agents * sum_time
                print('frames per seconds: ', self.batch_size / scaled_time)
                self.writer.add_scalar('performance/fps', self.batch_size / scaled_time, frame)
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

                self.logger.log_stat("whirl/performance/fps", self.batch_size / scaled_time, self.num_env_steps_train)
                self.logger.log_stat("whirl/performance/upd_time", update_time, self.num_env_steps_train)
                self.logger.log_stat("whirl/performance/play_time", play_time, self.num_env_steps_train)
                self.logger.log_stat("whirl/losses/a_loss", np.mean(a_losses), self.num_env_steps_train)
                self.logger.log_stat("whirl/losses/c_loss", np.mean(c_losses), self.num_env_steps_train)
                self.logger.log_stat("whirl/losses/entropy", np.mean(entropies), self.num_env_steps_train)
                self.logger.log_stat("whirl/info/last_lr", last_lr * lr_mul, self.num_env_steps_train)
                self.logger.log_stat("whirl/info/lr_mul", lr_mul, self.num_env_steps_train)
                self.logger.log_stat("whirl/info/e_clip", self.e_clip * lr_mul, self.num_env_steps_train)
                self.logger.log_stat("whirl/info/kl", np.mean(kls), self.num_env_steps_train)
                self.logger.log_stat("whirl/epochs", epoch_num, self.num_env_steps_train)

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
                    self.logger.log_stat("whirl/rewards/mean", mean_rewards, self.num_env_steps_train)
                    self.logger.log_stat("whirl/rewards/time", mean_rewards, total_time)
                    self.logger.log_stat("whirl/episode_lengths/mean", mean_lengths, self.num_env_steps_train)
                    self.logger.log_stat("whirl/episode_lengths/time", mean_lengths, total_time)
                    self.logger.log_stat("whirl/win_rate/mean", mean_scores, self.num_env_steps_train)
                    self.logger.log_stat("whirl/win_rate/time",mean_scores, total_time)

                    if self.has_curiosity:
                        if len(self.curiosity_rewards) > 0:
                            mean_cur_rewards = np.mean(self.curiosity_rewards)
                            mean_min_rewards = np.mean(self.curiosity_mins)
                            mean_max_rewards = np.mean(self.curiosity_maxs)
                            self.writer.add_scalar('rnd/rewards_sum', mean_cur_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_min', mean_min_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_max', mean_max_rewards, frame)
                    if rep_count % 10 == 0:
                        self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                        rep_count += 1

                    if mean_rewards > last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'] + self.env_name)
                        if last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                            return last_mean_rewards, epoch_num

                if epoch_num > self.max_epochs:
                    self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return last_mean_rewards, epoch_num                               
                update_time = 0


class ContinuousA2CBase(A2CBase):
    def __init__(self, base_name, observation_space, action_space, config, init_arrays=True):
        A2CBase.__init__(self, base_name, observation_space, action_space, config)
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.actions_low = action_space.low
        self.actions_high = action_space.high
        self.actions_num = action_space.shape[0]
        batch_size = self.num_agents * self.num_actors
        if init_arrays:
            self.mb_obs = np.zeros((self.steps_num, batch_size) + self.state_shape, dtype = observation_space.dtype)
            self.mb_rewards = np.zeros((self.steps_num, batch_size), dtype = np.float32)
            self.mb_actions = np.zeros((self.steps_num, batch_size, self.actions_num), dtype = np.float32)
            self.mb_dones = np.zeros((self.steps_num, batch_size), dtype  = np.bool)
            self.mb_neglogpacs = np.zeros((self.steps_num, batch_size), dtype = np.float32)
            self.mb_mus = np.zeros((self.steps_num, batch_size, self.actions_num), dtype = np.float32)
            self.mb_sigmas = np.zeros((self.steps_num, batch_size, self.actions_num), dtype = np.float32)
            self.mb_values = np.zeros((self.steps_num, batch_size), dtype = np.float32)

            if self.has_curiosity:
                self.mb_values = np.zeros((self.steps_num, batch_size, 2), dtype = np.float32)
                self.mb_intrinsic_rewards = np.zeros((self.steps_num, batch_size), dtype = np.float32)
    def play_steps(self):
        # Here, we init the lists that will contain the mb of experiences
        #mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mus, mb_sigmas = [],[],[],[],[],[],[],[]
        mb_states = []
        epinfos = []
        #'''
        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_actions = self.mb_actions
        mb_values = self.mb_values
        mb_neglogpacs = self.mb_neglogpacs
        mb_mus = self.mb_mus
        mb_sigmas = self.mb_sigmas
        mb_dones = self.mb_dones
        if self.has_curiosity:
            mb_intrinsic_rewards = self.mb_intrinsic_rewards
        #'''
        #mb_states = self.mb_states
        # For n in range number of steps
        for n in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)
            actions, values, neglogpacs, mu, sigma, self.states = self.get_action_values(self.obs)
            values = np.squeeze(values)
            neglogpacs = np.squeeze(neglogpacs)
     
            
            mb_obs[n,:] = self.obs
            mb_dones[n,:] = self.dones


            self.obs[:], rewards, self.dones, infos = self.vec_env.step(rescale_actions(self.actions_low, self.actions_high, np.clip(actions, -1.0, 1.0)))
            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs)
                mb_intrinsic_rewards[n,:] = intrinsic_reward
            self.current_rewards += rewards
            self.current_lengths += 1
            for reward, length, done in zip(self.current_rewards, self.current_lengths, self.dones):
                if done:
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)

            mb_actions[n,:] = actions
            mb_values[n,:] = values
            mb_neglogpacs[n,:] = neglogpacs
            
            mb_mus[n,:] = mu
            mb_sigmas[n,:] = sigma
            mb_rewards[n,:] = shaped_rewards
            
            self.current_rewards = self.current_rewards * (1.0 - self.dones)
            self.current_lengths = self.current_lengths * (1.0 - self.dones)

        last_values = self.get_values(self.obs)
        last_values = np.squeeze(last_values)


        if self.has_curiosity:
            mb_intrinsic_values = mb_values[:,:,1]
            mb_extrinsic_values = mb_values[:,:,0]
            last_intrinsic_values = last_values[:, 1]
            last_extrinsic_values = last_values[:, 0]
        else:
            mb_extrinsic_values = mb_values
            last_extrinsic_values = last_values

        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_extrinsic_values

        if self.has_curiosity:
            mb_intrinsic_rewards = np.asarray(mb_intrinsic_rewards, dtype=np.float32)
            mb_returns = self.calc_returns_with_rnd(mb_returns, last_intrinsic_values, mb_intrinsic_values, mb_intrinsic_rewards)

        batch_dict = {
            'obs' : mb_obs,
            'returns' : mb_returns,
            'dones' : mb_dones,
            'actions' : mb_actions,
            'values' : mb_values,
            'neglogpacs' : mb_neglogpacs,
            'mus' : mb_mus,
            'sigmas' : mb_sigmas,
        }
        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}
        if self.network.is_rnn():
            batch_dict['states'] = mb_states

        return batch_dict

    def train_epoch(self):
        play_time_start = time.time()
        batch_dict = self.play_steps()
        obses = batch_dict['obs']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        actions = batch_dict['actions']
        values = batch_dict['values']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        lstm_states = batch_dict.get('states', None)

        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.has_curiosity:
            self.train_intrinsic_reward(batch_dict)
            advantages[:,1] = advantages[:,1] * self.rnd_adv_coef
            advantages = np.sum(advantages, axis=1)            
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        if self.network.is_rnn():
            total_games = self.batch_size // self.seq_len
            num_games_batch = self.minibatch_size // self.seq_len
            game_indexes = np.arange(total_games)
            flat_indexes = np.arange(total_games * self.seq_len).reshape(total_games, self.seq_len)
            lstm_states = lstm_states[::self.seq_len]
            for _ in range(0, self.mini_epochs_num):
                np.random.shuffle(game_indexes)

                for i in range(0, self.num_minibatches):
                    batch = range(i * num_games_batch, (i + 1) * num_games_batch)
                    mb_indexes = game_indexes[batch]
                    mbatch = flat_indexes[mb_indexes].ravel()                        

                    input_dict = {}
                    input_dict['old_values'] = values[mbatch]
                    input_dict['old_logp_actions'] = neglogpacs[mbatch]
                    input_dict['advantages'] = advantages[mbatch]
                    input_dict['returns'] = returns[mbatch]
                    input_dict['actions'] = actions[mbatch]
                    input_dict['obs'] = obses[mbatch]
                    input_dict['masks'] = dones[mbatch]
                    input_dict['mu'] = mus[mbatch]
                    input_dict['sigma'] = sigmas[mbatch]
                    input_dict['states'] = lstm_states[batch]
                    input_dict['learning_rate'] = self.last_lr

                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(input_dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)        
                    mus[mbatch] = cmu
                    sigmas[mbatch] = csigma
                    if self.bounds_loss is not None:
                        b_losses.append(b_loss)                            

        else:
            for _ in range(0, self.mini_epochs_num):
                permutation = np.random.permutation(self.batch_size)
                obses = obses[permutation]
                returns = returns[permutation]
                
                actions = actions[permutation]
                values = values[permutation]
                neglogpacs = neglogpacs[permutation]
                advantages = advantages[permutation]
                mus = mus[permutation]
                sigmas = sigmas[permutation]
                for i in range(0, self.num_minibatches):
                    batch = range(i * self.minibatch_size, (i + 1) * self.minibatch_size)
                    input_dict = {}
                    input_dict['old_values'] = values[batch]
                    input_dict['old_logp_actions'] = neglogpacs[batch]
                    input_dict['advantages'] = advantages[batch]
                    input_dict['returns'] = returns[batch]
                    input_dict['actions'] = actions[batch]
                    input_dict['obs'] = obses[batch]
                    input_dict['masks'] = dones[batch]
                    input_dict['mu'] = mus[batch]
                    input_dict['sigma'] = sigmas[batch]
                    input_dict['learning_rate'] = self.last_lr
                    
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(input_dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)
                    mus[batch] = cmu
                    sigmas[batch] = csigma
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss) 

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


    def train(self):
        last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        frame = 0
        self.obs = self.vec_env.reset()
        while True:
            epoch_num = self.update_epoch()
            frame += self.batch_size_envs

            play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            if True:
                scaled_time = self.num_agents * sum_time
                print('frames per seconds: ', self.batch_size / scaled_time)
                self.writer.add_scalar('performance/fps', self.batch_size / scaled_time, frame)
                self.writer.add_scalar('performance/upd_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/a_loss', np.mean(a_losses), frame)
                self.writer.add_scalar('losses/c_loss', np.mean(c_losses), frame)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', np.mean(b_losses), frame)
                self.writer.add_scalar('losses/entropy', np.mean(entropies), frame)
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', np.mean(kls), frame)
                self.writer.add_scalar('epochs', epoch_num, frame)
                
                if len(self.game_rewards) > 0:
                    mean_rewards = np.mean(self.game_rewards)
                    mean_lengths = np.mean(self.game_lengths)
                    #mean_scores = np.mean(self.game_scores)
                    self.writer.add_scalar('rewards/mean', mean_rewards, frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/mean', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    #self.writer.add_scalar('win_rate/mean', mean_scores, frame)
                    #self.writer.add_scalar('win_rate/time', mean_scores, total_time)
                    if self.has_curiosity:
                        if len(self.curiosity_rewards) > 0:
                            mean_cur_rewards = np.mean(self.curiosity_rewards)
                            mean_min_rewards = np.mean(self.curiosity_mins)
                            mean_max_rewards = np.mean(self.curiosity_maxs)
                            self.writer.add_scalar('rnd/rewards_sum', mean_cur_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_min', mean_min_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_max', mean_max_rewards, frame)
                    if rep_count % 10 == 0:
                        self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                        rep_count += 1

                    if mean_rewards > last_mean_rewards:
                        print('saving next best rewards: ', mean_rewards)
                        last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'] + self.env_name)
                        if last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                            return last_mean_rewards, epoch_num

                if epoch_num > self.max_epochs:
                    self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return last_mean_rewards, epoch_num                               
                update_time = 0