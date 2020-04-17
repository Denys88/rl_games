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
        self.actions_num = action_space.n   
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
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))


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

class DiscreteA2CBase(A2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        A2CBase.__init__(self, base_name, observation_space, action_space, config)

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
            self.current_rewards += rewards

            self.current_lengths += 1
            for reward, length, done, info in zip(self.current_rewards[::self.num_agents], self.current_lengths[::self.num_agents], self.dones[::self.num_agents], infos):
                if done:
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)
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

    def train_epoch(self):
        play_time_start = time.time()
        obses, returns, dones, actions, values, neglogpacs, lstm_states, _ = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values
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
    def __init__(self, base_name, observation_space, action_space, config):
        A2CBase.__init__(self, base_name, observation_space, action_space, config)

    def play_steps(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mus, mb_sigmas = [],[],[],[],[],[],[],[]
        mb_states = []
        epinfos = []
        # For n in range number of steps
        for _ in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)
            actions, values, neglogpacs, mu, sigma, self.states = self.get_action_values(self.obs)
            actions = np.squeeze(actions)
            values = np.squeeze(values)
            neglogpacs = np.squeeze(neglogpacs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones.copy())
            mb_mus.append(mu)
            mb_sigmas.append(sigma)

            self.obs[:], rewards, self.dones, infos = self.vec_env.step(rescale_actions(self.actions_low, self.actions_high, np.clip(actions, -1.0, 1.0)))
            self.current_rewards += rewards
            self.current_lengths += 1

            for reward, length, done in zip(self.current_rewards, self.current_lengths, self.dones):
                if done:
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)
            mb_rewards.append(shaped_rewards)

            self.current_rewards = self.current_rewards * (1.0 - self.dones)
            self.current_lengths = self.current_lengths * (1.0 - self.dones)
        #using openai baseline approach
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_mus = np.asarray(mb_mus, dtype=np.float32)
        mb_sigmas = np.asarray(mb_sigmas, dtype=np.float32)
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
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_mus, mb_sigmas, mb_states )), epinfos)
        else:
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_mus, mb_sigmas)), None, epinfos)

        return result

    def train_epoch(self):
        play_time_start = time.time()
        obses, returns, dones, actions, values, neglogpacs, lstm_states, _ = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values
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
                    input_dict['mu'] = obses[mbatch]
                    input_dict['sigma'] = dones[mbatch]
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
                    input_dict['mu'] = obses[batch]
                    input_dict['sigma'] = dones[batch]
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