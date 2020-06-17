from rl_games.common import tr_helpers
from rl_games.common import vecenv

import numpy as np
import collections
import time
from collections import deque, OrderedDict
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

class A2CBase:
    def __init__(self, base_name, config):
        self.config = config
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']
        self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        self.env_info = self.vec_env.get_env_info()

        print('Env info:')
        print(self.env_info)

        self.observation_space = self.env_info['observation_space']
        observation_shape = self.observation_space.shape
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)
        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)

        self.name = base_name
        
        self.ppo = config['ppo']

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.is_polynom_decay_lr = config['lr_schedule'] == 'polynom_decay'
        self.is_exp_decay_lr = config['lr_schedule'] == 'exp_decay'

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
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

        self.games_to_log = self.config.get('games_to_track', 100)
        self.game_rewards = deque([], maxlen=self.games_to_log)
        self.game_lengths = deque([], maxlen=self.games_to_log)
        self.game_scores = deque([], maxlen=self.games_to_log)   
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.is_rnn = self.network.is_rnn()
        self.states = self.network.get_default_rnn_state()
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

        self.is_tensor_obses = False

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8)
        self.mb_obs = torch.zeros((self.steps_num, batch_size) + self.state_shape, dtype=torch_dtype).cuda()
        self.mb_rewards = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
        self.mb_values = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
        self.mb_dones = torch.zeros((self.steps_num, batch_size), dtype = torch.uint8)
        self.mb_neglogpacs = torch.zeros((self.steps_num, batch_size), dtype = torch.float32).cuda()

        if self.is_rnn:
            num_seqs = self.steps_num//self.seq_len * batch_size
            self.mb_states = 
            [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32).cuda() for s in self.states]


    def calc_returns_with_rnd(self, mb_returns, last_intrinsic_values, mb_intrinsic_values, mb_intrinsic_rewards):
        mb_intrinsic_advs = torch.zeros_like(mb_intrinsic_rewards)
        lastgaelam = 0

        self.curiosity_rewards.append(torch.sum(torch.mean(mb_intrinsic_rewards, axis=1)))
        self.curiosity_mins.append(torch.min(mb_intrinsic_rewards))
        self.curiosity_maxs.append(torch.max(mb_intrinsic_rewards))

        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextvalues = last_intrinsic_values
            else:
                nextvalues = mb_intrinsic_values[t+1]
            
            delta = mb_intrinsic_rewards[t] + self.curiosity_gamma * nextvalues - mb_intrinsic_values[t]
            mb_intrinsic_advs[t] = lastgaelam = delta + self.curiosity_gamma * self.tau * lastgaelam

        mb_intrinsic_returns = mb_intrinsic_advs + mb_intrinsic_values
        mb_returns = torch.stack((mb_returns, mb_intrinsic_returns), dim=-1)
        return mb_returns

    def env_reset(self):
        obs = self.vec_env.reset()
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        else:
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).cuda()
            else:
                obs = torch.FloatTensor(obs).cuda()
        return obs

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

    def get_intrinsic_reward(self, obs):
        pass

    def train_intrinsic_reward(self, dict):
        pass

    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        if len(obs_batch.size()) == 3:
            obs_batch = obs_batch.permute((0, 2, 1))
        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

class DiscreteA2CBase(A2CBase):
    def __init__(self, base_name, config):
        A2CBase.__init__(self, base_name, config)
        self.actions_num = self.env_info['action_space'].n
        self.init_tensors()

    def init_tensors(self):
        A2CBase.init_tensors(self)
        batch_size = self.num_agents * self.num_actors
        self.mb_actions = torch.zeros((self.steps_num, batch_size), dtype = torch.long).cuda()
        if self.has_curiosity:
            self.mb_values = torch.zeros((self.steps_num, batch_size, 2), dtype = torch.float32)
            self.mb_intrinsic_rewards = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
    
    def env_step(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = self.vec_env.step(actions)
        
        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            return torch.from_numpy(obs).cuda(), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def play_steps(self):
        mb_states = []
        epinfos = []

        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_actions = self.mb_actions
        mb_values = self.mb_values
        mb_neglogpacs = self.mb_neglogpacs
        mb_dones = self.mb_dones
        mb_states = self.mb_states

        if self.has_curiosity:
            mb_intrinsic_rewards = self.mb_intrinsic_rewards

        # For n in range number of steps
        if self.is_rnn:
            mb_masks = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
            indices = torch.zeros((batch_size), dtype = torch.long)

        for n in range(self.steps_num):
            if self.is_rnn:

                seq_indices = indices % self.seq_len
                state_indices = (seq_indices == 0).nonzero()
                state_pos = indices // self.seq_len

                for s, mb_s in zip(self.states, mb_states)
                    mb_s[:, state_pos[state_indices], :] = s[:, state_indices, :]
                indices += 1

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                actions, values, neglogpacs, _, self.states = self.get_masked_action_values(self.obs, masks)
            else:
                actions, values, neglogpacs, self.states = self.get_action_values(self.obs)
                
            values = torch.squeeze(values)
            neglogpacs = torch.squeeze(neglogpacs)
     
            mb_obs[n,:] = self.obs
            mb_dones[n,:] = self.dones

            self.obs, rewards, self.dones, infos = self.env_step(actions)

            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs)
                mb_intrinsic_rewards[n,:] = intrinsic_reward
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero()
            done_indices = all_done_indices[::self.num_agents]
            if self.is_rnn:
                for i in range(len(self.states)):
                    self.states[i][all_done_indices,:,:,:] = self.states[i][all_done_indices,:,:,:] * 0.0
                     

            self.game_rewards.extend(self.current_rewards[done_indices])
            self.game_lengths.extend(self.current_lengths[done_indices])

            for ind in done_indices:
                game_res = infos[ind//self.num_agents].get('battle_won', 0.0)
                self.game_scores.append(game_res)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)

            mb_actions[n,:] = actions
            mb_values[n,:] = values
            mb_neglogpacs[n,:] = neglogpacs
            mb_rewards[n,:] = shaped_rewards
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)
        last_values = torch.squeeze(last_values)

        if self.has_curiosity:
            mb_intrinsic_values = mb_values[:,:,1]
            mb_extrinsic_values = mb_values[:,:,0]
            last_intrinsic_values = last_values[:, 1]
            last_extrinsic_values = last_values[:, 0]
        else:
            mb_extrinsic_values = mb_values
            last_extrinsic_values = last_values

        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        fdones = self.dones.float()
        mb_fdones = mb_dones.float()

        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_extrinsic_values

        if self.has_curiosity:
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
        returns = batch_dict['returns'].cuda()
        dones = batch_dict['dones'].cuda()
        values = batch_dict['values'].cuda()
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        lstm_states = batch_dict.get('states', None)

        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values

        if self.has_curiosity:
            self.train_intrinsic_reward(batch_dict)
            advantages[:,1] = advantages[:,1] * self.rnd_adv_coef
            advantages = torch.sum(advantages, axis=1)
                      
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
                    batch = torch.range(i * self.minibatch_size, (i + 1) * self.minibatch_size - 1, dtype=torch.long, device='cuda:0')
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
                permutation = torch.randperm(self.batch_size, dtype=torch.long, device='cuda:0')
                obses = obses[permutation]
                returns = returns[permutation]
                
                actions = actions[permutation]
                values = values[permutation]
                neglogpacs = neglogpacs[permutation]
                advantages = advantages[permutation]

                for i in range(0, self.num_minibatches):
                    batch = torch.range(i * self.minibatch_size, (i + 1) * self.minibatch_size - 1, dtype=torch.long, device='cuda:0')
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
        self.obs = self.env_reset()

        while True:
            epoch_num = self.update_epoch()
            frame += self.batch_size_envs

            play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            if True:
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                if self.print_stats:
                    fps_step = self.batch_size / scaled_play_time
                    fps_total = self.batch_size / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')
                    
                self.writer.add_scalar('performance/total_fps', self.batch_size / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', self.batch_size / scaled_play_time, frame)
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
                    self.writer.add_scalar('rewards/frame', mean_rewards, frame)
                    self.writer.add_scalar('rewards/iter', mean_rewards, epoch_num)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    self.writer.add_scalar('win_rate/mean', mean_scores, frame)
                    self.writer.add_scalar('win_rate/time', mean_scores, total_time)

                    if self.has_curiosity:
                        if len(self.curiosity_rewards) > 0:
                            mean_cur_rewards = np.mean(self.curiosity_rewards)
                            mean_min_rewards = np.mean(self.curiosity_mins)
                            mean_max_rewards = np.mean(self.curiosity_maxs)
                            self.writer.add_scalar('rnd/rewards_sum', mean_cur_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_min', mean_min_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_max', mean_max_rewards, frame)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards > last_mean_rewards and epoch_num >= self.save_best_after:
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
    def __init__(self, base_name, config):
        A2CBase.__init__(self, base_name, config)
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.bounds_loss_coef = config.get('bounds_loss_coef', None)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low).float().cuda()
        self.actions_high = torch.from_numpy(action_space.high).float().cuda()
        self.init_tensors()

    def init_tensors(self):
        A2CBase.init_tensors(self)
        batch_size = self.num_agents * self.num_actors
        self.mb_actions = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()
        self.mb_mus = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()
        self.mb_sigmas = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()
        if self.has_curiosity:
            self.mb_values = torch.zeros((self.steps_num, batch_size, 2), dtype = torch.float32)
            self.mb_intrinsic_rewards = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)

    def play_steps(self):
        mb_states = []
        epinfos = []

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

        # For n in range number of steps
        for n in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)
            actions, values, neglogpacs, mu, sigma, self.states = self.get_action_values(self.obs)
            values = torch.squeeze(values)
            neglogpacs = torch.squeeze(neglogpacs)
     
            mb_obs[n,:] = self.obs
            mb_dones[n,:] = self.dones

            self.obs, rewards, self.dones, infos = self.env_step(actions)

            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs)
                mb_intrinsic_rewards[n,:] = intrinsic_reward
            
            self.current_rewards += rewards
            self.current_lengths += 1

            done_indices = self.dones.nonzero()[::self.num_agents]
            self.game_rewards.extend(self.current_rewards[done_indices])
            self.game_lengths.extend(self.current_lengths[done_indices])
            
            shaped_rewards = self.rewards_shaper(rewards)
            #epinfos.append(infos)

            mb_actions[n,:] = actions
            mb_values[n,:] = values
            mb_neglogpacs[n,:] = neglogpacs
            
            mb_mus[n,:] = mu
            mb_sigmas[n,:] = sigma
            mb_rewards[n,:] = shaped_rewards
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)
        last_values = torch.squeeze(last_values)

        if self.has_curiosity:
            mb_intrinsic_values = mb_values[:,:,1]
            mb_extrinsic_values = mb_values[:,:,0]
            last_intrinsic_values = last_values[:, 1]
            last_extrinsic_values = last_values[:, 0]
        else:
            mb_extrinsic_values = mb_values
            last_extrinsic_values = last_values

        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        fdones = self.dones.float()
        mb_fdones = mb_dones.float()

        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_extrinsic_values

        if self.has_curiosity:
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
        with torch.no_grad():
            batch_dict = self.play_steps()
            
        obses = batch_dict['obs']
        returns = batch_dict['returns'].cuda()
        dones = batch_dict['dones'].cuda()
        values = batch_dict['values'].cuda()
        actions = batch_dict['actions']
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
            advantages = torch.sum(advantages, axis=1)  

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
            permutation = torch.randperm(self.batch_size, dtype=torch.long, device='cuda:0')
            obses = obses[permutation]
            returns = returns[permutation]
                
            actions = actions[permutation]
            values = values[permutation]
            neglogpacs = neglogpacs[permutation]
            advantages = advantages[permutation]
            mus = mus[permutation]
            sigmas = sigmas[permutation]

            for _ in range(0, self.mini_epochs_num):
                for i in range(0, self.num_minibatches):
                    batch = torch.range(i * self.minibatch_size, (i + 1) * self.minibatch_size - 1, dtype=torch.long, device='cuda:0')
                    #batch = range(i * self.minibatch_size, (i + 1) * self.minibatch_size)
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
    
    def env_step(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()
        obs, rewards, dones, infos = self.vec_env.step(rescaled_actions)

        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            return torch.from_numpy(obs).cuda(), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def train(self):
        last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        frame = 0
        self.obs = self.env_reset()

        while True:
            epoch_num = self.update_epoch()
            frame += self.batch_size_envs

            play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time

            if True:
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                if self.print_stats:
                    fps_step = self.batch_size / scaled_play_time
                    fps_total = self.batch_size / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

                self.writer.add_scalar('performance/total_fps', self.batch_size / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', self.batch_size / scaled_play_time, frame)
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
                    self.writer.add_scalar('rewards/frame', mean_rewards, frame)
                    self.writer.add_scalar('rewards/iter', mean_rewards, epoch_num)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    #self.writer.add_scalar('win_rate/frame', mean_scores, frame)
                    #self.writer.add_scalar('win_rate/iter', mean_scores, epoch_num)
                    #self.writer.add_scalar('win_rate/time', mean_scores, total_time)

                    if self.has_curiosity:
                        if len(self.curiosity_rewards) > 0:
                            mean_cur_rewards = np.mean(self.curiosity_rewards)
                            mean_min_rewards = np.mean(self.curiosity_mins)
                            mean_max_rewards = np.mean(self.curiosity_maxs)
                            self.writer.add_scalar('rnd/rewards_sum', mean_cur_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_min', mean_min_rewards, frame)
                            self.writer.add_scalar('rnd/rewards_max', mean_max_rewards, frame)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards > last_mean_rewards and epoch_num >= self.save_best_after:
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