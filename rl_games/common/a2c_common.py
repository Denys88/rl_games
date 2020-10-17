from rl_games.common import tr_helpers
from rl_games.common import vecenv

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.self_play_manager import  SelfPlayManager
import numpy as np
import collections
import time
from collections import deque, OrderedDict
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
 
from time import sleep

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
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            self.state_shape = None
            if self.state_space.shape != None:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None


        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
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
        self.seq_len = self.config.get('seq_len', 4)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = self.config['normalize_input']
        self.normalize_reward = self.config.get('normalize_reward', False)

        self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = deque([], maxlen=self.games_to_track)
        self.game_lengths = deque([], maxlen=self.games_to_track)
        self.game_scores = deque([], maxlen=self.games_to_track)   
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.steps_num * self.num_actors * self.num_agents
        self.batch_size_envs = self.steps_num * self.num_actors
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)
        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("%d, %H:%M:%S"))        
        
        if self.normalize_reward:
            self.reward_mean_std = RunningMeanStd((1,))

        # curiosity
        self.curiosity_config = self.config.get('rnd_config', None)
        self.has_curiosity = self.curiosity_config is not None
        if self.has_curiosity:
            self.curiosity_gamma = self.curiosity_config['gamma']
            self.curiosity_lr = self.curiosity_config['lr']
            self.curiosity_rewards = deque([], maxlen=self.games_to_track)
            self.curiosity_mins = deque([], maxlen=self.games_to_track)
            self.curiosity_maxs = deque([], maxlen=self.games_to_track)
            self.rnd_adv_coef = self.curiosity_config.get('adv_coef', 1.0)

        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']

        self.is_tensor_obses = False

        #self_play
        if self.has_self_play_config:
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8)
        self.mb_obs = torch.zeros((self.steps_num, batch_size) + self.obs_shape, dtype=torch_dtype).cuda()

        if self.has_central_value:
            self.mb_vobs = torch.zeros((self.steps_num, self.num_actors) + self.state_shape, dtype=torch_dtype).cuda()
        
        self.mb_rewards = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
        self.mb_values = torch.zeros((self.steps_num, batch_size), dtype = torch.float32)
        self.mb_dones = torch.zeros((self.steps_num, batch_size), dtype = torch.uint8)
        self.mb_neglogpacs = torch.zeros((self.steps_num, batch_size), dtype = torch.float32).cuda()

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()
        if self.is_rnn:
            self.rnn_states = model.get_default_rnn_state()
            batch_size = self.num_agents * self.num_actors
            num_seqs = self.steps_num * batch_size // self.seq_len
            assert((self.steps_num * batch_size // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32).cuda() for s in self.rnn_states]

    def init_rnn_step(self, batch_size, mb_rnn_states):
        mb_rnn_states = self.mb_rnn_states
        mb_rnn_masks = torch.zeros(self.steps_num*batch_size, dtype = torch.float32).cuda()
        steps_mask = torch.arange(0, batch_size * self.steps_num, self.steps_num, dtype=torch.long, device='cuda:0')
        play_mask = torch.arange(0, batch_size, 1, dtype=torch.long, device='cuda:0')
        steps_state = torch.arange(0, batch_size * self.steps_num//self.seq_len, self.steps_num//self.seq_len, dtype=torch.long, device='cuda:0')
        indices = torch.zeros((batch_size), dtype = torch.long).cuda()
        return mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states

    def process_rnn_indices(self, mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states):
        seq_indices = None
        if self.is_rnn:
            if indices.max().item() >= self.steps_num:
                return seq_indices, True
            mb_rnn_masks[indices + steps_mask] = 1
            seq_indices = indices % self.seq_len
            state_indices = (seq_indices == 0).nonzero(as_tuple=False)
            state_pos = indices // self.seq_len
            rnn_indices = state_pos[state_indices] + steps_state[state_indices]
            for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                mb_s[:, rnn_indices, :] = s[:, state_indices, :]
        return seq_indices, False

    def process_rnn_dones(self, all_done_indices, indices, seq_indices):
        if self.is_rnn:
            if len(all_done_indices) > 0:
                all_done_indices = all_done_indices.squeeze(-1)
                shifts = self.seq_len - 1 - seq_indices[all_done_indices]
                indices[all_done_indices] += shifts
                for s in self.rnn_states:
                    s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0
            indices += 1  

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
        
    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).cuda()
            else:
                obs = torch.FloatTensor(obs).cuda()
        return obs
        
    def obs_to_tensors(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self.cast_obs(value)
        else:
            upd_obs = {'obs' : self.cast_obs(obs)}
        
        return upd_obs

    def env_step(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.cpu(), dones.cpu(), infos
        else:
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).float(), torch.from_numpy(dones), infos

    def env_reset(self):
        obs = self.vec_env.reset() 
        obs = self.obs_to_tensors(obs)
        return obs


    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs


    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32)
        self.last_mean_rewards = -100500
        self.obs = self.env_reset()

    def update_epoch(self):
        pass

    def get_action_values(self, obs):
        pass

    def get_masked_action_values(self, obs, action_masks):
        pass

    def get_values(self, obs, actions=None):
        pass


    def play_steps(self):
        pass

    def train(self):       
        pass

    def train_epoch(self):
        pass

    def train_actor_critic(self, obs_dict):
        pass 

    def get_intrinsic_reward(self, obs):
        return self.rnd_curiosity.get_loss(obs)

    def train_intrinsic_reward(self, obs_dict):
        obs = obs_dict['obs']
        self.rnd_curiosity.train_net(obs)

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self, obs_dict):
        obs_dict['e_clip'] = self.e_clip
        return self.central_value_net.train_net(obs_dict)

    def get_full_state_weights(self):
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()      

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        if self.has_curiosity:
            state['rnd_nets'] = self.rnd_curiosity.state_dict()
        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        if self.has_curiosity:
            self.rnd_curiosity.load_state_dict(weights['rnd_nets'])
        self.optimizer.load_state_dict(weights['optimizer'])

    def get_weights(self):
        state = {'model': self.model.state_dict()}

        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_reward:
            state['reward_mean_std'] = self.reward_mean_std.state_dict()   
        return state

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_reward:
            self.reward_mean_std.load_state_dict(weights['reward_mean_std'])


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

    def play_steps(self):
        mb_rnn_states = []
        epinfos = []

        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards.fill_(0)
        mb_values = self.mb_values.fill_(0)
        mb_dones = self.mb_dones.fill_(1)
        mb_actions = self.mb_actions
        mb_neglogpacs = self.mb_neglogpacs
  
        if self.has_curiosity:
            mb_intrinsic_rewards = self.mb_intrinsic_rewards

        if self.has_central_value:
            mb_vobs = self.mb_vobs

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None
        if self.is_rnn:
            mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states)

        for n in range(self.steps_num):
            if self.is_rnn:
                seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)
                if full_tensor:
                    break

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                actions, values, neglogpacs, _, self.rnn_states = self.get_masked_action_values(self.obs, masks)
            else:
                actions, values, neglogpacs, self.rnn_states = self.get_action_values(self.obs)
                
            values = torch.squeeze(values)
            neglogpacs = torch.squeeze(neglogpacs)

            if self.is_rnn:
                mb_dones[indices.cpu(), play_mask.cpu()] = self.dones.byte()
                mb_obs[indices,play_mask] = self.obs['obs']    
            else:
                mb_obs[n,:] = self.obs['obs']
                mb_dones[n,:] = self.dones

            self.obs, rewards, self.dones, infos = self.env_step(actions)

            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs['obs'])
                mb_intrinsic_rewards[n,:] = intrinsic_reward

            if self.has_central_value:
                mb_vobs[n,:] = self.obs['states']
            
            shaped_rewards = self.rewards_shaper(rewards)
            
            if self.normalize_reward:
                shaped_rewards = self.reward_mean_std(shaped_rewards)
            if self.is_rnn:
                mb_actions[indices,play_mask] = actions
                mb_neglogpacs[indices,play_mask] = neglogpacs
                mb_values[indices.cpu(), play_mask.cpu()] = values
                mb_rewards[indices.cpu(), play_mask.cpu()] = shaped_rewards
            else: 
                mb_actions[n,:] = actions
                mb_neglogpacs[n,:] = neglogpacs
                mb_values[n,:] = values
                mb_rewards[n,:] = shaped_rewards
                
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            if self.is_rnn:
                self.process_rnn_dones(all_done_indices, indices, seq_indices)  
                     
            self.game_rewards.extend(self.current_rewards[done_indices])
            self.game_lengths.extend(self.current_lengths[done_indices])

            for ind in done_indices:
                info = infos[ind//self.num_agents]
                game_res = 0
                if info is not None:
                    game_res = infos[ind//self.num_agents].get('battle_won', 0.0)
                self.game_scores.append(game_res)

            epinfos.append(infos)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones
        
        last_values = self.get_values(self.obs, actions)
        last_values = torch.squeeze(last_values)

        if self.has_curiosity:
            mb_intrinsic_values = mb_values[:,:,1]
            mb_extrinsic_values = mb_values[:,:,0]
            last_intrinsic_values = last_values[:, 1]
            last_extrinsic_values = last_values[:, 0]
        else:
            mb_extrinsic_values = mb_values
            last_extrinsic_values = last_values

        fdones = self.dones.float()
        mb_fdones = mb_dones.float()

        '''
        TODO: rework this usecase better

        '''
        if self.is_rnn:
            non_finished = (indices != self.steps_num).nonzero()
            ind_to_fill = indices[non_finished]
            mb_fdones[ind_to_fill,non_finished] = fdones[non_finished]
            mb_extrinsic_values[ind_to_fill,non_finished] = last_extrinsic_values[non_finished]
            fdones[non_finished] = 1.0
            last_extrinsic_values[non_finished] = 0.0
        mb_advs = self.discount_values(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards)

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
        if self.has_central_value:
            batch_dict['states'] = mb_vobs

        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}

        if self.is_rnn:
            batch_dict['rnn_states'] = mb_rnn_states
            batch_dict['rnn_masks'] = mb_rnn_masks

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
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values

        if self.has_curiosity:
            self.train_intrinsic_reward(batch_dict)
            advantages[:,1] = advantages[:,1] * self.rnd_adv_coef
            advantages = torch.sum(advantages, axis=1)

        if self.has_central_value:
            self.train_central_value(batch_dict)

        if self.normalize_advantage:
            if self.is_rnn:
                sum_mask = rnn_masks.sum()
                advantages_mask = advantages * rnn_masks
                advantages_mean = advantages_mask.sum() / sum_mask
                advantages_std = torch.sqrt(rnn_masks*(((advantages_mask - advantages_mean)**2)/(sum_mask-1)).sum())
                advantages = (advantages_mask - advantages_mean) / (advantages_std + 1e-8)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        a_losses = []
        c_losses = []
        entropies = []
        kls = []

        if self.is_rnn:
            print(rnn_masks.sum().item() / (rnn_masks.nelement()))
            total_games = self.batch_size // self.seq_len
            num_games_batch = self.minibatch_size // self.seq_len
            game_indexes = torch.arange(total_games, dtype=torch.long, device='cuda:0')
            flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device='cuda:0').reshape(total_games, self.seq_len)
            for _ in range(0, self.mini_epochs_num):
                #permutation = torch.randperm(total_games, dtype=torch.long, device='cuda:0')
                #game_indexes = game_indexes[permutation]
                for i in range(0, self.num_minibatches):
                    batch = torch.range(i * num_games_batch, (i + 1) * num_games_batch - 1, dtype=torch.long, device='cuda:0')
                    mb_indexes = game_indexes[batch]
 
                    mbatch = flat_indexes[mb_indexes].flatten()           
                    input_dict = {}
                    input_dict['old_values'] = values[mbatch]
                    input_dict['old_logp_actions'] = neglogpacs[mbatch]
                    input_dict['advantages'] = advantages[mbatch]
                    input_dict['returns'] = returns[mbatch]
                    input_dict['actions'] = actions[mbatch]
                    input_dict['obs'] = obses[mbatch]
                    input_dict['rnn_states'] = [s[:,mb_indexes,:] for s in rnn_states]
                    input_dict['rnn_masks'] = rnn_masks[mbatch]
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
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()

        while True:
            epoch_num = self.update_epoch()
            self.frame += self.batch_size_envs
            frame = self.frame

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

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'])
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                            return self.last_mean_rewards, epoch_num

                if epoch_num > self.max_epochs:
                    self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num                               
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
        mb_rnn_states = []
        epinfos = []

        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards.fill_(0)
        mb_values = self.mb_values.fill_(0)
        mb_dones = self.mb_dones.fill_(1)
        mb_actions = self.mb_actions
        mb_neglogpacs = self.mb_neglogpacs
        mb_mus = self.mb_mus.fill_(0)
        mb_sigmas = self.mb_sigmas.fill_(0) 
        if self.has_curiosity:
            mb_intrinsic_rewards = self.mb_intrinsic_rewards

        if self.has_central_value:
            mb_vobs = self.mb_vobs

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None
        if self.is_rnn:
            mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states)

        for n in range(self.steps_num):
            if self.is_rnn:
                seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)
                if full_tensor:
                    break

            actions, values, neglogpacs, mu, sigma, self.rnn_states = self.get_action_values(self.obs)
                
            values = torch.squeeze(values)
            neglogpacs = torch.squeeze(neglogpacs)

            if self.is_rnn:
                mb_dones[indices.cpu(), play_mask.cpu()] = self.dones.byte()
                mb_obs[indices,play_mask] = self.obs['obs']    
            else:
                mb_obs[n,:] = self.obs['obs']
                mb_dones[n,:] = self.dones


            clamped_actions = torch.clamp(actions, -1.0, 1.0)	            
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
            self.obs, rewards, self.dones, infos = self.env_step(rescaled_actions)

            if self.has_curiosity:
                intrinsic_reward = self.get_intrinsic_reward(self.obs['obs'])
                mb_intrinsic_rewards[n,:] = intrinsic_reward

            if self.has_central_value:
                mb_vobs[n,:] = self.obs['states']

            shaped_rewards = self.rewards_shaper(rewards)
            if self.normalize_reward:
                shaped_rewards = self.reward_mean_std(shaped_rewards)

            if self.is_rnn:
                mb_actions[indices,play_mask] = actions
                mb_neglogpacs[indices,play_mask] = neglogpacs
                mb_mus[indices,play_mask] = mu
                mb_sigmas[indices,play_mask] = sigma
                mb_values[indices.cpu(), play_mask.cpu()] = values
                mb_rewards[indices.cpu(), play_mask.cpu()] = shaped_rewards
            else: 
                mb_actions[n,:] = actions
                mb_neglogpacs[n,:] = neglogpacs
                mb_mus[n,:] = mu
                mb_sigmas[n,:] = sigma
                mb_values[n,:] = values
                mb_rewards[n,:] = shaped_rewards
                
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            if self.is_rnn:
                self.process_rnn_dones(all_done_indices, indices, seq_indices)  
                     
            self.game_rewards.extend(self.current_rewards[done_indices])
            self.game_lengths.extend(self.current_lengths[done_indices])
            if infos is not None:
                for ind in done_indices:
                    game_res = infos[ind//self.num_agents].get('battle_won', 0.0)
                    self.game_scores.append(game_res)

            epinfos.append(infos)

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

        fdones = self.dones.float()
        mb_fdones = mb_dones.float()

        '''
        TODO: rework this usecase better

        '''
        if self.is_rnn:
            non_finished = (indices != self.steps_num).nonzero()
            ind_to_fill = indices[non_finished]
            mb_fdones[ind_to_fill,non_finished] = fdones[non_finished]
            mb_extrinsic_values[ind_to_fill,non_finished] = last_extrinsic_values[non_finished]
            fdones[non_finished] = 1.0
            last_extrinsic_values[non_finished] = 0.0
            
        mb_advs = self.discount_values(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards)

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

        if self.has_central_value:
            batch_dict['states'] = mb_vobs

        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}

        if self.is_rnn:
            batch_dict['rnn_states'] = mb_rnn_states
            batch_dict['rnn_masks'] = mb_rnn_masks

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
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        play_time_end = time.time()
        update_time_start = time.time()
        advantages = returns - values

        if self.normalize_advantage:
            if self.is_rnn:
                sum_mask = rnn_masks.sum()
                advantages_mask = advantages * rnn_masks
                advantages_mean = advantages_mask.sum() / sum_mask
                advantages_std = torch.sqrt(rnn_masks*(((advantages_mask - advantages_mean)**2)/(sum_mask-1)).sum())
                advantages = (advantages_mask - advantages_mean) / (advantages_std + 1e-8)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.has_central_value:
            self.train_central_value(batch_dict)

        if self.has_curiosity:
            self.train_intrinsic_reward(batch_dict)
            advantages[:, 1] = advantages[:, 1] * self.rnd_adv_coef
            advantages = torch.sum(advantages, axis=1)  

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)
            self.curr_frames = int(self.batch_size_envs * frames_mask_ratio)
            total_games = self.batch_size // self.seq_len
            num_games_batch = self.minibatch_size // self.seq_len
            game_indexes = torch.arange(total_games, dtype=torch.long, device='cuda:0')
            flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device='cuda:0').reshape(total_games, self.seq_len)
            for _ in range(0, self.mini_epochs_num):
                permutation = torch.randperm(total_games, dtype=torch.long, device='cuda:0')
                game_indexes = game_indexes[permutation]
                for i in range(0, self.num_minibatches):
                    batch = torch.range(i * num_games_batch, (i + 1) * num_games_batch - 1, dtype=torch.long, device='cuda:0')
                    mb_indexes = game_indexes[batch]
                    mbatch = flat_indexes[mb_indexes].flatten()        
            
                    input_dict = {}
                    input_dict['old_values'] = values[mbatch]
                    input_dict['old_logp_actions'] = neglogpacs[mbatch]
                    input_dict['advantages'] = advantages[mbatch]
                    input_dict['returns'] = returns[mbatch]
                    input_dict['actions'] = actions[mbatch]
                    input_dict['obs'] = obses[mbatch]
                    input_dict['rnn_states'] = [s[:,mb_indexes,:] for s in rnn_states]
                    input_dict['rnn_masks'] = rnn_masks[mbatch]
                    input_dict['learning_rate'] = self.last_lr
                    input_dict['mu'] = mus[mbatch]
                    input_dict['sigma'] = sigmas[mbatch]
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(input_dict)
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    kls.append(kl)
                    entropies.append(entropy)        
                    mus[mbatch] = cmu
                    sigmas[mbatch] = csigma
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)                            

        else:
            '''permutation = torch.randperm(self.batch_size, dtype=torch.long, device='cuda:0')
            obses = obses[permutation]
            returns = returns[permutation]      
            actions = actions[permutation]
            values = values[permutation]
            neglogpacs = neglogpacs[permutation]
            advantages = advantages[permutation]
            mus = mus[permutation]
            sigmas = sigmas[permutation]'''

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
    
    def train(self):
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        while True:
            epoch_num = self.update_epoch()
            
            play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            self.frame += self.curr_frames
            frame = self.frame
            if True:
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                if self.print_stats:
                    fps_step = self.curr_frames / scaled_play_time
                    fps_total = self.curr_frames / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')
                self.writer.add_scalar('performance/total_fps', self.curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', self.curr_frames / scaled_play_time, frame)
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
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        self.save("./nn/" + self.config['name'])
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save("./nn/" + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                            return self.last_mean_rewards, epoch_num

                if epoch_num > self.max_epochs:
                    self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num                               
                update_time = 0