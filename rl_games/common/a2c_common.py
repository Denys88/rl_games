from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.moving_mean_std import MovingMeanStd
from rl_games.algos_torch.self_play_manager import  SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
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

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.ppo_device = config.get('device', 'cuda:0')
        print('Env info:')
        print(self.env_info)
        self.value_size = self.env_info.get('value_size', 1)
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
        self.max_epochs = self.config.get('max_epochs', 1e6)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')
        if self.is_adaptive_lr:
            self.lr_threshold = config['lr_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.lr_threshold)
        elif self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                max_steps=self.max_epochs, 
                apply_to_entropy=config.get('schedule_entropy', False),
                start_entropy_coef=config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.steps_num = config['steps_num']
        self.seq_len = self.config.get('seq_length', 4)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
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
        
        self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)
        
        # features
        self.algo_observer = config['features'].get('observer', 'EmptyObserver')

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            value = self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            value = self.value_mean_std.train()

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    #'actions' : res_dict['action'],
                    #'rnn_states' : self.rnn_states
                }
                value = self.get_central_value(input_dict)
                res_dict['value'] = value
        if self.normalize_value:
            res_dict['value'] = self.value_mean_std(res_dict['value'], True)
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
                }
                
                result = self.model(input_dict)
                value = result['value']

            if self.normalize_value:
                value = self.value_mean_std(value, True)
            return value

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors
 
        val_shape = (self.steps_num, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.mb_obs = torch.zeros((self.steps_num, batch_size) + self.obs_shape, dtype=torch_dtype, device=self.ppo_device)

        if self.has_central_value:
            self.mb_vobs = torch.zeros((self.steps_num, self.num_actors) + self.state_shape, dtype=torch_dtype, device=self.ppo_device)

        self.mb_rewards = torch.zeros(val_shape, dtype = torch.float32, device=self.ppo_device)
        self.mb_values = torch.zeros(val_shape, dtype = torch.float32, device=self.ppo_device)
        self.mb_dones = torch.zeros((self.steps_num, batch_size), dtype = torch.uint8, device=self.ppo_device)
        self.mb_neglogpacs = torch.zeros((self.steps_num, batch_size), dtype = torch.float32, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            batch_size = self.num_agents * self.num_actors
            num_seqs = self.steps_num * batch_size // self.seq_len
            assert((self.steps_num * batch_size // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def init_rnn_step(self, batch_size, mb_rnn_states):
        mb_rnn_states = self.mb_rnn_states
        mb_rnn_masks = torch.zeros(self.steps_num*batch_size, dtype = torch.float32, device=self.ppo_device)
        steps_mask = torch.arange(0, batch_size * self.steps_num, self.steps_num, dtype=torch.long, device=self.ppo_device)
        play_mask = torch.arange(0, batch_size, 1, dtype=torch.long, device=self.ppo_device)
        steps_state = torch.arange(0, batch_size * self.steps_num//self.seq_len, self.steps_num//self.seq_len, dtype=torch.long, device=self.ppo_device)
        indices = torch.zeros((batch_size), dtype = torch.long, device=self.ppo_device)
        return mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states

    def process_rnn_indices(self, mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states):
        seq_indices = None
        if indices.max().item() >= self.steps_num:
            return seq_indices, True

        mb_rnn_masks[indices + steps_mask] = 1
        seq_indices = indices % self.seq_len
        state_indices = (seq_indices == 0).nonzero(as_tuple=False)
        state_pos = indices // self.seq_len
        rnn_indices = state_pos[state_indices] + steps_state[state_indices]

        for s, mb_s in zip(self.rnn_states, mb_rnn_states):
            mb_s[:, rnn_indices, :] = s[:, state_indices, :]

        self.last_rnn_indices = rnn_indices
        self.last_state_indices = state_indices
        return seq_indices, False

    def process_rnn_dones(self, all_done_indices, indices, seq_indices):
        if len(all_done_indices) > 0:
            shifts = self.seq_len - 1 - seq_indices[all_done_indices]
            indices[all_done_indices] += shifts
            for s in self.rnn_states:
                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0
        indices += 1  


    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self.cast_obs(value)
        else:
            upd_obs = {'obs' : self.cast_obs(obs)}
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

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
            nextnonterminal = nextnonterminal.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * mb_masks[t].unsqueeze(1)
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):       
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        pass

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass 

    def calc_gradients(self, opt_step):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()      

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.epoch_num = weights['epoch']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        self.optimizer.load_state_dict(weights['optimizer'])

    def get_weights(self):
        state = {'model': self.model.state_dict()}

        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_value:
            state['reward_mean_std'] = self.value_mean_std.state_dict()   
        return state

    def get_stats_weights(self):
        state = {}
        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_value:
            state['reward_mean_std'] = self.value_mean_std.state_dict()
        if self.has_central_value:
            state['assymetric_vf_mean_std'] = self.central_value_net.get_stats_weights()
        return state

    def set_stats_weights(self, weights):
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.has_central_value:
            self.central_value_net.set_stats_weights(state['assymetric_vf_mean_std'])
  
    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])

    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        #if len(obs_batch.size()) == 3:
        #    obs_batch = obs_batch.permute((0, 2, 1))
        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def play_steps(self):
        mb_rnn_states = []
        epinfos = []

        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_values = self.mb_values
        mb_dones = self.mb_dones
        
        tensors_dict = self.tensors_dict
        update_list = self.update_list
        update_dict = self.update_dict

        if self.has_central_value:
            mb_vobs = self.mb_vobs

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        for n in range(self.steps_num):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            mb_obs[n,:] = self.obs['obs']
            mb_dones[n,:] = self.dones
            for k in update_list:
                tensors_dict[k][n,:] = res_dict[k]

            if self.has_central_value:
                mb_vobs[n,:] = self.obs['states']

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['action'])

            shaped_rewards = self.rewards_shaper(rewards)
            mb_rewards[n,:] = shaped_rewards

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        if self.has_central_value and self.central_value_net.use_joint_obs_actions:
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                val_dict = self.get_masked_action_values(self.obs, masks)
            else:
                val_dict = self.get_action_values(self.obs)
            last_values = val_dict['value']
        else:
            last_values = self.get_values(self.obs)

        mb_extrinsic_values = mb_values
        last_extrinsic_values = last_values

        fdones = self.dones.float()
        mb_fdones = mb_dones.float()
        mb_advs = self.discount_values(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards)
        mb_returns = mb_advs + mb_extrinsic_values
        batch_dict = {
            'obs' : mb_obs,
            'returns' : mb_returns,
            'dones' : mb_dones,
        }
        for k in update_list:
            batch_dict[update_dict[k]] = tensors_dict[k]

        if self.has_central_value:
            batch_dict['states'] = mb_vobs

        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}

        return batch_dict

    def play_steps_rnn(self):
        mb_rnn_states = []
        epinfos = []

        mb_obs = self.mb_obs
        mb_values = self.mb_values.fill_(0)
        mb_rewards = self.mb_rewards.fill_(0)
        mb_dones = self.mb_dones.fill_(1)

        tensors_dict = self.tensors_dict
        update_list = self.update_list
        update_dict = self.update_dict

        if self.has_central_value:
            mb_vobs = self.mb_vobs

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states)

        for n in range(self.steps_num):
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)
            if full_tensor:
                break
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                
            self.rnn_states = res_dict['rnn_state']

            mb_dones[indices, play_mask] = self.dones.byte()
            mb_obs[indices,play_mask] = self.obs['obs']   

            for k in update_list:
                tensors_dict[k][indices,play_mask] = res_dict[k]

            if self.has_central_value:
                mb_vobs[indices[::self.num_agents] ,play_mask[::self.num_agents]//self.num_agents] = self.obs['states']

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['action'])

            shaped_rewards = self.rewards_shaper(rewards)

            mb_rewards[indices, play_mask] = shaped_rewards

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)  
            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)
        
            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        if self.has_central_value and self.central_value_net.use_joint_obs_actions:
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                val_dict = self.get_masked_action_values(self.obs, masks)
            else:
                val_dict = self.get_action_values(self.obs)
            
            last_values = val_dict['value']
        else:
            last_values = self.get_values(self.obs)

        mb_extrinsic_values = mb_values
        last_extrinsic_values = last_values
        fdones = self.dones.float()
        mb_fdones = mb_dones.float()

        non_finished = (indices != self.steps_num).nonzero(as_tuple=False)
        ind_to_fill = indices[non_finished]
        mb_fdones[ind_to_fill,non_finished] = fdones[non_finished]
        mb_extrinsic_values[ind_to_fill,non_finished] = last_extrinsic_values[non_finished]
        fdones[non_finished] = 1.0
        last_extrinsic_values[non_finished] = 0

        mb_advs = self.discount_values_masks(fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_rnn_masks.view(-1,self.steps_num).transpose(0,1))

        mb_returns = mb_advs + mb_extrinsic_values

        batch_dict = {
            'obs' : mb_obs,
            'returns' : mb_returns,
            'dones' : mb_dones,
        }
        for k in update_list:
            batch_dict[update_dict[k]] = tensors_dict[k]

        if self.has_central_value:
            batch_dict['states'] = mb_vobs

        batch_dict = {k: swap_and_flatten01(v) for k, v in batch_dict.items()}

        batch_dict['rnn_states'] = mb_rnn_states
        batch_dict['rnn_masks'] = mb_rnn_masks

        return batch_dict

class DiscreteA2CBase(A2CBase):
    def __init__(self, base_name, config):
        A2CBase.__init__(self, base_name, config)
        batch_size = self.num_agents * self.num_actors
        action_space = self.env_info['action_space'] 
        if type(action_space) is gym.spaces.Discrete:
            self.actions_shape = (self.steps_num, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is gym.spaces.Tuple:
            self.actions_shape = (self.steps_num, batch_size, len(action_space)) 
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        self.is_discrete = True

    def init_tensors(self):
        A2CBase.init_tensors(self)
        batch_size = self.num_agents * self.num_actors
        self.mb_actions = torch.zeros(self.actions_shape, dtype = torch.long, device=self.ppo_device)

        self.update_list = ['action', 'neglogp', 'value']
        self.update_dict = {
            'action' : 'actions',
            'neglogp' : 'neglogpacs',
            'value' : 'values',
        }
        self.tensors_dict = {
            'action' : self.mb_actions,
            'neglogp' : self.mb_neglogpacs,
            'value' : self.mb_values,
        }

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()

        if self.is_rnn:
            print('non masked rnn obs ratio: ',rnn_masks.sum().item() / (rnn_masks.nelement()))

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)    
                
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, np.mean(ep_kls))
            self.update_lr(self.last_lr)
            kls.append(np.mean(ep_kls))
            
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)
        obses = batch_dict['obs']
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        rnn_states = batch_dict.get('rnn_states', None)
        advantages = returns - values
        
        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)       

        advantages = torch.sum(advantages, axis=1)
        
        if self.normalize_advantage:
            if self.is_rnn:
                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        
        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states'] 
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
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
                self.writer.add_scalar('performance/update_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/a_loss', np.mean(a_losses), frame)
                self.writer.add_scalar('losses/c_loss', np.mean(c_losses), frame)
                self.writer.add_scalar('losses/entropy', np.mean(entropies), frame)
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', np.mean(kls), frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    if self.value_size > 1:
                        for i in range(self.value_size):
                            self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                            self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                            self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)
                    else:
                        self.writer.add_scalar('rewards/frame', mean_rewards[0], frame)
                        self.writer.add_scalar('rewards/iter', mean_rewards[0], epoch_num)
                        self.writer.add_scalar('rewards/time', mean_rewards[0], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
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
        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)	            
        rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()
        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        batch_size = self.num_agents * self.num_actors
        self.mb_actions = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32, device=self.ppo_device)
        self.mb_mus = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32, device=self.ppo_device)
        self.mb_sigmas = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32, device=self.ppo_device)

        self.update_list = ['action', 'neglogp', 'value', 'mu', 'sigma']
        self.update_dict = {
            'action' : 'actions',
            'neglogp' : 'neglogpacs',
            'value' : 'values',
            'mu' : 'mus',
            'sigma' : 'sigmas'
        }
        self.tensors_dict = {
            'action' : self.mb_actions,
            'neglogp' : self.mb_neglogpacs,
            'value' : self.mb_values,
            'mu' : self.mb_mus,
            'sigma' : self.mb_sigmas,
        }

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 
        play_time_end = time.time()
        update_time_start = time.time()

        rnn_masks = batch_dict.get('rnn_masks', None)

        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        
        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)
            self.curr_frames = int(self.batch_size_envs * frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)   
                if self.schedule_type == 'legacy':  
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl)
                    self.update_lr(self.last_lr)

            if self.schedule_type == 'standard': 
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,kl)
                self.update_lr(self.last_lr)
            kls.append(np.mean(ep_kls))

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obs']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
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
                self.writer.add_scalar('performance/update_time', update_time, frame)
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
                self.writer.add_scalar('info/epochs', epoch_num, frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    if self.value_size > 1:
                        for i in range(self.value_size):
                            self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                            self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                            self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)
                    else:
                        self.writer.add_scalar('rewards/frame', mean_rewards[0], frame)
                        self.writer.add_scalar('rewards/iter', mean_rewards[0], epoch_num)
                        self.writer.add_scalar('rewards/time', mean_rewards[0], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save("./nn/" + 'last_' + self.config['name'] + 'ep=' + str(epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
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