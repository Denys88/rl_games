from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd

from rl_games.common import vecenv, schedulers, experience

from rl_games.common.a2c_common import  ContinuousA2CBase
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rl_games.algos_torch import  model_builder

import torch 
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time

import gym


class SHACAgent(ContinuousA2CBase):
    def __init__(self, base_name, params):
        ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(build_config)
        self.critic = self.critic_network.build(build_config)
        if self.normalize_input:
            self.critic.input_mean_std = self.model.input_mean_std

        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                                    weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'normalize_value': self.normalize_value,
                'network': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
                'hvd': self.hvd if self.multi_gpu else None
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                           self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.algo_observer.after_init(self)

    def load_networks(self, params):
        ContinuousA2CBase.load_networks(self, params)
        if critic_config in self.config:
            builder = model_builder.ModelBuilder()
            print('Adding Critic Network')
            network = builder.load(params['config']['critic_config'])
            self.critic_network = network

    def get_actions(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
        res_dict = self.model(input_dict)
        return res_dict

    def get_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
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
            processed_obs = self._preproc_obs(obs['obs'])
            result = self.critic_model(input_dict)
            value = result['values']
            return value

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)


        self.optimizer.zero_grad(set_to_none=True)


        self.scaler.scale(loss).backward()
        self.trancate_gradients()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask


    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result
