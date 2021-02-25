import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common  import common_losses
from rl_games.common import datasets


class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, value_size, ppo_device, num_agents, num_steps, num_actors, num_actions, seq_len, model, config, writter):
        nn.Module.__init__(self)
        self.ppo_device = ppo_device
        self.num_agents, self.num_steps, self.num_actors, self.seq_len = num_agents, num_steps, num_actors, seq_len
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.value_size = value_size
        state_config = {
            'value_size' : value_size,
            'input_shape' : state_shape,
            'actions_num' : num_actions,
            'num_agents' : num_agents,
            'num_seqs' : num_actors
        }
        self.config = config
        self.model = model.build('cvalue', **state_config)
        self.lr = config['lr']
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.num_minibatches = self.num_steps * self.num_actors // self.mini_batch
        self.clip_value = config['clip_value']
        self.normalize_input = config['normalize_input']
        self.normalize_value = config.get('normalize_value', False)
        self.running_mean_std = None
        if self.normalize_input:
             self.running_mean_std = RunningMeanStd(state_shape)

        self.writter = writter
        self.use_joint_obs_actions = config.get('use_joint_obs_actions', False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr), eps=1e-07)
        self.frame = 0

        self.grad_norm = config.get('grad_norm', 1)
        self.truncate_grads = config.get('truncate_grads', False)
        self.e_clip = config.get('e_clip', 0.2)

        # todo - from the ьфшт  config!
        self.mixed_precision = self.config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None
        self.batch_size = self.num_steps * self.num_actors

        if self.is_rnn:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                self.rnn_states = self.model.get_default_rnn_state()
                self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]
                num_seqs = self.num_steps * self.num_actors // self.seq_len
                assert((self.num_steps * self.num_actors // self.num_minibatches) % self.seq_len == 0)
                self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

        self.dataset = datasets.PPODataset(self.batch_size, self.mini_batch, True, self.is_rnn, self.ppo_device, self.seq_len)

    def get_stats_weights(self):
        if self.normalize_input:
            return self.running_mean_std.state_dict()
        else:
            return None

    def set_stats_weights(self, weights): 
        self.running_mean_std.load_state_dict(weights)

    def update_dataset(self, batch_dict):
        value_preds = batch_dict['old_values']     
        returns = batch_dict['returns']   
        actions = batch_dict['actions']
        rnn_masks = batch_dict['rnn_masks']

        if self.num_agents > 1:
            res = self.update_multiagent_tensors(value_preds, returns, actions, rnn_masks)
            batch_dict['old_values'] = res[0]
            batch_dict['returns']  = res[1]
            batch_dict['actions']  = res[2]

        if self.is_rnn:
            batch_dict['rnn_states'] = self.mb_rnn_states
            if self.num_agents > 1:
                rnn_masks = res[3]
            batch_dict['rnn_masks'] = rnn_masks
        self.dataset.update_values_dict(batch_dict)

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

    def pre_step_rnn(self, rnn_indices, state_indices):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            if self.num_agents > 1:
                rnn_indices = rnn_indices[::self.num_agents] 
                shifts = rnn_indices % (self.num_steps // self.seq_len)
                rnn_indices = (rnn_indices - shifts) // self.num_agents + shifts
                state_indices = state_indices[::self.num_agents] // self.num_agents

            for s, mb_s in zip(self.rnn_states, self.mb_rnn_states):
                mb_s[:, rnn_indices, :] = s[:, state_indices, :]

    def post_step_rnn(self, all_done_indices):
        all_done_indices = all_done_indices[::self.num_agents] // self.num_agents
        for s in self.rnn_states:
            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

    def forward(self, input_dict):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            value, rnn_states = self.model(input_dict)

        return value, rnn_states

    def get_value(self, input_dict):
        self.eval()
        obs_batch = input_dict['states']
        
        actions = input_dict.get('actions', None)

        obs_batch = self._preproc_obs(obs_batch)
        value, self.rnn_states = self.forward({'obs': obs_batch, 'actions': actions, 
                                             'rnn_states': self.rnn_states})
        if self.num_agents > 1:
            value = value.repeat(1, self.num_agents)
            value = value.view(value.size()[0]*self.num_agents, -1)

        return value

    def train_critic(self, input_dict, opt_step = True):
        self.train()
        loss = self.calc_gradients(input_dict, opt_step)

        return loss.item()

    def update_multiagent_tensors(self, value_preds, returns, actions, rnn_masks):
        batch_size = self.batch_size
        ma_batch_size = self.num_actors * self.num_agents * self.num_steps
        value_preds = value_preds.view(self.num_actors, self.num_agents, self.num_steps, self.value_size).transpose(0, 1)
        returns = returns.view(self.num_actors, self.num_agents, self.num_steps, self.value_size).transpose(0, 1)
        value_preds = value_preds.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        returns = returns.contiguous().view(ma_batch_size, self.value_size)[:batch_size]

        if self.use_joint_obs_actions:
            assert(len(actions.size() == 2), 'use_joint_obs_actions not yet supported in continuous environment for central value')
            actions = actions.view(self.num_actors, self.num_agents, self.num_steps).transpose(0, 1)
            actions = actions.contiguous().view(batch_size, self.num_agents)

        if self.is_rnn:
            rnn_masks = rnn_masks.view(self.num_actors, self.num_agents, self.num_steps).transpose(0, 1)
            rnn_masks = rnn_masks.flatten(0)[:batch_size]

        return value_preds, returns, actions, rnn_masks

    def train_net(self):
        self.train()
        loss = 0
        for _ in range(self.mini_epoch):
            for idx in range(len(self.dataset)):
                loss += self.train_critic(self.dataset[idx])
        avg_loss = loss / (self.mini_epoch * self.num_minibatches)

        self.writter.add_scalar('losses/cval_loss', avg_loss, self.frame)
        self.frame += self.batch_size

        return avg_loss

    def calc_gradients(self, batch, opt_step):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            obs_batch = self._preproc_obs(batch['obs']) 
            value_preds_batch = batch['old_values']
            returns_batch = batch['returns']
            actions_batch = batch['actions']
            rnn_masks_batch = batch.get('rnn_masks')

            if self.is_rnn:
                batch_dict['rnn_states'] = batch['rnn_states']

            batch_dict = {'obs': obs_batch, 
                'actions': actions_batch,
                'seq_length': self.seq_len }

            values, _ = self.forward(batch_dict)
            loss = common_losses.critic_loss(value_preds_batch, values, self.e_clip, returns_batch, self.clip_value)
            losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
            loss = losses[0]

        for param in self.model.parameters():
            param.grad = None

        self.scaler.scale(loss).backward()
        if self.config['truncate_grads']:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        if opt_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss
