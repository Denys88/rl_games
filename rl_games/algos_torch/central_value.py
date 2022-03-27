import torch
from torch import nn
import gym
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.common  import common_losses
from rl_games.common import datasets
from rl_games.common import schedulers

class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, value_size, ppo_device, num_agents, horizon_length, num_actors, num_actions, seq_len, normalize_value,network, config, writter, max_epochs, multi_gpu):
        nn.Module.__init__(self)
        self.ppo_device = ppo_device
        self.num_agents, self.horizon_length, self.num_actors, self.seq_len = num_agents, horizon_length, num_actors, seq_len
        self.normalize_value = normalize_value
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.value_size = value_size
        self.max_epochs = max_epochs
        self.multi_gpu = multi_gpu
        self.truncate_grads = config.get('truncate_grads', False)
        self.config = config
        self.normalize_input = config['normalize_input']
        state_config = {
            'value_size' : value_size,
            'input_shape' : state_shape,
            'actions_num' : num_actions,
            'num_agents' : num_agents,
            'num_seqs' : num_actors,
            'normalize_input' : self.normalize_input,
            'normalize_value': self.normalize_value,
        }

        self.model = network.build(state_config)
        self.lr = float(config['learning_rate'])
        self.linear_lr = config.get('lr_schedule') == 'linear'
        if self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(self.lr, 
                max_steps=self.max_epochs, 
                apply_to_entropy=False,
                start_entropy_coef=0)
        else:
            self.scheduler = schedulers.IdentityScheduler()
        
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.num_minibatches = self.horizon_length * self.num_actors // self.mini_batch
        self.clip_value = config['clip_value']

        self.writter = writter
        self.weight_decay = config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr), eps=1e-08, weight_decay=self.weight_decay)
        self.frame = 0
        self.epoch_num = 0
        self.running_mean_std = None
        self.grad_norm = config.get('grad_norm', 1)
        self.truncate_grads = config.get('truncate_grads', False)
        self.e_clip = config.get('e_clip', 0.2)
        self.truncate_grad = self.config.get('truncate_grads', False)

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None
        self.batch_size = self.horizon_length * self.num_actors
        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]
            total_agents = self.num_actors #* self.num_agents
            num_seqs = self.horizon_length // self.seq_len
            assert ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [ torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32, device=self.ppo_device) for s in self.rnn_states]

        self.dataset = datasets.PPODataset(self.batch_size, self.mini_batch, True, self.is_rnn, self.ppo_device, self.seq_len)

    def update_lr(self, lr):

        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'cv_learning_rate')
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_stats_weights(self):
        return {}

    def set_stats_weights(self, weights): 
        pass
        
    def update_dataset(self, batch_dict):
        value_preds = batch_dict['old_values']     
        returns = batch_dict['returns']   
        actions = batch_dict['actions']
        dones = batch_dict['dones']
        rnn_masks = batch_dict['rnn_masks']
        if self.num_agents > 1:
            res = self.update_multiagent_tensors(value_preds, returns, actions, dones)
            batch_dict['old_values'] = res[0]
            batch_dict['returns'] = res[1]
            batch_dict['actions'] = res[2]
            batch_dict['dones'] = res[3]
        
        if self.is_rnn:
            states = []
            for mb_s in self.mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1,2,0,3).reshape(-1, t_size, h_size))

            batch_dict['rnn_states'] = states
            if self.num_agents > 1:
                rnn_masks = res[3]
            batch_dict['rnn_masks'] = rnn_masks
        self.dataset.update_values_dict(batch_dict)

    def _preproc_obs(self, obs_batch):
        if isinstance(obs_batch, dict):
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def pre_step_rnn(self, n):
        if not self.is_rnn:
            return
        if n % self.seq_len == 0:
            for s, mb_s in zip(self.rnn_states, self.mb_rnn_states):
                mb_s[n // self.seq_len,:,:,:] = s

    def post_step_rnn(self, all_done_indices):
        if not self.is_rnn:
            return
        all_done_indices = all_done_indices[::self.num_agents] // self.num_agents
        for s in self.rnn_states:
            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

    def forward(self, input_dict):
        return self.model(input_dict)


    def get_value(self, input_dict):
        self.eval()
        obs_batch = input_dict['states']
        actions = input_dict.get('actions', None)

        obs_batch = self._preproc_obs(obs_batch)
        res_dict = self.forward({'obs' : obs_batch, 'actions': actions,
                                    'rnn_states': self.rnn_states,
                                    'is_train' : False})
        value, self.rnn_states = res_dict['values'], res_dict['rnn_states']
        if self.num_agents > 1:
            value = value.repeat(1, self.num_agents)
            value = value.view(value.size()[0]*self.num_agents, -1)

        return value

    def train_critic(self, input_dict):
        self.train()
        loss = self.calc_gradients(input_dict)
        return loss.item()

    def update_multiagent_tensors(self, value_preds, returns, actions, dones):
        batch_size = self.batch_size
        ma_batch_size = self.num_actors * self.num_agents * self.horizon_length
        value_preds = value_preds.view(self.num_actors, self.num_agents, self.horizon_length, self.value_size).transpose(0,1)
        returns = returns.view(self.num_actors, self.num_agents, self.horizon_length, self.value_size).transpose(0,1)
        value_preds = value_preds.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        returns = returns.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        dones = dones.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        #if self.is_rnn:
        #    rnn_masks = rnn_masks.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
        #    rnn_masks = rnn_masks.flatten(0)[:batch_size]

        return value_preds, returns, actions, dones

    def train_net(self):
        self.train()
        loss = 0
        for _ in range(self.mini_epoch):
            for idx in range(len(self.dataset)):
                loss += self.train_critic(self.dataset[idx])
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch
        avg_loss = loss / (self.mini_epoch * self.num_minibatches)

        self.epoch_num += 1
        self.lr, _ = self.scheduler.update(self.lr, 0, self.epoch_num, 0, 0)
        self.update_lr(self.lr)
        self.frame += self.batch_size
        if self.writter != None:
            self.writter.add_scalar('losses/cval_loss', avg_loss, self.frame)
            self.writter.add_scalar('info/cval_lr', self.lr, self.frame)        
        return avg_loss

    def calc_gradients(self, batch):
        obs_batch = self._preproc_obs(batch['obs'])
        value_preds_batch = batch['old_values']
        returns_batch = batch['returns']
        actions_batch = batch['actions']
        dones_batch = batch['dones']
        rnn_masks_batch = batch.get('rnn_masks')

        batch_dict = {'obs' : obs_batch, 
                    'actions' : actions_batch,
                    'seq_length' : self.seq_len,
                    'dones' : dones_batch}
        if self.is_rnn:
            batch_dict['rnn_states'] = batch['rnn_states']

        res_dict = self.model(batch_dict)
        values = res_dict['values']
        loss = common_losses.critic_loss(value_preds_batch, values, self.e_clip, returns_batch, self.clip_value)
        losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
        loss = losses[0]
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        loss.backward()

        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                #self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
            else:
                #self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()    
        else:
            self.optimizer.step()
        
        return loss
