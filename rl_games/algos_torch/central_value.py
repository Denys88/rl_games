import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common  import common_losses



class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, ppo_device, num_agents, num_steps, num_actors, num_actions, seq_len, model, config, writter):
        nn.Module.__init__(self)
        self.ppo_device = ppo_device
        self.num_agents, self.num_steps, self.num_actors, self.seq_len = num_agents, num_steps, num_actors, seq_len
        self.num_actions = num_actions
        self.state_shape = state_shape
        
        state_config = {
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
        self.normalize_reward = config.get('normalize_reward', False)
        self.writter = writter
        self.use_joint_obs_actions = config.get('use_joint_obs_actions', False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr), eps=1e-07)
        self.frame = 0
        self.running_mean_std = None
        self.grad_norm = config.get('grad_norm', 1)
        self.truncate_grads = config.get('truncate_grads', False)
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(state_shape)

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None
        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]
            num_seqs = self.num_steps * self.num_actors // self.seq_len
            assert((self.num_steps * self.num_actors // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

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
        rnn_indices = rnn_indices[::self.num_agents] // self.num_agents
        state_indices = state_indices[::self.num_agents] // self.num_agents
        for s, mb_s in zip(self.rnn_states, self.mb_rnn_states):
            mb_s[:, rnn_indices, :] = s[:, state_indices, :]

    def post_step_rnn(self, all_done_indices):
        all_done_indices = all_done_indices[::self.num_agents] // self.num_agents
        for s in self.rnn_states:
            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

    def forward(self, input_dict):
        value, rnn_states = self.model(input_dict)
        return value, rnn_states

    def get_value(self, input_dict):
        self.eval()
        obs_batch = input_dict['states']
        
        actions = input_dict.get('actions', None)

        obs_batch = self._preproc_obs(obs_batch)
        value, self.rnn_states = self.forward({'obs' : obs_batch, 'actions': actions, 
                                             'rnn_states': self.rnn_states})

        if self.num_agents > 1:
            value = value.repeat(1, self.num_agents)
            value = value.view(value.size()[0]*self.num_agents, -1)
        
        return value

    def train_net(self, input_dict):
        self.train()
        obs = input_dict['states']
        batch_size = obs.size()[0]
        value_preds = input_dict['values'].cuda()
        returns = input_dict['returns'].cuda()
        actions = input_dict['actions']
        

        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks'] 
            rnn_states = self.mb_rnn_states      

        if self.num_agents > 1:
            value_preds, returns, actions, rnn_masks = self.update_multiagent_tensors(value_preds, returns, batch_size, actions, rnn_masks) 
        e_clip = input_dict.get('e_clip', 0.2)
        lr = input_dict.get('lr', self.lr)
        obs = self._preproc_obs(obs)

        self.frame = self.frame + 1
        num_minibatches = batch_size // self.mini_batch
        if self.is_rnn:
            sum_loss = self.train_rnn(batch_size, obs, value_preds, returns, actions, rnn_masks, rnn_states, e_clip)
        else:
            sum_loss = self.train_mlp(batch_size, obs, value_preds, returns, actions, e_clip)

        avg_loss = sum_loss / (num_minibatches * self.mini_epoch)
        self.writter.add_scalar('cval/train_loss', avg_loss, self.frame)
        return avg_loss

    def update_multiagent_tensors(self, value_preds, returns, batch_size, actions, rnn_masks):
        value_preds = value_preds.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
        returns = returns.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
        value_preds = value_preds.flatten(0)[:batch_size]
        returns = returns.flatten(0)[:batch_size]
        if self.use_joint_obs_actions:
            assert(len(actions.size()) == 2, 'use_joint_obs_actions not yet supported in continuous environment for central value')
            actions = actions.view(self.num_actors, self.num_agents, self.num_steps).permute(0,2,1)
            actions = actions.contiguous().view(batch_size, self.num_agents)
        if self.is_rnn:
            rnn_masks = rnn_masks.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
            rnn_masks = rnn_masks.flatten(0)[:batch_size] 
        return value_preds, returns, actions, rnn_masks

    def train_mlp(self, batch_size, obs, value_preds, returns, actions, e_clip):
        mini_batch = self.mini_batch
        num_minibatches = batch_size // mini_batch
        sum_loss = 0
        for _ in range(self.mini_epoch):
            for i in range(num_minibatches):
                start = i * mini_batch
                end = (i + 1) * mini_batch

                obs_batch = obs[start:end]
                value_preds_batch = value_preds[start:end]
                returns_batch = returns[start:end]
                actions_batch = None
                if self.use_joint_obs_actions:
                    actions_batch = actions[start:end].view(mini_batch * self.num_agents)
                values, _ = self.forward({'obs' : obs_batch, 'actions' : actions_batch})
                loss = common_losses.critic_loss(value_preds_batch, values, e_clip, returns_batch, self.clip_value)
                loss = loss.mean()
                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                if self.config['truncate_grads']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                sum_loss += loss.item()
        return sum_loss

    def train_rnn(self, batch_size, obs, value_preds, returns, actions, rnn_masks, rnn_states, e_clip):
        mini_batch = self.mini_batch
        num_minibatches = batch_size // mini_batch
        total_games = batch_size // self.seq_len
        num_games_batch = self.mini_batch // self.seq_len
        game_indexes = torch.arange(total_games, dtype=torch.long, device='cuda:0')
        flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device='cuda:0').reshape(total_games, self.seq_len)
        sum_loss = 0
        for _ in range(self.mini_epoch):
            for i in range(num_minibatches):
                start = i * num_games_batch
                end = (i + 1) * num_games_batch
                mb_indexes = game_indexes[start:end]
                mbatch = flat_indexes[mb_indexes].flatten()     
                obs_batch = obs[mbatch]
                value_preds_batch = value_preds[mbatch]
                returns_batch = returns[mbatch]
                actions_batch = actions[mbatch]
                rnn_masks_batch = rnn_masks[mbatch]

                batch_dict = {'obs' : obs_batch, 
                            'actions' : actions_batch,
                            'seq_length' : self.seq_len }

                batch_dict['rnn_states'] = [s[:,mb_indexes,:] for s in rnn_states]

                values, _ = self.forward(batch_dict)

                loss = common_losses.critic_loss(value_preds_batch, values, e_clip, returns_batch, self.clip_value)
                losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
                loss = losses[0]

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                if self.truncate_grads:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                sum_loss += loss.item()
        return sum_loss
