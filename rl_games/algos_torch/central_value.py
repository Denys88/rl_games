import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common  import common_losses



class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, num_agents, num_steps, num_actors, num_actions, model, config, writter):
        nn.Module.__init__(self)
        self.num_agents, self.num_steps, self.num_actors = num_agents, num_steps, num_actors
        self.num_actions = num_actions
        self.state_shape = state_shape
        state_shape = torch_ext.shape_whc_to_cwh(self.state_shape) 
        state_config = {
            'input_shape' : state_shape,
            'actions_num' : num_actions,
            'num_agents' : num_agents,
        }
        self.config = config
        self.model = model.build('cvalue', **state_config)
        self.lr = config['lr']
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.clip_value = config['clip_value']
        self.normalize_input = config['normalize_input']
        self.normalize_reward = config.get('normalize_reward', False)
        self.seq_len = config.get('seq_len', 4)
        self.writter = writter
        self.use_joint_obs_actions = config.get('use_joint_obs_actions', False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr), eps=1e-07)
        self.frame = 0
        self.running_mean_std = None
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(state_shape)

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None
        assert(not self.is_rnn, 'RNN is not supported yet!')
        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            num_seqs = self.steps_num * self.num_actors // self.seq_len
            assert((self.steps_num * batch_size // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((s.size()[0], num_seqs, s.size()[2]), dtype = torch.float32).cuda() for s in self.rnn_states]

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

    def forward(self, input_dict):
        value, rnn_states = self.model(input_dict)

        return value, rnn_states

    def get_value(self, input_dict):
        self.eval()
        obs_batch = input_dict['states']
        is_done = input_dict['is_done']
        actions = input_dict['actions']
        #step_indices = input_dict['step_indices']

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

        if self.num_agents > 1:
            value_preds = value_preds.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
            returns = returns.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
            value_preds = value_preds.flatten(0)[:batch_size]
            returns = returns.flatten(0)[:batch_size]
            if self.use_joint_obs_actions:
                assert(len(actions.size()) == 2, 'use_joint_obs_actions not yet supported in continuous environment for central value')
                actions = actions.view(self.num_actors, self.num_agents, self.num_steps).permute(0,2,1)
                actions = actions.contiguous().view(batch_size, self.num_agents)
            if self.is_rnn:
                assert(False, 'RNN not yet supported for central value')
                rnn_masks = input_dict['rnn_masks']
                rnn_masks = rnn_masks.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
                rnn_masks = rnn_masks.flatten(0)[:batch_size]           

        e_clip = input_dict.get('e_clip', 0.2)
        lr = input_dict.get('lr', self.lr)
        obs = self._preproc_obs(obs)


        self.frame = self.frame + 1
        if self.is_rnn:
            num_games_batch = mini_batch // self.seq_len
            game_indexes = torch.arange(total_games, dtype=torch.long, device='cuda:0')
            flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device='cuda:0').reshape(total_games, self.seq_len)
            avg_loss = 0
            for i in range(num_minibatches):
                batch = torch.range(i * num_games_batch, (i + 1) * num_games_batch - 1, dtype=torch.long, device='cuda:0')
                mb_indexes = game_indexes[batch]
                mbatch = flat_indexes[mb_indexes].flatten()     
                obs_batch = obs[mbatch]
                value_preds_batch = value_preds[mbatch]
                returns_batch = returns[mbatch]
                actions_batch = actions[mbatch]
                batch_dict = {'obs' : obs_batch}
                batch_dict['rnn_states'] = [s[:,mb_indexes,:] for s in self.rnn_states]
                values, _ = self.model()
                loss = common_losses.critic_loss(value_preds_batch, values, e_clip, returns_batch, self.clip_value)
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

        else:
            mini_batch = self.mini_batch
            num_minibatches = batch_size // mini_batch
            for _ in range(self.mini_epoch):
                # returning loss from last epoch
                avg_loss = 0
                for i in range(num_minibatches):
                    batch = torch.range(i * mini_batch, (i + 1) * mini_batch - 1, dtype=torch.long, device='cuda:0')
                    obs_batch = obs[batch]
                    value_preds_batch = value_preds[batch]
                    returns_batch = returns[batch]
                    actions_batch = None
                    if self.use_joint_obs_actions:
                        actions_batch = actions[batch].view(mini_batch * self.num_agents)
                    values, _ = self.forward({'obs' : obs_batch, 'actions' : actions_batch})
                    loss = common_losses.critic_loss(value_preds_batch, values, e_clip, returns_batch, self.clip_value)
                    loss = loss.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    avg_loss += loss.item()

        self.writter.add_scalar('cval/train_loss', avg_loss, self.frame)
        return avg_loss / num_minibatches
