import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common  import common_losses



class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, num_agents, num_steps, num_actors, model, config, writter, _preproc_obs):
        nn.Module.__init__(self)
        self.num_agents, self.num_steps, self.num_actors = num_agents, num_steps, num_actors
        self.state_shape = state_shape
        state_shape = torch_ext.shape_whc_to_cwh(self.state_shape) 
        state_config = {
            'input_shape' : state_shape,
        }
        self.config = config
        self.model = model.build('cvalue', **state_config).cuda()
        self.lr = config['lr']
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.clip_value = config['clip_value']
        self.normalize_input = config['normalize_input']
        self.writter = writter
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr))
        self._preproc_obs = _preproc_obs
        self.frame = 0
        self.running_mean_std = None
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(state_shape).cuda()

    def get_value(self, input_dict):
        obs_batch = input_dict['states']
        if self.normalize_input:
            self.running_mean_std.eval()
        obs_batch = self._preproc_obs(obs_batch, self.running_mean_std)
        value = self.model({'obs' : obs_batch})
        value = value.repeat(1, self.num_agents)
        value = value.view(value.size()[0]*self.num_agents, -1)
        return value

    def train_net(self, input_dict):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()


        obs = input_dict['states']
        batch_size = obs.size()[0]
        value_preds = input_dict['values'].cuda()
        returns = input_dict['returns'].cuda()
        if self.num_agents > 1:
            value_preds = value_preds.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
            returns = returns.view(self.num_actors, self.num_agents, self.num_steps).transpose(0,1)
            value_preds = value_preds.flatten(0)[:batch_size]
            returns = returns.flatten(0)[:batch_size]        

        e_clip = input_dict.get('e_clip', 0.2)
        lr = input_dict.get('lr', self.lr)
        obs = self._preproc_obs(obs, self.running_mean_std)

        mini_batch = self.mini_batch
        num_minibatches = batch_size // mini_batch
        self.frame = self.frame + 1
        for _ in range(self.mini_epoch):
            # returning loss from last epoch
            avg_loss = 0
            for i in range(num_minibatches):
                obs_batch = obs[i * mini_batch: (i + 1) * mini_batch]
                value_preds_batch = value_preds[i * mini_batch: (i + 1) * mini_batch]
                returns_batch = returns[i * mini_batch: (i + 1) * mini_batch]
                values = self.model({'obs' : obs_batch})
                loss = common_losses.critic_loss(value_preds_batch, values, e_clip, returns_batch, self.clip_value)
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

        self.writter.add_scalar('cval/train_loss', avg_loss, self.frame)
        return avg_loss / num_minibatches
