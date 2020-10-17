from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import torch 
from torch import nn
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class PpoPlayerContinuous(BasePlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        obs_shape = torch_ext.shape_whc_to_cwh(self.state_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()   

    def get_action(self, obs, is_determenistic = False):
        if len(obs.size()) == len(self.state_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            neglogp, value, action, mu, sigma, self.states = self.model(input_dict)
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())
        return  rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

def reset(self):
    self.init_rnn()

class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        self.actions_num = self.action_space.n
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']

        obs_shape = torch_ext.shape_whc_to_cwh(self.state_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()      

    def get_masked_action(self, obs, action_masks, is_determenistic = True):
        if len(obs.size()) == len(self.state_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            neglogp, value, action, logits, self.states = self.model(input_dict)

        if is_determenistic:
            return torch.argmax(logits.squeeze().detach(), axis=-1)
        else:    
            return action.squeeze()

    def get_action(self, obs, is_determenistic = False):
        if len(obs.size()) == len(self.state_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            neglogp, value, action, logits, self.states = self.model(input_dict)
        
        if is_determenistic:
            return torch.argmax(logits.detach(), axis=1).squeeze()
        else:    
            return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()