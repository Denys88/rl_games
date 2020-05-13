from common.player import BasePlayer
import torch 
from torch import nn
import algos_torch.torch_ext
import numpy as np
from algos_torch.running_mean_std import RunningMeanStd

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
        self.actions_low = self.action_space.low
        self.actions_high = self.action_space.high
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        print(self.state_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : algos_torch.torch_ext.shape_whc_to_cwh(self.state_shape),
            'games_num' : 1,
            'batch_num' : 1,
        } 
        self.model = self.network.build(config)
        self.model.cuda()
        self.model.eval()

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(observation_space.shape).cuda()
            self.running_mean_std.eval()
            

    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == np.uint8:
            obs_batch = torch.cuda.ByteTensor(obs_batch)
            obs_batch = obs_batch.float() / 255.0
        else:
            obs_batch = torch.cuda.FloatTensor(obs_batch)
        if len(obs_batch.size()) == 3:
            obs_batch = obs_batch.permute((0, 2, 1))
        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def get_action(self, obs, is_determenistic = False):
        obs = self._preproc_obs(np.expand_dims(obs, axis=0))
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
        }
        with torch.no_grad():
            neglogp, value, action, mu, sigma = self.model(input_dict)
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = np.squeeze(current_action.detach().cpu().numpy())
        return  rescale_actions(self.actions_low, self.actions_high, np.clip(current_action, -1.0, 1.0))

    def restore(self, fn):
        checkpoint = algos_torch.torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state



class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        self.actions_num = self.action_space.n
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        observation_shape = algos_torch.torch_ext.shape_whc_to_cwh(self.state_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : observation_shape,
            'games_num' : 1,
            'batch_num' : 1,
        } 
        self.model = self.network.build(config)
        self.model.cuda()
        self.model.eval()

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(observation_shape).cuda()
            self.running_mean_std.eval()
            

    def _preproc_obs(self, obs_batch):
        if obs_batch.dtype == np.uint8:
            obs_batch = torch.cuda.ByteTensor(obs_batch)
            obs_batch = obs_batch.float() / 255.0
        else:
            obs_batch = torch.cuda.FloatTensor(obs_batch)
        if len(obs_batch.size()) == 3:
            obs_batch = obs_batch.permute((0, 2, 1))
        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch

    def get_masked_action(self, obs, action_masks, is_determenistic = False):
        if self.num_agents == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).cuda()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
            'action_masks' : action_masks
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        
        if is_determenistic:
            return np.argmax(logits.squeeze().detach().cpu().numpy(), axis=1)
        else:    
            return action.squeeze().detach().cpu().numpy()


    def get_action(self, obs, is_determenistic = False):
        if self.num_agents == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        
        if is_determenistic:
            return np.argmax(logits.detach().cpu().numpy(), axis=1).squeeze()
        else:    
            return action.squeeze().detach().cpu().numpy()

    def restore(self, fn):
        checkpoint = algos_torch.torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        if self.network.is_rnn():
            self.last_state = self.initial_state