from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import rnd_curiosity

from torch import optim
import torch 
from torch import nn
import numpy as np


class DiscreteA2CAgent(a2c_common.DiscreteA2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        a2c_common.DiscreteA2CBase.__init__(self, base_name, observation_space, action_space, config)
        obs_shape = torch_ext.shape_whc_to_cwh(self.state_shape) 

        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'games_num' : 1,
            'batch_num' : 1,
        } 
        self.model = self.network.build(config)
        self.model.cuda()
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr))
        #self.optimizer = torch_ext.RangerQH(self.model.parameters(), float(self.last_lr))

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).cuda()

        if self.has_curiosity:
            self.rnd_curiosity = rnd_curiosity.RNDCurisityTrain(torch_ext.shape_whc_to_cwh(self.state_shape), self.curiosity_config['network'], 
                                    self.curiosity_config, self.writer, lambda obs: self._preproc_obs(obs))


    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()


    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = {'epoch': self.epoch_num, 'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}
        if self.normalize_input:
            state['running_mean_std'] = self.running_mean_std.state_dict()
        if self.has_curiosity:
            state['rnd_nets'] = self.rnd_curiosity.state_dict()
        torch_ext.save_scheckpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.epoch_num = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.has_curiosity:
            self.rnd_curiosity.load_state_dict(checkpoint['rnd_nets'])
            for state in self.rnd_curiosity.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def get_masked_action_values(self, obs, action_masks):
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
        return action.detach().cpu(), value.detach().cpu(), neglogp.detach(), logits.detach(), None


    def get_action_values(self, obs):
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        return action.detach().cpu(), value.detach().cpu(), neglogp.detach(), None

    def get_values(self, obs):
        obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        return value.detach().cpu()

    def get_intrinsic_reward(self, obs):
        return self.rnd_curiosity.get_loss(obs)

    def train_intrinsic_reward(self, dict):
        obs = dict['obs']
        self.rnd_curiosity.train(obs)

    def get_weights(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())
    
    def set_weights(self, weights):
        torch.nn.utils.vector_to_parameters(weights, self.model.parameters())

    def train_actor_critic(self, input_dict):
        self.model.train()
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        input_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'inputs' : obs_batch
        }
        action_log_probs, values, entropy = self.model(input_dict)

        if self.ppo:
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip) * advantage
            a_loss = torch.max(-surr1, -surr2).mean()
        else:
            a_loss = (action_log_probs * advantage).mean()

        values = torch.squeeze(values)
        if self.clip_value:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses,
                                         value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        if self.has_curiosity:
            c_loss = c_loss.sum(dim=1).mean()
        else:
            c_loss = c_loss.mean()
        loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_log_probs_batch - action_log_probs)**2).mean()
            kl_dist = kl_dist.item()
            if self.is_adaptive_lr:
                if kl_dist > (2.0 * self.lr_threshold):
                    self.last_lr = max(self.last_lr / 1.5, 1e-6)
                if kl_dist < (0.5 * self.lr_threshold):
                    self.last_lr = min(self.last_lr * 1.5, 1e-2)    
        return a_loss.item(), c_loss.item(), entropy.item(), kl_dist, self.last_lr, lr_mul