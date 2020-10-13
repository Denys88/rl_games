from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import central_value, rnd_curiosity
from rl_games.common import common_losses

from torch import optim
import torch 
from torch import nn
import numpy as np

class DiscreteA2CAgent(a2c_common.DiscreteA2CBase):
    def __init__(self, base_name, config):
        a2c_common.DiscreteA2CBase.__init__(self, base_name, config)
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape) 

        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents
        } 
        self.model = self.network.build(config)
        self.model.cuda()

        self.init_rnn_from_model(self.model)

        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-07, weight_decay=self.weight_decay)
        #self.optimizer = torch_ext.RangerQH(self.model.parameters(), float(self.last_lr))

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).cuda()

        if self.has_central_value:
            self.central_value_net = central_value.CentralValueTrain(torch_ext.shape_whc_to_cwh(self.state_shape), self.num_agents, self.steps_num, self.num_actors, self.actions_num, self.central_value_config['network'], 
                                    self.central_value_config, self.writer).cuda()

        if self.has_curiosity:
            self.rnd_curiosity = rnd_curiosity.RNDCuriosityTrain(torch_ext.shape_whc_to_cwh(self.obs_shape), self.curiosity_config['network'], 
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
        state = self.get_full_state_weights()
        torch_ext.save_scheckpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)


    def get_masked_action_values(self, obs, action_masks):
        processed_obs = self._preproc_obs(obs['obs'])
        action_masks = torch.Tensor(action_masks).cuda()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'action_masks' : action_masks,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            neglogp, value, action, logits, rnn_states = self.model(input_dict)
            if self.has_central_value:
                input_dict = {
                    'is_train': False,
                    'states' : obs['states'],
                    'is_done': self.dones,
                    'actions' : action,
                }
                value = self.get_central_value(input_dict)
                
        return action.detach(), value.detach().cpu(), neglogp.detach(), logits.detach(), rnn_states

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
            neglogp, value, action, logits, rnn_states = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : action,
                    #'rnn_states' : self.rnn_states
                }
                value = self.central_value_net(input_dict)

        return action.detach(), value.detach().cpu(), neglogp.detach(), rnn_states

    def get_values(self, obs, actions=None):
        if self.has_central_value:
            states = obs['states']
            self.central_value_net.eval()
            input_dict = {
                'is_train': False,
                'states' : states,
                'actions' : actions,
                'is_done': self.dones,
            }
            return self.get_central_value(input_dict).detach().cpu()
        else:
            self.model.eval()
            processed_obs = self._preproc_obs(obs['obs'])
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
                'rnn_states' : self.rnn_states
            }
            with torch.no_grad():
                _, value, _, _, _ = self.model(input_dict)
            return value.detach().cpu()

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

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        action_log_probs, values, entropy, _ = self.model(batch_dict)

        a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

        if self.has_central_value:
            c_loss = torch.zeros(1).cuda()
        else:
            c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)

        if self.has_curiosity:
            c_loss = c_loss.sum(dim=1)

        losses, sum_mask = torch_ext.apply_masks([a_loss, c_loss, entropy], rnn_masks)
        a_loss, c_loss, entropy = losses[0], losses[1], losses[2]

        loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef


        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_log_probs_batch - action_log_probs)**2)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / sum_mask
            else:
                kl_dist = kl_dist.mean()
            kl_dist = kl_dist.item()
            if self.is_adaptive_lr:
                if kl_dist > (2.0 * self.lr_threshold):
                    self.last_lr = max(self.last_lr / 1.5, 1e-6)
                if kl_dist < (0.5 * self.lr_threshold):
                    self.last_lr = min(self.last_lr * 1.5, 1e-2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr
        return a_loss.item(), c_loss.item(), entropy.item(), kl_dist, self.last_lr, lr_mul