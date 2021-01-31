from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

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
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1)
        }

        self.model = self.network.build(config)
        self.model.to(self.ppo_device)

        self.init_rnn_from_model(self.model)

        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape' : torch_ext.shape_whc_to_cwh(self.state_shape), 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'num_steps' : self.steps_num, 
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len, 
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', False)        
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

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
        action_masks = torch.Tensor(action_masks).to(self.ppo_device)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'action_masks' : action_masks,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                input_dict = {
                    'is_train': False,
                    'states' : obs['states'],
                    #'actions' : action,
                }
                value = self.get_central_value(input_dict)
                res_dict['value'] = value

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return res_dict

    def train_actor_critic(self, input_dict, opt_step = True):
        self.set_train()
        self.calc_gradients(input_dict, opt_step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        return self.train_result

    def calc_gradients(self, input_dict, opt_step):
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

        res_dict = self.model(batch_dict)
        action_log_probs = res_dict['prev_neglogp']
        values = res_dict['value']
        entropy = res_dict['entropy']
        a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

        if self.use_experimental_cv:
            c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
        else:
            if self.has_central_value:
                c_loss = torch.zeros(1, device=self.ppo_device)
            else:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                
        losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1)], rnn_masks)

        a_loss, c_loss, entropy = losses[0], losses[1], losses[2]
        loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef

        for param in self.model.parameters():
            param.grad = None

        loss.backward()
        if self.config['truncate_grads']:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        if opt_step:
            self.optimizer.step()
        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_log_probs_batch - action_log_probs)**2)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / sum_mask
            else:
                kl_dist = kl_dist.mean()
            kl_dist = kl_dist.item()

        self.train_result =  (a_loss.item(), c_loss.item(), entropy.item(), kl_dist,self.last_lr, lr_mul)