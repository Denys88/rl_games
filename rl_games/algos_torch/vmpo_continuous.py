from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import ppg_aux

from torch import optim
import torch 
from torch import nn
import numpy as np
import gym

class VMPOAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, config):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, config)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1)
        }

        self.model = self.network.build(config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True).to(self.device)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True).to(self.device)

        self.eta = torch.tensor(1.0).float().to(self.device)
        self.alpha = torch.tensor(1.0).float().to(self.device)
        self.eta.requires_grad = True
        self.alpha.requires_grad = True
        self.eps_eta = config.get('eps_eta', 0.01)
        self.eps_alpha = config.get('eps_alpha', 1)
        params = [
                {'params': self.model.parameters()},
                {'params': self.eta},
                {'params': self.alpha}
            ]
        self.optimizer = optim.Adam(params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_input:
            if isinstance(self.observation_space,gym.spaces.Dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape).to(self.ppo_device)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'num_steps' : self.horizon_length, 
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len, 
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        
  
        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                            or (not self.has_central_value) 
        self.algo_observer.after_init(self)
    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
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
        current_batch_size = obs_batch.size()[0]
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            advprobs = torch.stack((advantage,action_log_probs))
            
            advprobs = advprobs[:,torch.sort(advprobs[0],descending=True).indices]
            good_advantages = advprobs[0,:current_batch_size//2]
            good_logprobs = advprobs[1,:current_batch_size//2]
            
            # Get losses
            phis = torch.nn.functional.softmax(good_advantages/self.eta.detach(), dim=-1)
            L_pi = torch.mean(phis*good_logprobs)
            L_eta = self.eta*self.eps_eta+self.eta*torch.log(torch.mean(torch.exp(good_advantages/self.eta)))
            
            KL = torch_ext.policy_kl(mu, sigma, old_mu_batch, old_sigma_batch, False)
            
            L_alpha = torch.mean(self.alpha*(self.eps_alpha-KL.detach())+self.alpha.detach()*KL)
            
            a_loss = L_pi + L_eta + L_alpha
            
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks([c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            c_loss, entropy, b_loss = losses[0], losses[1], losses[2]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        with torch.no_grad():
            self.eta.copy_(torch.clamp(self.eta,min=1e-5))
            self.alpha.copy_(torch.clamp(self.alpha,min=1e-5))                    
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


