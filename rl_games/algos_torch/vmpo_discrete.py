from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
from torch import nn
import numpy as np
import gym

class VMPOAgent(a2c_common.DiscreteA2CBase):
    def __init__(self, base_name, params):
        a2c_common.DiscreteA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.eta = config.get('eta', 1.0)
        self.alpha = config.get('alpha', 0.1)
        self.eta = torch.tensor(self.eta).float().to(self.device)
        self.alpha = torch.tensor(self.alpha).float().to(self.device)
        self.eta.requires_grad = True
        self.alpha.requires_grad = True

        self.eps_eta = config.get('eps_eta', 0.02)
        self.eps_alpha = config.get('eps_alpha', [0.005, 0.01])
        self.is_const_eps_alpha = np.isscalar(self.eps_alpha)
        if not self.is_const_eps_alpha:
            self.eps_alpha = [torch.tensor(self.eps_alpha[0]).float().to(self.device).log() \
                            ,torch.tensor(self.eps_alpha[1]).float().to(self.device).log()]
        params = [
                {'params': self.model.parameters()},
                {'params': self.eta},
                {'params': self.alpha}
            ]
        self.optimizer = optim.Adam(params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)



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
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
  
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
        old_action_neglog_probs_batch = input_dict['old_logp_actions']
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
        current_batch_size = obs_batch.size()[0]
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_neglog_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']


            advprobs = torch.stack((advantage,action_neglog_probs))
            advprobs = advprobs[:,torch.sort(advprobs[0],descending=True).indices]
            good_advantages = advprobs[0,:current_batch_size//2]
            good_neglogprobs = advprobs[1,:current_batch_size//2]

            # Get losses
            '''
           with torch.no_grad():
                aug_adv_max = (good_advantages / self.eta).max()
                aug_adv = (good_advantages / self.eta.detach() - aug_adv_max).exp()
                norm_aug_adv = aug_adv / aug_adv.sum()            
            '''

            #pi_loss2 = (norm_aug_adv * good_logprobs).sum()
            pi_loss = torch.exp(good_advantages / self.eta.detach()) / torch.sum(torch.exp(good_advantages / self.eta.detach()))*good_neglogprobs
            pi_loss = pi_loss.mean()
            eta_loss = self.eta * self.eps_eta + self.eta * (good_advantages / self.eta).exp().mean().log()
            kl = self.model.kl({'logits' : input_dict['old_logits']}, res_dict)
            if self.is_const_eps_alpha:
                coef_alpha = self.eps_alpha
            else:
                coef_alpha = torch.distributions.Uniform(self.eps_alpha[0], self.eps_alpha[1]).sample().exp()
            alpha_loss = torch.mean(self.alpha*(coef_alpha-kl.detach())+self.alpha.detach()*kl)
            
            a_loss = pi_loss + eta_loss + alpha_loss
            
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks([c_loss, entropy.unsqueeze(1)], rnn_masks)
            c_loss, entropy = losses[0], losses[1]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef
            
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
                nn.utils.clip_grad_norm_([*self.model.parameters(), self.eta, self.alpha], self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_([*self.model.parameters(), self.eta, self.alpha], self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            self.eta.copy_(torch.clamp(self.eta,min=1e-8))
            self.alpha.copy_(torch.clamp(self.alpha,min=1e-8))

        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_neglog_probs_batch - action_neglog_probs)**2)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel() # / sum_mask
            else:
                kl_dist = kl_dist.mean()

        self.train_result =  (a_loss, c_loss, entropy, kl_dist,self.last_lr, lr_mul)

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        a2c_common.DiscreteA2CBase.write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
        if self.writer:
            self.writer.add_scalar('info/eta', self.eta, frame)
            self.writer.add_scalar('info/alpha', self.alpha, frame)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def add_custom_data_to_dataset(self, dataset_dict, batch_dict):
        dataset_dict['old_logits'] = self.get_action_values({'obs' : batch_dict['obses']})['logits']
        return dataset_dict


