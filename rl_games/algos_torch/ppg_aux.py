
from rl_games.common import tr_helpers
from rl_games.algos_torch import torch_ext
import torch 
from torch import nn


class PPGAux:
    def __init__(self, algo, config)
        self.config = config
        assert(algo.is_continuous, 'Only continuous space supported right now')
        self.writter = algo.writter
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.mixed_precision = algo.mixed_precision
        self.is_rnn = algo.is_rnn
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.kl_coef = config.get('kl_coef', 1.0)
        self.is_continuous = algo.is_continuous
        self.last_lr = config['learning_rate']

    def train_net(self, algo):
        dataset = algo.dataset
        model = algo.model
        loss = 0
        for _ in range(self.mini_epoch):
            for idx in range(len(dataset)):
                loss_c, loss_kl = self.calc_gradients(dataset[idx])
        avg_loss_c = loss_c / (self.mini_epoch * self.num_minibatches)
        avg_loss_kl = loss_kl / (self.mini_epoch * self.num_minibatches)
        if self.writter != None:
            self.writter.add_scalar('losses/pgg_loss_c', avg_loss_c, self.frame)
            self.writter.add_scalar('losses/pgg_loss_kl', avg_loss_kl, self.frame)
        self.frame += self.batch_size
        return avg_loss

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        old_mu_batch = input_dict['mus']
        old_sigma_batch = input_dict['sigmas']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        lr = self.last_lr
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }
        if self.use_action_masks:
            batch_dict['action_masks'] = input_dict['action_masks']
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
            sigma = res_dict['sigma']

            kl_dist = torch_ext.policy_kl(mu, sigma, old_mu_batch, old_sigma_batch, False)
            c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)

            losses, sum_mask = torch_ext.apply_masks([c_loss.unsqueeze(1), kl_loss.unsqueeze(1)], rnn_masks)
            c_loss, kl_loss = losses[0], losses[1]
            loss = c_loss + kl_loss * self.kl_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
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
        
        return c_loss, kl_loss
        