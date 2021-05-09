
from rl_games.common import tr_helpers
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses

import torch 
from torch import nn
from torch import optim

class PPGAux:
    def __init__(self, algo, config):
        self.config = config
        assert(not algo.is_discrete, 'Only continuous space supported right now')
        self.writer = algo.writer
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.mixed_precision = algo.mixed_precision
        self.is_rnn = algo.network.is_rnn()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.kl_coef = config.get('kl_coef', 1.0)
        self.is_continuous = True
        self.last_lr = config['learning_rate']
        self.optimizer = optim.Adam(algo.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=algo.weight_decay)
    def train_net(self, algo):
        dataset = algo.dataset
        loss = 0
        for _ in range(self.mini_epoch):
            for idx in range(len(dataset)):
                loss_c, loss_kl = self.calc_gradients(algo, dataset[idx])
        avg_loss_c = loss_c / (self.mini_epoch * algo.num_minibatches)
        avg_loss_kl = loss_kl / (self.mini_epoch * algo.num_minibatches)
        if self.writer != None:
            self.writer.add_scalar('losses/pgg_loss_c', avg_loss_c, algo.frame)
            self.writer.add_scalar('losses/pgg_loss_kl', avg_loss_kl, algo.frame)

    def calc_gradients(self, algo, input_dict):
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        obs_batch = input_dict['obs']
        actions_batch = input_dict['actions']
        obs_batch = algo._preproc_obs(obs_batch)
        lr = self.last_lr
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }
        #if self.use_action_masks:
        #    batch_dict['action_masks'] = input_dict['action_masks']
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = algo.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            kl_loss = torch_ext.policy_kl(mu, sigma, old_mu_batch, old_sigma_batch, False)
            c_loss = common_losses.critic_loss(value_preds_batch, values, algo.e_clip, return_batch, algo.clip_value)

            losses, sum_mask = torch_ext.apply_masks([c_loss.unsqueeze(1), kl_loss.unsqueeze(1)], rnn_masks)
            c_loss, kl_loss = losses[0], losses[1]
            loss = c_loss + kl_loss * self.kl_coef

            if algo.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in algo.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if algo.truncate_grads:
            if algo.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(algo.model.parameters(), algo.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(algo.model.parameters(), algo.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        return c_loss, kl_loss
        