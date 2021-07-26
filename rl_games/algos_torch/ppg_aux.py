
from rl_games.common import tr_helpers
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.common.datasets import DatasetList
import torch 
from torch import nn
from torch import optim
import copy

class PPGAux:
    def __init__(self, algo, config):
        self.config = config
        self.writer = algo.writer
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.mixed_precision = algo.mixed_precision
        self.is_rnn = algo.network.is_rnn()
        self.kl_coef = config.get('kl_coef', 1.0)
        self.n_aux = config.get('n_aux', 16)
        self.is_continuous = True
        self.last_lr = config['learning_rate']

        self.optimizer = optim.Adam(algo.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=algo.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self._freeze_grads(algo.model)
        self.value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, algo.model.parameters()), float(self.last_lr), eps=1e-08, weight_decay=algo.weight_decay)
        self.value_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self._unfreeze_grads(algo.model)
        self.dataset_list = DatasetList()

    def _freeze_grads(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.a2c_network.value.weight.requires_grad = True
        model.a2c_network.value.bias.requires_grad = True

    def _unfreeze_grads(self, model):
        for param in model.parameters():
            param.requires_grad = True       

    def train_value(self, algo, input_dict):
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        obs_batch = input_dict['obs']
        actions_batch = input_dict['actions']
        obs_batch = algo._preproc_obs(obs_batch)

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

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = algo.model(batch_dict)
            values = res_dict['values']
            c_loss = common_losses.critic_loss(value_preds_batch, values, algo.e_clip, return_batch, algo.clip_value)
            losses, sum_mask = torch_ext.apply_masks([c_loss], rnn_masks)
            c_loss = losses[0]
            loss = c_loss

            if algo.multi_gpu:
                self.value_optimizer.zero_grad()
            else:
                for param in algo.model.parameters():
                    param.grad = None

        self.value_scaler.scale(loss).backward()
        if algo.truncate_grads:
            if algo.multi_gpu:
                self.value_optimizer.synchronize()
                self.value_scaler.unscale_(self.value_optimizer)
                nn.utils.clip_grad_norm_(algo.model.parameters(), algo.grad_norm)
                with self.value_optimizer.skip_synchronize():
                    self.value_scaler.step(self.value_optimizer)
                    self.value_scaler.update()
            else:
                self.value_scaler.unscale_(self.value_optimizer)
                nn.utils.clip_grad_norm_(algo.model.parameters(), algo.grad_norm)
                self.value_scaler.step(self.value_optimizer)
                self.value_scaler.update()    
        else:
            self.value_scaler.step(self.value_optimizer)
            self.value_scaler.update()
        
        return loss.detach()

    def update(self, algo):
        self.dataset_list.add_dataset(algo.dataset)

    def train_net(self, algo):
        self.update(algo)
        if algo.epoch_num % self.n_aux != 0:
            return
        self.old_model = copy.deepcopy(algo.model)
        self.old_model.eval()
        dataset = self.dataset_list

        for _ in range(self.mini_epoch):
            for idx in range(len(dataset)):
                loss_c, loss_kl = self.calc_gradients(algo, dataset[idx])
        avg_loss_c = loss_c / len(dataset)
        avg_loss_kl = loss_kl / len(dataset)
        if self.writer != None:
            self.writer.add_scalar('losses/pgg_loss_c', avg_loss_c, algo.frame)
            self.writer.add_scalar('losses/pgg_loss_kl', avg_loss_kl, algo.frame)

        self.dataset_list.clear()

    def calc_gradients(self, algo, input_dict):
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        obs_batch = input_dict['obs']
        actions_batch = input_dict['actions']
        obs_batch = algo._preproc_obs(obs_batch)

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
            with torch.no_grad():
                old_dict = self.old_model(batch_dict.copy())

            res_dict = algo.model(batch_dict)
            values = res_dict['values']

            if 'mu' in res_dict:
                old_mu_batch = input_dict['mu']
                old_sigma_batch = input_dict['sigma']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                #kl_loss = torch_ext.policy_kl(mu, sigma.detach(), old_mu_batch, old_sigma_batch, False)
                kl_loss = torch.abs(mu - old_mu_batch)
            else:
                kl_loss = algo.model.kl(res_dict, old_dict)
            c_loss = common_losses.critic_loss(value_preds_batch, values, algo.e_clip, return_batch, algo.clip_value)
            losses, sum_mask = torch_ext.apply_masks([c_loss, kl_loss.unsqueeze(1)], rnn_masks)
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
        