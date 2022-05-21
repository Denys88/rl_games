from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_based.model_trainer import ModelTrainer
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_based.env_model import ModelEnvironment
from rl_games.common.experience import ExperienceBuffer, VectorizedReplayBuffer
from torch import optim
import torch
from torch import nn
import numpy as np
import gym
import time
import copy

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class MBAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'value_size': self.env_info.get('value_size', 1)
        }


        self.model = self.network.build(config)
        self.model.to(self.ppo_device)
        print(model_config)
        self.env_model = self.config['model_network'].build('env_model', **model_config)
        self.env_model.to(self.ppo_device)

        self.model_training_config = params['model_training_config'].copy()
        self.replay_buffer_size = self.model_training_config.pop('replay_buffer_size')
        self.model_trainer = ModelTrainer(self.env_model, **self.model_training_config)
        self.fake_env = ModelEnvironment(self.env_model, self.env_info)
        self.states = None
        self.is_rnn = False
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                                    weight_decay=self.weight_decay)
        self.model_horizon_length = self.horizon_length
        self.max_model_horizon = 16
        self.original_horizon_length = self.horizon_length

        model_info = {
            'num_actors' : self.num_actors * self.model_horizon_length//self.max_model_horizon,
            'horizon_length' : self.max_model_horizon,
            'has_central_value' : False,
            'use_action_masks' : False,
        }

        self.replay_buffer = VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                               self.env_info['action_space'].shape,
                                                               self.replay_buffer_size,
                                                               self.device)

        self.model_experience_buffer = ExperienceBuffer(self.env_info, model_info, self.ppo_device)
        self.model_datasets = [datasets.PPODataset(self.num_agents * self.num_actors*  self.model_horizon_length, self.minibatch_size, False, False,
                                           self.ppo_device, self.seq_len)]

        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                           self.ppo_device, self.seq_len)
        self.original_env = self.vec_env
        self.original_obs = None
        self.original_dataset = self.dataset
        self.original_experience_buffer = None
        self.original_dones = None

        if self.normalize_value:
            self.value_mean_std = self.model.value_mean_std
        self.has_value_loss = True

        self.dataset_list = datasets.DatasetList([self.dataset] + self.model_datasets) #
        #self.dataset_list = datasets.DatasetList([self.dataset])
        self.algo_observer.after_init(self)



    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

        self.config['model_network'] = builder.get_network_builder().load(params['model_network'])
        has_central_value_net = self.config.get('central_value_config') is not  None

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['next_obses'] = batch_dict['next_obses']
        self.dataset.values_dict['rewards'] = batch_dict['rewards']
        return


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

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

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

            a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy = losses[0], losses[1], losses[2]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        self.trancate_gradients()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), torch.zeros_like(a_loss))

    def prepare_model_env(self):
        start_obs = []
        for _ in range(self.horizon_length // self.max_model_horizon):
            start_obs_idx = np.random.randint(0, self.original_horizon_length)
            start_obs.append(self.original_experience_buffer.tensor_dict['obses'][start_obs_idx])
        start_obs = torch.cat(start_obs)

        self.original_obs = copy.deepcopy(self.obs)
        self.obs['obs'] = start_obs
        self.fake_env.reset(start_obs)
        self.horizon_length = self.max_model_horizon
        self.dataset = self.model_datasets[0]
        self.experience_buffer = self.model_experience_buffer
        self.vec_env = self.fake_env
        self.original_dones = copy.deepcopy(self.dones)
        self.dones = torch.zeros(start_obs.size()[0], device=self.device)
        self.is_tensor_obses = True

    def prepare_original_env(self):
        self.dataset = self.original_dataset
        self.experience_buffer = self.original_experience_buffer if self.original_experience_buffer != None else self.experience_buffer
        self.obs = copy.deepcopy(self.original_obs) if self.original_obs != None else self.obs
        self.vec_env = self.original_env
        self.horizon_length = self.original_horizon_length
        self.dones = copy.deepcopy(self.original_dones) if self.original_dones != None else self.dones
        self.is_tensor_obses = False

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def init_tensors(self):
        super().init_tensors()
        self.tensor_list += ['next_obses', 'rewards']
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.model_experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.model_experience_buffer.tensor_dict['obses'])
        self.model_experience_buffer.tensor_dict['rewards'] = torch.zeros_like(self.model_experience_buffer.tensor_dict['rewards'])
        self.original_experience_buffer = self.experience_buffer

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict = self.play_steps()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)

        with torch.no_grad():
            batch_dict = self.play_fake_steps()
            self.prepare_dataset(batch_dict)

        self.prepare_original_env()
        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = None


        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []

        self.model_trainer.train_epoch(self)
        self.set_train()
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset_list)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(
                    self.dataset_list[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset_list.update_mu_sigma(cmu, csigma)

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef,
                                                                            self.epoch_num, 0, kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num,
                                                                        0, av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch
        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict[
                   'step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def play_steps(self):
        update_list = self.update_list
        self.prepare_original_env()
        step_time = 0.0

        for n in range(self.horizon_length):
            last_obs = self.obs
            last_dones = self.dones
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.replay_buffer.add(last_obs['obs'],
                                   res_dict['actions'],
                                   shaped_rewards,
                                   self.obs['obs'],
                                   last_dones.unsqueeze(1))
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones


        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values



        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        return batch_dict

    def play_fake_steps(self):
        update_list = self.update_list
        self.prepare_model_env()
        step_time = 0.0
        for n in range(self.max_model_horizon):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            #shaped_rewards = self.rewards_shaper(rewards)
            shaped_rewards = rewards
            #if self.value_bootstrap and 'time_outs' in infos:
            #    shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            #self.experience_buffer.update_data('next_obses', n, self.obs['obs'])

            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            not_dones = 1.0 - self.dones.float()

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

