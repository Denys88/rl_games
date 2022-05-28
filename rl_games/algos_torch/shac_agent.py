from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import vecenv, schedulers, experience

from rl_games.common.a2c_common import  ContinuousA2CBase
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import  model_builder
from torch import optim
import torch
import time
import os
import copy


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class SHACAgent(ContinuousA2CBase):

    def __init__(self, base_name, params):
        ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.critic_lr = self.config.get('critic_learning_rate', 0.0001)
        self.use_target_critic = self.config.get('use_target_critic', True)
        self.target_critic_alpha = self.config.get('target_critic_alpha', 0.4)

        self.max_episode_length = 1000 # temporary hardcoded
        self.actor_model = self.network.build(build_config)
        self.critic_model = self.critic_network.build(build_config)

        self.actor_model.to(self.ppo_device)
        self.critic_model.to(self.ppo_device)
        self.target_critic = copy.deepcopy(self.critic_model)

        if self.normalize_input:
            self.critic_model.running_mean_std = self.actor_model.running_mean_std
            self.target_critic.running_mean_std = self.critic_model.running_mean_std
        if self.normalize_value:
            self.target_critic.value_mean_std = self.critic_model.value_mean_std
            self.actor_model.value_mean_std = None

        self.states = None
        self.model = self.actor_model
        self.init_rnn_from_model(self.actor_model)

        self.last_lr = float(self.last_lr)
        self.betas = self.config.get('betas',[0.9, 0.999])
        self.optimizer = self.actor_optimizer = optim.Adam(self.actor_model.parameters(), float(self.last_lr), betas=self.betas, eps=1e-08,
                                    weight_decay=self.weight_decay)
        # self.critic_optimizer = optim.Adam(self.critic_model.parameters(), float(self.critic_lr), betas=self.betas, eps=1e-08,
        #                                    weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), float(self.last_lr), betas=self.betas, eps=1e-08,
                                    weight_decay=self.weight_decay)

        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                           self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.critic_model.value_mean_std

        self.algo_observer.after_init(self)

    def play_steps(self):
        update_list = self.update_list
        accumulated_rewards = torch.zeros((self.horizon_length + 1, self.num_actors), dtype=torch.float32, device=self.device)
        actor_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        mb_values = self.experience_buffer.tensor_dict['values']
        gamma = torch.ones(self.num_actors, dtype = torch.float32, device = self.device)
        step_time = 0.0
        self.critic_model.eval()
        self.target_critic.eval()
        self.actor_model.train()
        if self.normalize_input:
            self.actor_model.running_mean_std.train()
        obs = self.initialize_trajectory()
        last_values = None
        for n in range(self.horizon_length):
            res_dict = self.get_actions(obs)
            if last_values is None:
                res_dict['values'] = self.get_values(obs)
            else:
                res_dict['values'] = last_values

            with torch.no_grad():
                self.experience_buffer.update_data('obses', n, obs['obs'].detach())
                self.experience_buffer.update_data('dones', n, self.dones.detach())

                self.experience_buffer.update_data('values', n, res_dict['values'].detach())

            step_time_start = time.time()
            actions = torch.tanh(res_dict['actions'])
            obs, rewards, self.dones, infos = self.env_step(actions)
            step_time_end = time.time()
            episode_ended = self.current_lengths == self.max_episode_length - 1
            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            real_obs = self.obs_to_tensors(infos['obs_before_reset'])
            shaped_rewards += self.gamma * self.get_values(real_obs) * episode_ended.unsqueeze(1).float()
            self.experience_buffer.update_data('rewards', n, shaped_rewards.detach())

            self.current_rewards += rewards.detach()
            self.current_lengths += 1

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)
            fdones = self.dones.float()
            not_dones = 1.0 - fdones

            accumulated_rewards[n + 1] = accumulated_rewards[n] + gamma * shaped_rewards.squeeze(1)

            last_values = self.get_values(obs)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            accumulated_rewards[n + 1, :] = accumulated_rewards[n, :] + gamma * shaped_rewards.squeeze(1)
            if n < self.horizon_length - 1:
                actor_loss = actor_loss - (
                            accumulated_rewards[n + 1, env_done_indices]).sum()
            else:
                actor_loss = actor_loss - (
                            accumulated_rewards[n + 1, :] + self.gamma * gamma * last_values.squeeze() * (1.0-episode_ended.float()) * not_dones).sum()
        gamma = gamma * self.gamma
        gamma[env_done_indices] = 1.0
        accumulated_rewards[n + 1, env_done_indices] = 0.0
        fdones = self.dones.float().detach()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float().detach()
        mb_rewards = self.experience_buffer.tensor_dict['rewards'].detach()
        mb_advs = self.discount_values(fdones, last_values.detach(), mb_fdones, mb_values.detach(), mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        actor_loss = actor_loss / (self.horizon_length * self.num_actors)
        return batch_dict, actor_loss

    def env_step(self, actions):
        #actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos

    def load_networks(self, params):
        ContinuousA2CBase.load_networks(self, params)
        if 'critic_config' in self.config:
            builder = model_builder.ModelBuilder()
            print('Adding Critic Network')
            network = builder.load(params['config']['critic_config'])
            self.critic_network = network

    def get_actions(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        input_dict = {
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
        res_dict = self.actor_model(input_dict)
        return res_dict

    def get_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        processed_obs = self._preproc_obs(obs['obs'])
        if self.use_target_critic:
            result = self.target_critic(input_dict)
        else:
            result = self.critic_model(input_dict)
        value = result['values']
        return value

    def initialize_trajectory(self):
        #obs = self.vec_env.reset()
        obs = self.vec_env.initialize_trajectory()
        obs = self.obs_to_tensors(obs)
        return obs

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

    def prepare_critic_dataset(self, batch_dict):
        obses = batch_dict['obses'].detach()
        returns = batch_dict['returns'].detach()
        dones = batch_dict['dones'].detach()
        values = batch_dict['values'].detach()
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()



        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['returns'] = returns
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks

        self.dataset.update_values_dict(dataset_dict)

    def train_actor(self, actor_loss):
        self.actor_model.train()

        self.actor_optimizer.zero_grad(set_to_none=True)

        actor_loss.backward()

        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.grad_norm)

        self.actor_optimizer.step()

        return actor_loss.detach()

    def train_critic(self, batch):
        self.critic_model.train()
        if self.normalize_input:
            self.critic_model.running_mean_std.eval()
        if self.normalize_value:
            self.critic_model.value_mean_std.eval()

        obs_batch = self._preproc_obs(batch['obs'])
        value_preds_batch = batch['old_values']
        returns_batch = batch['returns']
        dones_batch = batch['dones']
        rnn_masks_batch = batch.get('rnn_masks')
        batch_dict = {'obs' : obs_batch,
                    'seq_length' : self.seq_len,
                    'dones' : dones_batch}

        res_dict = self.critic_model(batch_dict)
        values = res_dict['values']
        loss = common_losses.critic_loss(value_preds_batch, values, self.e_clip, returns_batch, self.clip_value)
        losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
        critic_loss = losses[0]
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()

        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.grad_norm)

        self.critic_optimizer.step()

        return critic_loss.detach()

    def update_lr(self, actor_lr, critic_lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr])
            self.hvd.broadcast_value(lr_tensor, 'learning_rate')
            lr = lr_tensor.item()

        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr

        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

    def train_epoch(self):
        play_time_start = time.time()
        batch_dict, actor_loss = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_critic_dataset(batch_dict)
        self.algo_observer.after_steps()
        a_loss = self.train_actor(actor_loss)
        a_losses = [a_loss]
        c_losses = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                c_loss = self.train_critic(self.dataset[i])
                c_losses.append(c_loss)

            self.diagnostics.mini_epoch(self, mini_ep)

        # update target critic
        with torch.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic_model.parameters(), self.target_critic.parameters()):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1. - alpha) * param.data)
        self.last_lr, _ = self.scheduler.update(self.last_lr, 0, self.epoch_num,   0, None)
        #self.critic_lr, _ = self.scheduler.update(self.critic_lr, 0, self.epoch_num, 0, None)
        self.update_lr(self.last_lr, self.last_lr)
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses  = self.train_epoch()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')
                    print('actor loss:', a_losses[0].item())

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

            if should_exit:
                return self.last_mean_rewards, epoch_num

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, curr_frames):
        # do we need scaled time?
        frame = self.frame
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / update_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.writer.add_scalar('info/last_lr/frame', self.last_lr, frame)
        self.writer.add_scalar('info/last_lr/epoch_num', self.last_lr, epoch_num)

        self.algo_observer.after_print_stats(frame, epoch_num, total_time)