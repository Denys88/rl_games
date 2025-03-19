import os
import copy
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler
import torch.distributed as dist
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.common import schedulers


class CentralValueTrain(nn.Module):

    def __init__(
        self, state_shape, value_size, ppo_device, num_agents, horizon_length, num_actors,
        num_actions, seq_length, normalize_value, network, config, writter, max_epochs,
        multi_gpu, zero_rnn_on_done
    ):
        nn.Module.__init__(self)

        self.ppo_device = ppo_device
        self.mixed_precision = config.get('mixed_precision', False)

        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.num_agents = num_agents
        self.horizon_length = horizon_length
        self.num_actors = num_actors
        self.seq_length = seq_length
        self.normalize_value = normalize_value
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.value_size = value_size
        self.max_epochs = max_epochs
        self.multi_gpu = multi_gpu
        self.truncate_grads = config.get('truncate_grads', False)
        self.config = config
        self.normalize_input = config['normalize_input']
        self.zero_rnn_on_done = zero_rnn_on_done

        state_config = {
            'value_size': value_size,
            'input_shape': state_shape,
            'actions_num': num_actions,
            'num_agents': num_agents,
            'num_seqs': num_actors,
            'normalize_input': self.normalize_input,
            'normalize_value': self.normalize_value,
        }

        self.model = network.build(state_config)
        self.lr = float(config['learning_rate'])
        self.linear_lr = config.get('lr_schedule') == 'linear'

        # todo: support max frames as well
        if self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(self.lr,
                max_steps=self.max_epochs,
                apply_to_entropy=False,
                start_entropy_coef=0)
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.mini_epoch = config['mini_epochs']
        # Either minibatch_size_per_env or minibatch_size should be present in a config
        # if both are present, minibatch_size is used
        # otherwise minibatch_size_per_env is used to calculate minibatch_size
        if 'minibatch_size' not in self.config and 'minibatch_size_per_env' not in self.config:
            raise ValueError(
                "Configuration must include either 'minibatch_size' or 'minibatch_size_per_env'. "
                "Neither was found in the provided config."
            )

        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.num_minibatches = self.horizon_length * self.num_actors // self.minibatch_size
        self.clip_value = config['clip_value']

        self.writter = writter
        self.weight_decay = config.get('weight_decay', 0.0)
        self.model = self.model.to(self.ppo_device)  # Move explicitly to the device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            float(self.lr),
            eps=1e-08,
            weight_decay=self.weight_decay,
            fused=True
        )
        self.frame = 0
        self.epoch_num = 0
        self.running_mean_std = None
        self.grad_norm = config.get('grad_norm', 1)
        self.truncate_grads = config.get('truncate_grads', False)
        self.e_clip = config.get('e_clip', 0.2)
        self.truncate_grad = self.config.get('truncate_grads', False)

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None
        self.batch_size = self.horizon_length * self.num_actors

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_actors #* self.num_agents
            num_seqs = self.horizon_length // self.seq_length
            assert ((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32, device=self.ppo_device) for s in self.rnn_states]

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name

            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, True, self.is_rnn, self.ppo_device, self.seq_length)

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device_name)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_stats_weights(self, model_stats=False):
        state = {}
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        pass

    def update_dataset(self, batch_dict):
        value_preds = batch_dict['old_values']
        returns = batch_dict['returns']
        actions = batch_dict['actions']
        dones = batch_dict['dones']
        rnn_masks = batch_dict['rnn_masks']
        if self.num_agents > 1:
            res = self.update_multiagent_tensors(value_preds, returns, actions, dones)
            batch_dict['old_values'] = res[0]
            batch_dict['returns'] = res[1]
            batch_dict['actions'] = res[2]
            batch_dict['dones'] = res[3]

        if self.is_rnn:
            states = []
            for mb_s in self.mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))

            batch_dict['rnn_states'] = states
            # if self.num_agents > 1:
            #     rnn_masks = res[3]
            # batch_dict['rnn_masks'] = rnn_masks
        self.dataset.update_values_dict(batch_dict)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def pre_step_rnn(self, n):
        if not self.is_rnn:
            return
        if n % self.seq_length == 0:
            for s, mb_s in zip(self.rnn_states, self.mb_rnn_states):
                mb_s[n // self.seq_length, :, :, :] = s

    def post_step_rnn(self, all_done_indices, zero_rnn_on_done=True):
        if not self.is_rnn:
            return
        if not self.zero_rnn_on_done:
            return
        all_done_indices = all_done_indices[::self.num_agents] // self.num_agents
        for s in self.rnn_states:
            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

    def forward(self, input_dict):
        return self.model(input_dict)

    def get_value(self, input_dict):
        self.eval()
        obs_batch = input_dict['states']
        actions = input_dict.get('actions', None)

        obs_batch = self._preproc_obs(obs_batch)
        res_dict = self.forward({'obs': obs_batch, 'actions': actions,
                                'rnn_states': self.rnn_states,
                                'is_train': False})
        value, self.rnn_states = res_dict['values'], res_dict['rnn_states']
        if self.num_agents > 1:
            value = value.repeat(1, self.num_agents)
            value = value.view(value.size()[0]*self.num_agents, -1)

        return value

    def train_critic(self, input_dict):
        self.train()
        loss = self.calc_gradients(input_dict)
        return loss.item()

    def update_multiagent_tensors(self, value_preds, returns, actions, dones):
        batch_size = self.batch_size
        ma_batch_size = self.num_actors * self.num_agents * self.horizon_length
        value_preds = value_preds.view(self.num_actors, self.num_agents, self.horizon_length, self.value_size).transpose(0, 1)
        returns = returns.view(self.num_actors, self.num_agents, self.horizon_length, self.value_size).transpose(0, 1)
        value_preds = value_preds.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        returns = returns.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        dones = dones.contiguous().view(ma_batch_size, self.value_size)[:batch_size]

        return value_preds, returns, actions, dones

    def train_net(self):
        self.train()
        loss = 0
        for _ in range(self.mini_epoch):
            if self.config.get('freeze_critic', False):
                break
            for data in self.dataset:
                loss += self.train_critic(data)

            if self.normalize_input:
                # don't need to update statistics more than one miniepoch
                self.model.running_mean_std.eval()

        avg_loss = loss / (self.mini_epoch * self.num_minibatches)

        self.epoch_num += 1
        self.lr, _ = self.scheduler.update(self.lr, 0, self.epoch_num, 0, 0)
        self.update_lr(self.lr)
        self.frame += self.batch_size
        if self.writter is not None:
            self.writter.add_scalar('losses/cval_loss', avg_loss, self.frame)
            self.writter.add_scalar('info/cval_lr', self.lr, self.frame)
        return avg_loss

    # Unfortunately mode='max-autotune' does not work with torch.compile() here
    @torch.compile()
    def calc_loss(self, value_preds_batch, values, returns_batch, rnn_masks_batch):
        loss = common_losses.critic_loss(
            self.model,
            value_preds_batch,
            values,
            self.e_clip,
            returns_batch,
            self.clip_value
        )
        losses, _ = torch_ext.apply_masks([loss], rnn_masks_batch)
        return losses[0]

    def calc_gradients(self, batch):
        obs_batch = self._preproc_obs(batch['obs'])
        value_preds_batch = batch['old_values']
        returns_batch = batch['returns']
        actions_batch = batch['actions']
        dones_batch = batch['dones']
        rnn_masks_batch = batch.get('rnn_masks')

        batch_dict = {
            'obs': obs_batch,
            'actions': actions_batch,
            'seq_length': self.seq_length,
            'dones': dones_batch
        }
        if self.is_rnn:
            batch_dict['rnn_states'] = batch['rnn_states']

        res_dict = self.model(batch_dict)

        values = res_dict['values']
        loss = self.calc_loss(value_preds_batch, values, returns_batch, rnn_masks_batch)

        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        loss.backward()

        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset: offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()

        return loss

    def train(self, input_dict):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.mixed_precision):
            values = self.model(input_dict)['values']
            loss = (values - input_dict['returns']).pow(2).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
