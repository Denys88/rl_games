import os
import time
import numpy as np
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch import model_builder, sac_helper, torch_ext
from rl_games.common import vecenv, schedulers, experience
from rl_games.common.a2c_common import merge_rank_stats, print_statistics
from rl_games.interfaces.base_algorithm import BaseAlgorithm


class SACAgent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)
        self._init_multi_gpu(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.critic_tau = float(self.config.get("critic_tau", 0.005))  # target-net Polyak rate
        self.gamma = self.config.get("gamma", 0.99)
        self.gamma_tensor = torch.tensor(self.gamma, device=self._device, dtype=torch.float32)
        self.num_steps_per_episode = self.config.get("num_steps_per_episode", 1)
        # utd_ratio: gradient updates per env FRAME (one env step = num_actors
        # frames). Overrides the legacy absolute num_updates_per_step, which
        # silently diluted UTD as num_actors grew.
        utd_ratio = self.config.get("utd_ratio", None)
        if utd_ratio is not None:
            self.num_updates_per_step = max(1, round(utd_ratio * self.num_actors))
        else:
            self.num_updates_per_step = self.config.get("num_updates_per_step", 1)
        effective_utd = self.num_updates_per_step / self.num_actors
        print(f"SAC effective UTD ratio: {effective_utd:.3f} "
              f"({self.num_updates_per_step} updates per step of {self.num_actors} frames)")
        if effective_utd < 0.5:
            print(f"WARNING: effective UTD {effective_utd:.3f} < 0.5 — the critic will be "
                  f"data-starved; set utd_ratio: 1.0 unless this is intentional")
        # Warmup: num_warmup_frames (total env frames) takes priority over num_warmup_steps (epochs)
        num_warmup_frames = self.config.get("num_warmup_frames", None)
        if num_warmup_frames is not None:
            frames_per_epoch = self.num_actors * self.num_steps_per_episode
            self.num_warmup_steps = int(np.ceil(num_warmup_frames / frames_per_epoch))
        else:
            self.num_warmup_steps = self.config.get("num_warmup_steps", 1000)
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.save_replay_buffer = config.get("save_replay_buffer", False)
        self.normalize_input = config.get("normalize_input", False)
        self.amp_dtype = torch_ext.resolve_mixed_precision(config.get("mixed_precision", False), self._device)
        self.enable_mixed_precision = self.amp_dtype is not None
        self.critic_scaler = torch_ext.grad_scaler(self.amp_dtype)
        self.actor_scaler = torch_ext.grad_scaler(self.amp_dtype)

        # Update the actor once every policy_frequency critic updates
        self.policy_frequency = config.get("policy_frequency", 2)
        self.update_counter = 0

        # Gradient clipping for stability (0 = disabled)
        self.critic_grad_clip = config.get("critic_grad_clip", 5.0)

        # When True, timeouts are stored as truncations (bootstrap V at next_obs)
        # rather than hard terminations. Default matches PPO.
        self.value_bootstrap = config.get('value_bootstrap', True)

        # Autoreset semantics declared by the vec env (envpool/gymnasium = 'next_step',
        # everything else defaults to 'same_step'). Drives replay-write logic in play_steps.
        self.autoreset_mode = (self.env_info or {}).get('autoreset_mode', 'same_step')
        if self.autoreset_mode == 'next_step' and self.num_agents > 1:
            raise ValueError("SAC next_step autoreset handling does not support multi-agent envs "
                             "(num_agents > 1): per-env done tracking and live-row striding would misalign.")


        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        # n-step returns, folded at insertion; the bootstrap discount is gamma^k per row
        self.n_step = int(config.get('n_step', 1))
        if self.n_step < 1:
            raise ValueError(f'n_step must be >= 1, got {self.n_step}')

        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        # action groups (e.g. one per hand): one actor tower, Q twin pair, reward head and temperature each
        self.action_groups = sac_helper.resolve_action_groups(
            config.get('action_groups') or self.env_info.get('action_groups'), self.actions_num)
        self.num_heads = len(self.action_groups) if self.action_groups else 1

        # Store proper action bounds per dimension
        self.action_low = torch.tensor(action_space.low, device=self._device, dtype=torch.float32)
        self.action_high = torch.tensor(action_space.high, device=self._device, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        # Change-of-variables constant for the affine rescale [-1,1] -> env bounds
        # (see _env_log_prob). Zero for unit bounds; matches reference SAC implementations.
        self.log_action_scale_sum = torch.log(self.action_scale.clamp_min(1e-8)).sum()
        # Not read anywhere in the package (players use the per-dim rescale above);
        # kept for backward compatibility with external code.
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'normalize_input': self.normalize_input,
            'action_groups': self.action_groups,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)
        sac_network = self.model.sac_network
        # an ensemble critic (flash_sac network) takes the per-group update path
        self.ensemble_critic = getattr(sac_network, 'ensemble_critic', False)
        if self.num_heads > 1 and not self.ensemble_critic:
            raise ValueError('action_groups with more than one group need an ensemble critic network (flash_sac)')
        self._post_actor_step = getattr(sac_network, 'post_actor_step', None)
        self._post_critic_step = getattr(sac_network, 'post_critic_step', None)

        if self.ensemble_critic:
            groups = self.action_groups or [list(range(self.actions_num))]
            masks = sac_helper.group_masks(groups, self.actions_num).to(self._device)
            self.group_masks = masks.float()
            log_scale = torch.log(self.action_scale.clamp_min(1e-8))
            self.group_log_scale = (self.group_masks * log_scale).sum(-1)
            self.log_alpha = torch.full((self.num_heads,), float(np.log(self.init_alpha)), device=self._device)
        else:
            self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_lr = float(self.config['actor_lr'])
        self.critic_lr = float(self.config['critic_lr'])
        self.alpha_lr = float(self.config['alpha_lr'])
        self.actor_optimizer = optim.Adam(self.model.sac_network.actor.parameters(),
                                        lr=self.actor_lr,
                                        betas=self.config.get("actor_betas", [0.9, 0.999]),
                                        fused=True)

        self.critic_optimizer = optim.Adam(self.model.sac_network.critic.parameters(),
                                        lr=self.critic_lr,
                                        betas=self.config.get("critic_betas", [0.9, 0.999]),
                                        fused=True)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                        lr=self.alpha_lr,
                                        betas=self.config.get("alphas_betas", [0.9, 0.999]),
                                        fused=True)

        self._init_lr_schedule(config)
        self.replay_buffer = self._build_replay_buffer(config)
        self._init_grad_sync(config)

        # update_cuda_graph: each update (critic step, actor + temperature step, target EMA) is captured
        # once per branch in a CUDA graph and replayed; sampling and the LR schedule stay on the host
        self.graph_step = None
        if config.get('update_cuda_graph', False):
            if not self.ensemble_critic or torch.device(self._device).type != 'cuda':
                raise ValueError('update_cuda_graph needs the flash_sac network on a CUDA device')
            compile_cfg = config.get('torch_compile', False)
            mode = compile_cfg.get('mode') if isinstance(compile_cfg, dict) else compile_cfg
            if mode is True or mode in ('reduce-overhead', 'max-autotune'):
                raise ValueError("update_cuda_graph captures the whole update; torch_compile must not add CUDA "
                                 "graphs of its own: use 'default' or 'max-autotune-no-cudagraphs'")
            self._make_capturable()
            self.graph_step = sac_helper.CudaGraphStep(self._update_step,
                                                       warmup=int(config.get('update_cuda_graph_warmup', 3)))

        # Target entropy per group: -coef * dims, or that of an isotropic Gaussian with
        # std target_entropy_sigma per normalized action dim (0.5 n ln(2 pi e sigma^2)).
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        sigma = config.get('target_entropy_sigma', None)
        dims = [len(ids) for ids in self.action_groups] if self.action_groups else [self.actions_num]
        if sigma is not None:
            entropies = [sac_helper.gaussian_entropy(n, float(sigma)) for n in dims]
        else:
            entropies = [self.target_entropy_coef * -n for n in dims]
        if self.ensemble_critic:
            self.target_entropy = torch.tensor(entropies, device=self._device)
        else:
            self.target_entropy = entropies[0]
        print("Target entropy", self.target_entropy)

        # per-head reward scale from the spread of the running discounted return
        self.reward_scaler = None
        if config.get('normalize_reward', False):
            self.reward_scaler = sac_helper.ReturnScaler(
                self.num_heads, self.gamma, float(config.get('normalized_return_max', 5.0)), self._device,
                all_reduce=dist.all_reduce if self.multi_gpu else None)

        # exploration noise held over 1..held_noise_max_steps policy steps (truncated zeta)
        self.held_noise = None
        if config.get('held_noise', False):
            self.held_noise = sac_helper.HeldNoise(float(config.get('held_noise_zeta_mu', 2.0)),
                                                   int(config.get('held_noise_max_steps', 16)))
        self.actor_steps = 0
        self._head_stats = {}

        self.algo_observer = config['features']['observer']


        # ────────────────────────────────────
        # Summary writer
        # ────────────────────────────────────
        if self.global_rank == 0:
            self.writer = SummaryWriter(self.summaries_dir)
        else:
            class _DummyWriter:
                def add_scalar(self, *_, **__): pass
                def close(self): pass
            self.writer = _DummyWriter()

    def _init_multi_gpu(self, config):
        """multi_gpu (torchrun): one rank per GPU, each with its own envs and replay. Every update samples
        a local batch; the gradients and the batch-norm running statistics are averaged over the ranks
        before the optimizer steps, so all ranks hold the same networks (data-parallel SAC). The batch,
        UTD ratio and replay size are per rank; frames and max_frames count all ranks."""
        self.multi_gpu = config.get('multi_gpu', False)
        self.global_rank = int(os.getenv('RANK', '0'))
        self.local_rank, self.world_size = 0, 1
        if not self.multi_gpu:
            return
        self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        backend = config.get('multi_gpu_backend', 'nccl')
        if backend == 'nccl':  # this rank's GPU before any CUDA work or the process group
            device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.local_rank)
            torch_ext.check_nccl(device)
            config['device'] = device
            env_config = config.get('env_config') or {}
            if str(env_config.get('device', '')).startswith('cuda'):
                env_config['device'] = device
        if not dist.is_initialized():
            dist.init_process_group(backend, rank=self.global_rank, world_size=self.world_size)
        if self.global_rank != 0:
            config['print_stats'] = False
        self._can_sample_all = False

    def _setup_multi_gpu(self):
        """Start every rank from rank 0's networks and temperature."""
        if not self.multi_gpu:
            return
        with torch.no_grad():
            for t in list(self.model.state_dict().values()) + [self.log_alpha.data]:
                if torch.is_tensor(t):
                    dist.broadcast(t, 0)

    def _allreduce_mean_(self, tensors):
        """Average the tensors over the ranks in place, through one flat all-reduce."""
        if not self.multi_gpu or not tensors:
            return
        flat = torch.cat([t.reshape(-1) for t in tensors])
        dist.all_reduce(flat)
        flat.div_(self.world_size)
        torch._foreach_copy_(tensors, [f.view_as(t) for f, t in zip(flat.split([t.numel() for t in tensors]), tensors)])

    def _init_grad_sync(self, config):
        """multi_gpu_overlap (default True): all-reduce each gradient from a backward hook, overlapping the
        traffic with the rest of backward; else one flat all-reduce after backward. multi_gpu_grad_dtype:
        precision of the gradient traffic (float32 default, bfloat16 halves it)."""
        self._critic_reducer = self._actor_reducer = None
        if not self.multi_gpu or not config.get('multi_gpu_overlap', True):
            return
        dtype = config.get('multi_gpu_grad_dtype', None)
        dtype = None if dtype in (None, 'float32') else getattr(torch, dtype)
        net = self.model.sac_network
        self._critic_reducer = sac_helper.OverlappedGradAllReduce(net.critic.parameters(), self.world_size, dtype)
        actor_params = list(net.actor.parameters()) + ([self.log_alpha] if self.learnable_temperature else [])
        self._actor_reducer = sac_helper.OverlappedGradAllReduce(actor_params, self.world_size, dtype)

    @staticmethod
    def _float_buffers(modules):
        return [b for m in modules for b in m.buffers() if b.is_floating_point()]

    def _sync_critic_step(self):
        """Mean critic gradients and batch-norm statistics (online and target) over the ranks."""
        net = self.model.sac_network
        buffers = self._float_buffers([net.critic, net.critic_target])
        if self._critic_reducer is not None:
            self._critic_reducer.finish()
            self._allreduce_mean_(buffers)
        else:
            grads = [p.grad for p in net.critic.parameters() if p.grad is not None]
            self._allreduce_mean_(grads + buffers)

    def _sync_actor_step(self):
        net = self.model.sac_network
        buffers = self._float_buffers([net.actor])
        if self._actor_reducer is not None:
            self._actor_reducer.finish()
            self._allreduce_mean_(buffers)
        else:
            params = list(net.actor.parameters()) + ([self.log_alpha] if self.learnable_temperature else [])
            self._allreduce_mean_([p.grad for p in params if p.grad is not None] + buffers)

    def _all_ranks(self, flag):
        """True when flag holds on every rank."""
        if not self.multi_gpu:
            return flag
        t = torch.tensor([1.0 if flag else 0.0], device=self._device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def _make_capturable(self):
        for opt in (self.actor_optimizer, self.critic_optimizer, self.log_alpha_optimizer):
            sac_helper.make_capturable(opt, self._device)

    def _init_lr_schedule(self, config):
        """lr_schedule: cosine = linear warm-up over lr_warmup_steps updates, then cosine to
        lr_end_factor * lr at lr_decay_steps (default: the run's update count). Each schedule is
        indexed by the update count, or with lr_per_optimizer_step by its own optimizer's steps
        (the actor and temperature then step every policy_frequency updates)."""
        self.lr_schedule = config.get('lr_schedule', None)
        if self.lr_schedule not in (None, 'constant', 'cosine'):
            raise ValueError(f"SAC lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}")
        self.lr_warmup_steps = int(config.get('lr_warmup_steps', 0))
        self.lr_end_factor = float(config.get('lr_end_factor', 0.5))
        self.lr_per_optimizer_step = config.get('lr_per_optimizer_step', False)
        decay = config.get('lr_decay_steps', None)
        if decay is None and self.lr_schedule == 'cosine':
            if self.max_frames > 0:  # max_frames counts every rank
                env_steps = self.max_frames / (self.num_actors * self.num_agents * self.world_size)
            elif self.max_epochs > 0:
                env_steps = self.max_epochs * self.num_steps_per_episode
            else:
                raise ValueError('lr_schedule cosine needs lr_decay_steps, max_frames or max_epochs')
            decay = int(env_steps * self.num_updates_per_step)
        self.lr_decay_steps = int(decay or 0)

    def _set_lr(self, optimizer, base_lr, step):
        if self.lr_schedule != 'cosine':
            return
        lr = base_lr * schedulers.cosine_lr_factor(step, self.lr_warmup_steps, self.lr_decay_steps,
                                                   self.lr_end_factor)
        for group in optimizer.param_groups:
            if torch.is_tensor(group['lr']):
                group['lr'].fill_(lr)  # device LR read by a captured step
            else:
                group['lr'] = lr

    def _build_replay_buffer(self, config):
        """VectorizedReplayBuffer (default), or with replay_memory_efficient the ring that reads next obs
        by row offset. replay_obs_dtype sets the obs storage dtype, replay_storage 'host' keeps rows in
        page-locked host memory read by the GPU."""
        obs_shape = self.env_info['observation_space'].shape
        action_shape = self.env_info['action_space'].shape
        obs_dtype = getattr(torch, config.get('replay_obs_dtype', 'float32'))
        storage = config.get('replay_storage', 'device')
        # the config's compactor, else one the env reports (env_info['replay_obs_compactor'], an object)
        compactor = config.get('replay_obs_compactor', None) or (self.env_info or {}).get('replay_obs_compactor')
        if isinstance(compactor, str):
            # 'module:factory', called with env_info
            module, attr = compactor.split(':')
            compactor = getattr(__import__(module, fromlist=[attr]), attr)(self.env_info)
        self.obs_compactor = compactor
        if config.get('replay_memory_efficient', False):
            if self.autoreset_mode == 'next_step':
                raise ValueError('replay_memory_efficient needs same_step autoreset envs')
            if self.save_replay_buffer:
                raise ValueError('save_replay_buffer is not supported with replay_memory_efficient')
            return experience.MemoryEfficientReplayBuffer(
                obs_shape, action_shape, self.replay_buffer_size, self._device,
                num_envs=self.num_actors * self.num_agents, num_heads=self.num_heads, n_step=self.n_step,
                gamma=self.gamma, obs_dtype=obs_dtype, storage=storage, compactor=compactor)
        if compactor is not None:
            raise ValueError('replay_obs_compactor needs replay_memory_efficient')
        return experience.VectorizedReplayBuffer(
            obs_shape, action_shape, self.replay_buffer_size, self._device, num_heads=self.num_heads,
            n_step=self.n_step, gamma=self.gamma, obs_dtype=obs_dtype, storage=storage)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)

        self._device = config.get('device', 'cuda:0')

        # For Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        # self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        # MSELoss for the critic TD error - standard for SAC
        self.c_loss = nn.MSELoss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        # Optional user-supplied stop callback: callable(algo) -> bool.
        # See README "Custom stop callback" section.
        self.stop_fn = config.get('stop_fn', None)
        if self.stop_fn is not None and not callable(self.stop_fn):
            raise ValueError(f"'stop_fn' must be callable, got {type(self.stop_fn).__name__}")

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        # Min alpha in EXP space (not log space!) for comparison with self.alpha
        self.min_alpha = torch.tensor(float(config.get('min_alpha', 0.01))).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -float('inf')
        self.play_time = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)
        # Tracks which envs ended an episode on the PREVIOUS step (next_step autoreset:
        # their next row is the reset-step garbage row and must be skipped).
        self._prev_dones = torch.zeros(batch_size, dtype=torch.bool, device=self._device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'actor': self.model.sac_network.actor.state_dict(),
         'critic': self.model.sac_network.critic.state_dict(),
         'critic_target': self.model.sac_network.critic_target.state_dict()}
        if self.normalize_input:
            state['running_mean_std'] = self.model.running_mean_std.state_dict()
        return state

    def save(self, fn):
        if self.global_rank != 0:  # every rank holds the same networks
            return
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()
        state['log_alpha'] = self.log_alpha.detach().cpu()
        state['update_counter'] = self.update_counter
        state['actor_steps'] = self.actor_steps
        if self.reward_scaler is not None:
            state['reward_scaler'] = self.reward_scaler.state_dict()
        if self.critic_scaler.is_enabled():
            state['critic_scaler'] = self.critic_scaler.state_dict()
            state['actor_scaler'] = self.actor_scaler.state_dict()

        # set_full_state_weights restores last_mean_rewards (and frame/epoch above);
        # saving it keeps the best-ever checkpoint watermark across restarts.
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            state['env_state'] = self.vec_env.get_env_state()

        if self.save_replay_buffer:
            state['replay_buffer'] = self.replay_buffer.state_dict()

        # If the config declares a `capability_manifest`, store it in the
        # checkpoint as-is so the metadata travels with the policy.
        # rl_games never interprets its contents.
        capability_manifest = self.config.get('capability_manifest')
        if capability_manifest is not None:
            state['capability_manifest'] = capability_manifest

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

        if 'log_alpha' in weights:
            self.log_alpha.data.copy_(weights['log_alpha'].to(self._device))
        self.update_counter = weights.get('update_counter', self.update_counter)
        self.actor_steps = weights.get('actor_steps', self.actor_steps)
        if self.reward_scaler is not None and 'reward_scaler' in weights:
            self.reward_scaler.load_state_dict(weights['reward_scaler'])
        if 'critic_scaler' in weights and self.critic_scaler.is_enabled():
            self.critic_scaler.load_state_dict(weights['critic_scaler'])
            self.actor_scaler.load_state_dict(weights['actor_scaler'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -float('inf'))

        if 'replay_buffer' in weights:
            self.replay_buffer.load_state_dict(weights['replay_buffer'])

        if self.graph_step is not None:
            # new optimizer state tensors: recapture
            self._make_capturable()
            self.graph_step.reset()

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

        # Adopt the checkpoint's capability_manifest unless the current config
        # already declares one -- an explicit config value takes precedence.
        if 'capability_manifest' in weights:
            declared = self.config.get('capability_manifest')
            if declared is not None and declared != weights['capability_manifest']:
                print('WARNING: config capability_manifest differs from the '
                      'checkpoint one; keeping the config value')
            else:
                self.config['capability_manifest'] = weights['capability_manifest']

    def restore(self, fn, set_epoch=True):
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Checkpoint file not found: {fn}")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        raise NotImplementedError("Masked action values are not supported in SAC agent")

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            # Keep the normalizer in eval mode during updates: only fresh rollout
            # frames (via _update_obs_stats) may mutate its stats, never replays.
            self.model.running_mean_std.eval()

    def _update_obs_stats(self, obs):
        # Ingest one fresh rollout frame into the normalizer's running stats.
        if not self.normalize_input:
            return
        with torch.no_grad():
            rms = self.model.running_mean_std
            rms.train()
            rms(obs['obs'] if isinstance(obs, dict) else obs)
            rms.eval()

    def rescale_actions(self, actions):
        """Rescale actions from [-1, 1] to actual action space bounds."""
        rescaled = actions * self.action_scale + self.action_bias
        # Clamp to handle floating point errors
        return rescaled.clamp(self.action_low, self.action_high)

    def _env_log_prob(self, dist, action):
        """Log-prob of the env-space action via change of variables.

        `dist` is over normalized [-1, 1] actions and `action` is a normalized
        sample; the env executes a_env = action_scale * action + action_bias, so
        log pi_env(a_env) = log pi_norm(action) - sum(log(action_scale)).
        Keeps entropy (and the alpha objective) in the ENV action space.
        """
        return dist.log_prob(action).sum(-1, keepdim=True) - self.log_action_scale_sum

    def update_critic(self, obs, action, reward, next_obs, not_done, step, discount=None):
        # discount: per-row gamma^k of n-step windows, else gamma
        discount = self.gamma_tensor if discount is None else discount
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = self._env_log_prob(dist, next_action)
            # Rescale next_action for critic (as it outputs [-1, 1])
            next_action_rescaled = self.rescale_actions(next_action)

            target_Q1, target_Q2 = self.model.critic_target(
                next_obs.float(),
                next_action_rescaled.float()
            )
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_V = target_V.detach()

        # get current Q estimates and build target_Q inside autocast
        with torch_ext.autocast(self.amp_dtype):
            current_Q1, current_Q2 = self.model.critic(obs, action)

            # Build target_Q inside autocast to match dtypes
            target_Q = reward + (not_done * discount * target_V)

            critic1_loss = self.c_loss(current_Q1, target_Q)
            critic2_loss = self.c_loss(current_Q2, target_Q)
            critic_loss = 0.5 * (critic1_loss + critic2_loss)

        self._critic_step(critic_loss)
        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def _critic_step(self, critic_loss):
        self._set_lr(self.critic_optimizer, self.critic_lr, self.update_counter)
        self._critic_opt_step(critic_loss)

    def _critic_opt_step(self, critic_loss):
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_scaler.scale(critic_loss).backward()
        if self.multi_gpu:
            self._sync_critic_step()
        if self.critic_grad_clip > 0:
            self.critic_scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.sac_network.critic.parameters(), self.critic_grad_clip)
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.update()
        if self._post_critic_step is not None:
            self._post_critic_step()

    def update_actor_and_alpha(self, obs, step):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        with torch_ext.autocast(self.amp_dtype):
            dist = self.model.actor(obs)
            action = dist.rsample()
            # Rescale actions for critic evaluation
            action_rescaled = self.rescale_actions(action)
            log_prob = self._env_log_prob(dist, action)
            entropy = -log_prob.mean()
            actor_Q1, actor_Q2 = self.model.critic(obs, action_rescaled)
            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q).mean()

            if self.learnable_temperature:
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                total_loss = actor_loss + alpha_loss
            else:
                alpha_loss = None
                total_loss = actor_loss

        self._actor_step(total_loss)

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss

    def _actor_step(self, total_loss):
        step = self.actor_steps if self.lr_per_optimizer_step else self.update_counter - 1
        self._set_lr(self.actor_optimizer, self.actor_lr, step)
        self._set_lr(self.log_alpha_optimizer, self.alpha_lr, step)
        self._actor_opt_step(total_loss)
        self.actor_steps += 1

    def _actor_opt_step(self, total_loss):
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)

        self.actor_scaler.scale(total_loss).backward()
        if self.multi_gpu:
            self._sync_actor_step()

        self.actor_scaler.step(self.actor_optimizer)
        if self.learnable_temperature:
            self.actor_scaler.step(self.log_alpha_optimizer)
        self.actor_scaler.update()
        if self._post_actor_step is not None:
            self._post_actor_step()

    def _group_log_prob(self, dist, action):
        """Env-space log-prob per action group, (B, H)."""
        log_prob = dist.log_prob(action)
        return (log_prob.unsqueeze(1) * self.group_masks).sum(-1) - self.group_log_scale

    def _add_head_stat(self, name, value):
        self._head_stats.setdefault(name, []).append(value.detach())

    def update_critic_heads(self, obs, action, reward, next_obs, done, discount):
        """Critic step of an ensemble critic with one twin pair per action group (reward column).

        Target per group h: r_h + discount * (min-twin Q_h(s', a') - alpha_h log pi_h(a')) for a scalar
        head; a categorical head takes the bins of the min-Q twin, shifted the same way and projected
        back onto the bins. Online and target critics run on [obs; next_obs] with [a; a'] so batch
        norm sees both halves (CrossQ); only the first half enters the loss. a' comes from the actor's
        running statistics.
        """
        critic_loss, losses = self._critic_heads_loss(obs, action, reward, next_obs, done, discount)
        self._critic_step(critic_loss)
        if self.num_heads > 1:
            self._add_head_stat('losses/c_loss_head', losses.mean(dim=0))
        return critic_loss.detach(), losses[0].mean().detach(), losses[1].mean().detach()

    def _critic_heads_loss(self, obs, action, reward, next_obs, done, discount):
        """Critic loss of update_critic_heads and its (2, H) per-twin, per-head terms."""
        net = self.model.sac_network
        n, h = obs.shape[0], self.num_heads
        with torch.no_grad():
            net.actor.eval()
            dist = net.actor(next_obs)
            next_action = dist.rsample()
            bonus = (self.alpha * self._group_log_prob(dist, next_action)).t()  # (H, B)
            net.actor.train()
        obs_all = torch.cat([obs, next_obs])
        act_all = torch.cat([action, self.rescale_actions(next_action)])
        terminated = done.view(1, n).expand(h, n)
        with torch_ext.autocast(self.amp_dtype):
            with torch.no_grad():
                q_t, log_p_t = net.critic_target(obs_all, act_all)
            q, log_p = net.critic(obs_all, act_all)
            if log_p is None:
                disc = self.gamma_tensor if discount is None else discount.view(1, n)
                target_v = q_t[:, n:].reshape(2, h, n).amin(dim=0) - bonus
                target_q = reward.t() + (~terminated) * disc * target_v
                losses = (q[:, :n].reshape(2, h, n) - target_q.detach()).pow(2).mean(dim=2)  # (2, H)
            else:
                bins = log_p.shape[-1]
                disc = self.gamma if discount is None else discount.view(1, n).expand(h, n).reshape(-1)
                with torch.no_grad():
                    probs = sac_helper.min_twin_log_probs(
                        q_t[:, n:].reshape(2, h * n), log_p_t[:, n:].reshape(2, h * n, bins)).exp()
                    target = sac_helper.project_categorical(
                        probs, reward.t().reshape(-1), terminated.reshape(-1), bonus.reshape(-1), disc,
                        net.v_min, net.v_max).view(1, h, n, bins)
                losses = -(target * log_p[:, :n].reshape(2, h, n, bins)).sum(-1).mean(dim=2)  # (2, H)
            critic_loss = losses.mean()
        return critic_loss, losses.detach()

    def update_actor_and_alpha_heads(self, obs, next_obs):
        """Actor and temperature step with one loss and temperature per action group.

        The actor runs on [obs; next_obs] so its batch norm sees what the critic update sees; only the
        first half enters the loss. Q pair h sees the action with only group h's dims attached to the
        graph, so loss h trains group h's tower. The online critic uses its running statistics.
        """
        total_loss, actor_loss, losses, entropy, alpha_loss = self._actor_heads_loss(obs, next_obs)
        self._actor_step(total_loss)
        if self.num_heads > 1:
            self._add_head_stat('losses/a_loss_head', losses)
            self._add_head_stat('losses/entropy_head', entropy)
            self._add_head_stat('info/alpha_head', self.alpha)
        return actor_loss.detach(), entropy.mean(), self.alpha.detach(), alpha_loss

    def _actor_heads_loss(self, obs, next_obs):
        """Losses of update_actor_and_alpha_heads: total, actor, per-group actor (H,), entropy (H,), alpha."""
        net = self.model.sac_network
        n, h = obs.shape[0], self.num_heads
        for p in net.critic.parameters():
            p.requires_grad = False
        net.critic.eval()

        with torch_ext.autocast(self.amp_dtype):
            dist = net.actor(torch.cat([obs, next_obs]))
            action = dist.rsample()
            log_prob = self._group_log_prob(dist, action)[:n]  # (B, H)
            q, _ = net.critic(obs, net.gate_actions(self.rescale_actions(action[:n])))
            actor_q = q.reshape(2, h, n).amin(dim=0)  # (H, B)
            alpha = torch.max(self.alpha.detach(), self.min_alpha)
            losses = (alpha.unsqueeze(-1) * log_prob.t() - actor_q).mean(dim=-1)  # (H,)
            actor_loss = losses.sum()
            entropy = -log_prob.detach().float().mean(dim=0)  # (H,)

            if self.learnable_temperature:
                alpha_loss = (self.alpha * (entropy - self.target_entropy)).sum()
                total_loss = actor_loss + alpha_loss
            else:
                alpha_loss = None
                total_loss = actor_loss

        # the graph above holds no critic parameter, so backward later leaves the critic grads alone
        net.critic.train()
        for p in net.critic.parameters():
            p.requires_grad = True
        return total_loss, actor_loss.detach(), losses.detach(), entropy, alpha_loss

    def soft_update_params(self, net, target_net, tau):
        with torch.no_grad():
            # one multi-tensor kernel; the same lerp as a per-tensor lerp_
            torch._foreach_lerp_([p.data for p in target_net.parameters()], [p.data for p in net.parameters()], tau)

    def update_target_weights(self):
        tau = self.critic_tau
        self.soft_update_params(
            self.model.sac_network.critic,
            self.model.sac_network.critic_target,
            tau
        )

    def update(self, step):
        # The truncated column is not needed for the TD target: truncated rows store
        # done=False with next_obs = the true final obs, so the bootstrap is already
        # correct. It is kept for diagnostics.
        batch = self.replay_buffer.sample_batch(self.batch_size)
        obs, action, reward, next_obs, done = (batch['obs'], batch['action'], batch['reward'],
                                               batch['next_obs'], batch['done'])
        if self.reward_scaler is not None:
            reward = self.reward_scaler.scale(reward)
        not_done = ~done

        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)

        if self.graph_step is not None:
            return self._graphed_update(obs, action, reward, next_obs, done, batch['discount'])

        if self.ensemble_critic:
            critic_loss, critic1_loss, critic2_loss = self.update_critic_heads(
                obs, action, reward, next_obs, done, batch['discount'])
        else:
            critic_loss, critic1_loss, critic2_loss = self.update_critic(
                obs, action, reward, next_obs, not_done, step, batch['discount'])
        self.update_counter += 1

        # Delayed actor/alpha update (every policy_frequency critic updates).
        if self.update_counter % self.policy_frequency == 0:
            if self.ensemble_critic:
                actor_loss_info = self.update_actor_and_alpha_heads(obs, next_obs)
            else:
                actor_loss_info = self.update_actor_and_alpha(obs, step)
        else:
            # No actor update: return None so skipped steps don't dilute logged means.
            actor_loss_info = None

        self.update_target_weights()
        return actor_loss_info, critic1_loss, critic2_loss

    def _update_step(self, batch, actor_step):
        """Device work of one update_cuda_graph update (critic step, actor and temperature step when
        actor_step, target EMA) on the batch tensors; returns the stats as tensors. The LR schedule
        and the counters run on the host around it (they would freeze in a captured graph)."""
        critic_loss, losses = self._critic_heads_loss(batch['obs'], batch['action'], batch['reward'],
                                                      batch['next_obs'], batch['done'], batch['discount'])
        self._critic_opt_step(critic_loss)
        out = {'critic_loss': critic_loss.detach(), 'critic_losses': losses}
        if actor_step:
            total_loss, actor_loss, a_losses, entropy, alpha_loss = self._actor_heads_loss(batch['obs'],
                                                                                          batch['next_obs'])
            self._actor_opt_step(total_loss)
            out.update(actor_loss=actor_loss, actor_losses=a_losses, entropy=entropy, alpha=self.alpha.detach())
            if alpha_loss is not None:
                out['alpha_loss'] = alpha_loss.detach()
        self.update_target_weights()
        return out

    def _graphed_update(self, obs, action, reward, next_obs, done, discount):
        actor_step = (self.update_counter + 1) % self.policy_frequency == 0
        self._set_lr(self.critic_optimizer, self.critic_lr, self.update_counter)
        if actor_step:
            step = self.actor_steps if self.lr_per_optimizer_step else self.update_counter
            self._set_lr(self.actor_optimizer, self.actor_lr, step)
            self._set_lr(self.log_alpha_optimizer, self.alpha_lr, step)
        batch = {'obs': obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'done': done,
                 'discount': discount}
        out = self.graph_step(batch, actor_step)
        self.update_counter += 1
        losses = out['critic_losses']
        if self.num_heads > 1:
            self._add_head_stat('losses/c_loss_head', losses.mean(dim=0))
        actor_info = None
        if actor_step:
            self.actor_steps += 1
            if self.num_heads > 1:
                self._add_head_stat('losses/a_loss_head', out['actor_losses'])
                self._add_head_stat('losses/entropy_head', out['entropy'])
                self._add_head_stat('info/alpha_head', out['alpha'])
            actor_info = (out['actor_loss'], out['entropy'].mean(), out['alpha'], out.get('alpha_loss'))
        return actor_info, losses[0].mean(), losses[1].mean()

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs)

        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:
            upd_obs = {'obs': upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device), torch.from_numpy(rewards).to(self._device, dtype=torch.float32), torch.from_numpy(dones).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        # Count the initial reset obs as a fresh frame.
        self._update_obs_stats(obs)
        # Full reset: no env is in a post-episode reset step. Sized
        # num_agents * num_actors to match the dones/current_rewards rows.
        self._prev_dones = torch.zeros(self.num_agents * self.num_actors,
                                       dtype=torch.bool, device=self._device)

        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)

        if sample and self.held_noise is not None:
            actions = torch.tanh(dist.loc + dist.scale * self.held_noise(dist.loc))
        else:
            actions = dist.sample() if sample else dist.mean
        # Rescale from [-1, 1] to actual bounds
        actions = self.rescale_actions(actions)
        if actions.ndim != 2:
            raise ValueError(f"Actions tensor must be 2-dimensional, got shape {actions.shape}")

        return actions

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        if actor_loss_info is None:
            # No actor update happened on this critic step (policy_frequency skip)
            return
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_loss is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = -float('inf')  # last_mean_rewards (best-ever watermark) is deliberately NOT reset here
        self.algo_observer.after_clear_stats()

    def _replay_rewards(self, shaped_rewards, infos):
        """Replay reward columns (B, H): the shaped env reward, or with several action groups
        the shaped per-group rewards from infos['reward_heads']."""
        if self.num_heads == 1:
            return shaped_rewards.unsqueeze(1)
        heads = infos.get('reward_heads') if isinstance(infos, dict) else None
        if heads is None:
            raise ValueError(f"{self.num_heads} action groups need per-group rewards in infos['reward_heads'] (B, H)")
        heads = torch.as_tensor(heads, device=self._device).float().view(shaped_rewards.shape[0], self.num_heads)
        return self.rewards_shaper(heads)

    @staticmethod
    def _obs_aux(obs):
        return obs.get('obs_aux') if isinstance(obs, dict) else None

    def _store_transitions(self, obs, action, shaped_rewards, next_obs,
                           dones, terminated, truncated_mask, infos, replay_rewards=None, obs_aux=None,
                           next_obs_aux=None):
        """Write one vec-step of transitions to replay and update episode accounting.

        next_step autoreset: rows whose PREVIOUS step ended an episode are
        reset-step garbage (action ignored, filler reward, post-reset next_obs)
        and are excluded from both replay and accounting. next_obs at a
        truncation is already the TRUE final obs, so no proxy is needed. Live
        rows are gathered with a single nonzero + index_select per tensor
        (boolean masking would re-run nonzero — a host sync — per tensor).

        same_step autoreset: every row is live, so the full batch is written
        directly with no masking. next_obs holds the post-reset obs at a
        truncation, so substitute the true final obs (from info, else the
        s_t proxy).

        replay_rewards (B, H) are the reward columns (default: shaped_rewards).
        With n_step > 1 the replay folds full steps, so next_step rows go in
        with a live mask instead of being selected here.
        """
        if replay_rewards is None:
            replay_rewards = shaped_rewards.unsqueeze(1)
        if self.autoreset_mode == 'next_step':
            live = ~self._prev_dones
            live_f = live.float()
            self.current_rewards.add_(shaped_rewards * live_f)
            self.current_lengths.add_(live_f)
            all_done_indices = (dones.bool() & live).nonzero(as_tuple=False)

            if self.n_step > 1:
                self.replay_buffer.add(obs, action, replay_rewards, next_obs, terminated.unsqueeze(1),
                                       truncated_mask.unsqueeze(1), live=live)
            else:
                live_idx = live.nonzero(as_tuple=True)[0]
                if live_idx.numel() > 0:  # numel reads shape on host, no device sync
                    self.replay_buffer.add(
                        obs.index_select(0, live_idx),
                        action.index_select(0, live_idx),
                        replay_rewards.index_select(0, live_idx),
                        next_obs.index_select(0, live_idx),
                        terminated.index_select(0, live_idx).unsqueeze(1),
                        truncated_mask.index_select(0, live_idx).unsqueeze(1))
        else:
            replay_next_obs = next_obs
            if truncated_mask.any():
                replay_next_obs = next_obs.clone()
                final_obs = infos.get('final_observation') if isinstance(infos, dict) else None
                if isinstance(final_obs, dict):
                    final_obs = final_obs['obs']
                if final_obs is not None:
                    final_obs = self.cast_obs(final_obs).to(replay_next_obs.dtype)
                    replay_next_obs[truncated_mask] = final_obs[truncated_mask]
                else:
                    replay_next_obs[truncated_mask] = obs[truncated_mask]
                if next_obs_aux is not None:
                    # aux of the true final obs, int (B, aux_dim)
                    next_obs_aux = torch.where(truncated_mask.unsqueeze(-1),
                                               torch.as_tensor(infos['final_obs_aux'], device=self._device),
                                               next_obs_aux)

            self.current_rewards.add_(shaped_rewards)
            self.current_lengths.add_(1.0)
            all_done_indices = dones.nonzero(as_tuple=False)

            aux = {}
            if self.obs_compactor is not None:
                aux = {'obs_aux': obs_aux, 'next_obs_aux': next_obs_aux}
            self.replay_buffer.add(
                obs, action, replay_rewards, replay_next_obs,
                terminated.unsqueeze(1), truncated_mask.unsqueeze(1), **aux)

        done_indices = all_done_indices[::self.num_agents]
        self.game_rewards.update(self.current_rewards[done_indices])
        self.game_lengths.update(self.current_lengths[done_indices])
        self.algo_observer.process_infos(infos, done_indices)
        not_dones = 1.0 - dones.float()
        self.current_rewards.mul_(not_dones)
        self.current_lengths.mul_(not_dones)

        # clone is required: when dones is already torch.bool, .bool() returns an
        # alias that may share memory with an env-owned / numpy buffer.
        self._prev_dones = dones.bool().clone()

    def play_steps(self, random_exploration=False):
        total_time_start = time.perf_counter()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []

        obs = self.obs
        obs_aux = self._obs_aux(obs)
        if isinstance(obs, dict):
            obs = self.obs['obs']

        for s in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                # Sample uniformly from the correct action bounds
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device)
                action = self.action_low + (self.action_high - self.action_low) * action
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.perf_counter()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)

            step_end = time.perf_counter()
            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            # Shape rewards BEFORE tracking (so logged rewards match training rewards)
            shaped_rewards = self.rewards_shaper(rewards)
            replay_rewards = self._replay_rewards(shaped_rewards, infos)

            # With value_bootstrap=False the truncated column stays all-False, so
            # truncations train as hard terminations (legacy behavior).
            truncated_mask = torch.zeros(dones.shape[0], dtype=torch.bool, device=self._device)
            if self.value_bootstrap and 'time_outs' in infos:
                truncated_mask = self.cast_obs(infos['time_outs'])
                if truncated_mask.dtype != torch.bool:
                    truncated_mask = truncated_mask.bool()
            terminated = dones.bool() & ~truncated_mask

            if isinstance(next_obs, dict):
                next_obs_processed = next_obs['obs']
            else:
                next_obs_processed = next_obs
            # Ingest each fresh env frame into the normalizer exactly once here
            # (incl. warmup and post-reset frames); update() never touches it.
            self._update_obs_stats(next_obs_processed)
            self.obs = next_obs
            if self.reward_scaler is not None:
                self.reward_scaler.update(replay_rewards, dones.bool())

            next_obs_aux = self._obs_aux(next_obs)
            self._store_transitions(obs, action, shaped_rewards, next_obs_processed,
                                    dones, terminated, truncated_mask, infos, replay_rewards,
                                    obs_aux, next_obs_aux)

            obs = next_obs_processed
            obs_aux = next_obs_aux

            if not random_exploration:
                self.set_train()

                update_time_start = time.perf_counter()
                # num_updates_per_step gradient updates per env step (UTD ratio)
                updates = self.num_updates_per_step if self._can_sample() else 0
                for _ in range(updates):
                    actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                    self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                    critic1_losses.append(critic1_loss)
                    critic2_losses.append(critic2_loss)
                update_time_end = time.perf_counter()
                update_time = update_time_end - update_time_start
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.perf_counter()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses

    def _can_sample(self):
        if not self.multi_gpu:
            return self.replay_buffer.can_sample()
        if not self._can_sample_all:  # one host sync per step until every rank's replay can sample
            self._can_sample_all = self._all_ranks(self.replay_buffer.can_sample())
        return self._can_sample_all

    def _sync_obs_stats(self):
        if self.multi_gpu and self.normalize_input:
            merge_rank_stats(self.model.running_mean_std, lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM))

    def train_epoch(self):
        # epoch_num is 1-based here (train() increments it before calling train_epoch),
        # so `<=` gives exactly num_warmup_steps warmup epochs.
        random_exploration = self.epoch_num <= self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        self._setup_multi_gpu()
        total_time = 0

        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses = self.train_epoch()

            total_time += epoch_total_time
            self._sync_obs_stats()

            # frames of every rank
            curr_frames = self.num_frames_per_epoch * self.world_size
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                self.epoch_num, self.max_epochs, self.frame, self.max_frames)

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)

            # Skip warmup epochs (no updates -> empty loss lists). The per-list
            # length guards below also cover epochs where policy_frequency skipped
            # every actor update, so only computed stats are logged.
            if self.epoch_num > self.num_warmup_steps:
                if len(actor_losses) > 0:
                    self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)
                if len(entropies) > 0:
                    self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), self.frame)

                if len(alpha_losses) > 0:
                    self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), self.frame)
                if len(alphas) > 0:
                    self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), self.frame)
                # per action group, when there are several
                for name, values in self._head_stats.items():
                    for h, v in enumerate(torch.stack(values).float().mean(dim=0).tolist()):
                        self.writer.add_scalar(f'{name}{h}', v, self.frame)
                if self.lr_schedule == 'cosine':
                    self.writer.add_scalar('info/actor_lr', self.actor_optimizer.param_groups[0]['lr'], self.frame)
                    self.writer.add_scalar('info/critic_lr', self.critic_optimizer.param_groups[0]['lr'], self.frame)
            self._head_stats.clear()

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            # Exits and periodic saves run EVERY epoch; only reward/length stats
            # and the best-checkpoint comparison are gated on completed episodes.
            should_exit = False
            # -inf watermark until the first episode completes (str() -> '-inf' in checkpoint names)
            mean_rewards = -np.inf

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)))
                        should_exit = True

            checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

            if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                    + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX EPOCHS NUM!')
                should_exit = True

            if self.frame >= self.max_frames and self.max_frames != -1:
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                    + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX FRAMES NUM!')
                should_exit = True

            if not should_exit and self.stop_fn is not None and self.stop_fn(self):
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_custom_stop_ep_' + str(self.epoch_num)))
                print('Custom stop callback returned True. Stopping training.')
                should_exit = True

            update_time = 0

            if self.multi_gpu:  # rank 0 decides (score_to_win and stop_fn read its episodes)
                flag = torch.tensor([float(should_exit)], device=self._device)
                dist.broadcast(flag, 0)
                should_exit = bool(flag.item())
            if should_exit:
                return self.last_mean_rewards, self.epoch_num
