"""FlashSAC networks (network name ``flash_sac``, model ``soft_actor_critic``).

Batch norm and unit-norm residual MLPs, an ensemble critic with one twin pair per action group
(2 x H members, index twin * H + group), a categorical or scalar value head, one actor tower per
action group and an optional Conv1d encoder for a trailing window of the obs vector.

Every layer with a ``project`` method is re-projected after each optimizer step: linear rows to
unit L2 norm, batch norm (weight, bias) jointly and RMS norm weight to norm sqrt(d). The policy
head and the window encoder are not constrained. Batch norm follows ``module.training``.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.sac_helper import SquashedNormal, group_masks

_EPS = 1e-8


def _unit_rows_(weight):
    weight.copy_(F.normalize(weight, dim=-1, eps=_EPS))


def _scale_to_sqrt_dim_(*params):
    # joint L2 norm of the last dim -> sqrt(d)
    dim = params[0].shape[-1]
    sq = sum((p * p).sum(dim=-1, keepdim=True) for p in params)
    factor = math.sqrt(dim) * torch.rsqrt(sq + _EPS)
    for p in params:
        p.mul_(factor)


@torch.no_grad()
def project_weights(module):
    for m in module.modules():
        if m is not module and hasattr(m, 'project'):
            m.project()


class UnitLinear(nn.Module):
    """Bias-free linear map with unit-norm rows, (members, B, in) -> (members, B, out) when members > 0."""

    def __init__(self, in_dim, out_dim, members=0):
        super().__init__()
        shape = (members, out_dim, in_dim) if members else (out_dim, in_dim)
        self.weight = nn.Parameter(torch.empty(shape))
        for w in (self.weight.data if members else [self.weight.data]):
            nn.init.orthogonal_(w)

    def forward(self, x):
        if self.weight.dim() == 2:
            return F.linear(x, self.weight)
        return torch.matmul(x, self.weight.transpose(1, 2))

    @torch.no_grad()
    def project(self):
        _unit_rows_(self.weight)


class UnitBatchNorm(nn.Module):
    """Batch norm over the batch dim with (weight, bias) kept at joint norm sqrt(d); per member when members > 0."""

    def __init__(self, dim, members=0, momentum=0.01, eps=1e-5):
        super().__init__()
        shape = (members, dim) if members else (dim,)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if self.weight.dim() == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                self.training, self.momentum, self.eps)
        if self.training:
            var, mean = torch.var_mean(x, dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                n = x.shape[1]
                # running var gets the unbiased batch variance, as torch batch norm does
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_(var.squeeze(1).float() * (n / max(n - 1, 1)), self.momentum)
        else:
            mean, var = self.running_mean.unsqueeze(1), self.running_var.unsqueeze(1)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight.unsqueeze(-2) + self.bias.unsqueeze(-2)

    @torch.no_grad()
    def project(self):
        _scale_to_sqrt_dim_(self.weight, self.bias)


class UnitRMSNorm(nn.Module):
    def __init__(self, dim, members=0, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((members, dim) if members else (dim,)))
        self.eps = eps

    def forward(self, x):
        w = self.weight if self.weight.dim() == 1 else self.weight.unsqueeze(1)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * w

    @torch.no_grad()
    def project(self):
        _scale_to_sqrt_dim_(self.weight)


class ResidualBlock(nn.Module):
    """x + relu(BN(W2 relu(BN(W1 x)))), hidden width expansion * dim."""

    def __init__(self, dim, expansion=4, members=0, momentum=0.01):
        super().__init__()
        self.w1 = UnitLinear(dim, dim * expansion, members)
        self.norm1 = UnitBatchNorm(dim * expansion, members, momentum)
        self.w2 = UnitLinear(dim * expansion, dim, members)
        self.norm2 = UnitBatchNorm(dim, members, momentum)

    def forward(self, x):
        y = F.relu(self.norm1(self.w1(x)))
        y = F.relu(self.norm2(self.w2(y)))
        return x + y


class Trunk(nn.Module):
    """BN -> unit linear -> residual blocks -> RMS norm."""

    def __init__(self, in_dim, units, blocks, expansion, members=0, momentum=0.01):
        super().__init__()
        self.in_norm = UnitBatchNorm(in_dim, members, momentum)
        self.in_proj = UnitLinear(in_dim, units, members)
        self.blocks = nn.Sequential(*[ResidualBlock(units, expansion, members, momentum) for _ in range(blocks)])
        self.out_norm = UnitRMSNorm(units, members)

    def forward(self, x):
        return self.out_norm(self.blocks(self.in_proj(self.in_norm(x))))


class ObsWindowEncoder(nn.Module):
    """Replaces the last ``size`` obs columns, laid out (steps, size // steps) row-major, by a ``latent`` code.

    Pointwise Conv1d, two strided Conv1d (kernels follow the step count), flatten, linear; SiLU between.
    """

    def __init__(self, obs_dim, size, steps, hidden=128, latent=128):
        super().__init__()
        if size % steps or size > obs_dim:
            raise ValueError(f'obs window {size} must be a multiple of steps={steps} and fit the obs ({obs_dim})')
        self.size, self.steps, self.per_step = size, steps, size // steps
        k1, s1 = (2, 2) if steps <= 6 else ((4, 2) if steps <= 12 else (6, 2))
        len1 = (steps - k1) // s1 + 1
        k2, s2 = (1, 1) if len1 <= 1 else ((2, 1) if len1 <= 5 else (4, 2))
        len2 = (len1 - k2) // s2 + 1
        if len1 < 1 or len2 < 1:
            raise ValueError(f'cannot encode a window of {steps} steps')
        c1 = max(hidden // 2, 8)
        c2 = max(c1 // 2, 8)
        self.conv_in = nn.Conv1d(self.per_step, hidden, kernel_size=1)
        self.conv1 = nn.Conv1d(hidden, c1, kernel_size=k1, stride=s1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, stride=s2)
        self.head = nn.Linear(c2 * len2, latent)
        self.out_dim = obs_dim - size + latent

    def forward(self, obs):
        window = obs[:, -self.size:].reshape(obs.shape[0], self.steps, self.per_step).transpose(1, 2)
        x = F.silu(self.conv_in(window))
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return torch.cat([obs[:, :-self.size], self.head(x.flatten(1))], dim=-1)


class PolicyHead(nn.Module):
    """Plain linear mean (init N(0, 1e-3)) and log-std (init 0), log-std squashed into its bounds."""

    def __init__(self, units, action_dim, log_std_bounds):
        super().__init__()
        self.mu = nn.Linear(units, action_dim)
        self.log_std = nn.Linear(units, action_dim)
        nn.init.normal_(self.mu.weight, std=1e-3)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, x):
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(self.log_std(x)) + 1)
        return self.mu(x), log_std


class ActorTower(nn.Module):
    def __init__(self, obs_dim, action_dim, units, blocks, expansion, log_std_bounds, window=None, momentum=0.01):
        super().__init__()
        self.encoder = ObsWindowEncoder(obs_dim, **window) if window else nn.Identity()
        in_dim = self.encoder.out_dim if window else obs_dim
        self.trunk = Trunk(in_dim, units, blocks, expansion, 0, momentum)
        self.head = PolicyHead(units, action_dim, log_std_bounds)

    def forward(self, obs):
        return self.head(self.trunk(self.encoder(obs)))


class GroupActor(nn.Module):
    """One tower per action group; every tower reads the full obs. Returns a SquashedNormal in env action order."""

    def __init__(self, obs_dim, action_groups, **tower_args):
        super().__init__()
        self.towers = nn.ModuleList(ActorTower(obs_dim, len(ids), **tower_args) for ids in action_groups)
        order = [i for ids in action_groups for i in ids]
        self.reorder = order != sorted(order)
        self.register_buffer('to_env_order', torch.argsort(torch.tensor(order)), persistent=False)

    def forward(self, obs):
        outs = [tower(obs) for tower in self.towers]
        mu = torch.cat([m for m, _ in outs], dim=-1)
        log_std = torch.cat([s for _, s in outs], dim=-1)
        if self.reorder:
            mu, log_std = mu[:, self.to_env_order], log_std[:, self.to_env_order]
        return SquashedNormal(mu, log_std.exp())


class EnsembleCritic(nn.Module):
    """2 x H Q members over [encoded obs, action] as one batched ensemble.

    forward(obs (B, O), actions (B, A) or one view per member (M, B, A)) -> Q (M, B) and, for a
    categorical head, log-probs (M, B, num_bins) on bins evenly spaced over [v_min, v_max] (else None).
    """

    def __init__(self, obs_dim, action_dim, num_heads, units, blocks, expansion, num_bins=0, v_min=-5.0,
                 v_max=5.0, window=None, momentum=0.01):
        super().__init__()
        self.members = 2 * num_heads
        self.num_bins = num_bins
        self.encoder = ObsWindowEncoder(obs_dim, **window) if window else nn.Identity()
        in_dim = (self.encoder.out_dim if window else obs_dim) + action_dim
        self.trunk = Trunk(in_dim, units, blocks, expansion, self.members, momentum)
        out_dim = num_bins if num_bins else 1
        self.out_proj = UnitLinear(units, out_dim, self.members)
        self.out_bias = nn.Parameter(torch.zeros(self.members, out_dim))
        if num_bins:
            self.register_buffer('bins', torch.linspace(v_min, v_max, num_bins), persistent=False)

    def forward(self, obs, actions):
        z = self.encoder(obs)
        m = self.members
        if actions.dim() == 2:
            x = torch.cat([z, actions], dim=-1).unsqueeze(0).expand(m, -1, -1)
        else:
            x = torch.cat([z.unsqueeze(0).expand(m, -1, -1), actions], dim=-1)
        out = self.out_proj(self.trunk(x)) + self.out_bias.unsqueeze(1)
        if not self.num_bins:
            return out.squeeze(-1), None
        log_probs = F.log_softmax(out.float(), dim=-1)
        return (log_probs.exp() * self.bins).sum(dim=-1), log_probs


class FlashSACBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return FlashSACBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        """actor / critic / critic_target as SACBuilder.Network; the agent reads the group and value-head layout."""

        ensemble_critic = True

        def __init__(self, params, **kwargs):
            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            groups = kwargs.pop('action_groups', None) or [list(range(action_dim))]
            NetworkBuilder.BaseNetwork.__init__(self)

            actor_cfg = params.get('actor', {})
            critic_cfg = params.get('critic', {})
            expansion = params.get('expansion', 4)
            momentum = params.get('bn_momentum', 0.01)
            window = params.get('obs_window', None)

            self.action_groups = [list(ids) for ids in groups]
            self.num_heads = len(self.action_groups)
            self.register_buffer('group_masks', group_masks(self.action_groups, action_dim), persistent=False)
            self.num_bins = critic_cfg.get('num_bins', 0)
            self.v_min = float(critic_cfg.get('v_min', -5.0))
            self.v_max = float(critic_cfg.get('v_max', 5.0))

            self.actor = GroupActor(obs_dim, self.action_groups, units=actor_cfg.get('units', 128),
                                    blocks=actor_cfg.get('blocks', 2), expansion=expansion,
                                    log_std_bounds=params.get('log_std_bounds', [-5.0, 0.0]), window=window,
                                    momentum=momentum)
            self.critic = EnsembleCritic(obs_dim, action_dim, self.num_heads, units=critic_cfg.get('units', 1024),
                                         blocks=critic_cfg.get('blocks', 2), expansion=expansion,
                                         num_bins=self.num_bins, v_min=self.v_min, v_max=self.v_max, window=window,
                                         momentum=momentum)
            project_weights(self.actor)
            project_weights(self.critic)
            self.critic_target = copy.deepcopy(self.critic)

        def forward(self, obs_dict):
            dist = self.actor(obs_dict['obs'])
            return dist.loc, dist.scale

        def is_separate_critic(self):
            return True

        def gate_actions(self, actions):
            """(B, A) -> (2H, B, A): member twin * H + h carries gradient only through group h's dims."""
            held = actions.detach()
            gated = torch.stack([torch.where(m, actions, held) for m in self.group_masks])
            return gated.repeat(2, 1, 1)

        def post_actor_step(self):
            (self._project_actor or project_weights)(self.actor)

        def post_critic_step(self):
            (self._project_critic or project_weights)(self.critic)

        _project_actor = _project_critic = None

        def compile_projection(self):
            """Fuse the per-layer re-projection kernels (torch.compile, no CUDA graphs: it writes parameters)."""
            self._project_actor = torch.compile(project_weights)
            self._project_critic = torch.compile(project_weights)
