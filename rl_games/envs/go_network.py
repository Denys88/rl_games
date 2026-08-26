"""Go 9x9 ResNet actor-critic for rl_games (torch side).

KataGo-lite: a residual conv trunk with periodic global-pooling bias layers,
a masked 82-way policy head, a scalar value head (what PPO trains on), and
optional auxiliary heads trained from env-derived targets:

    ownership   : final owner per point, tanh, MSE          (target: go_ownership)
    score_dist  : 41-way categorical over score diff +/-20  (target: go_score)
    opp_policy  : opponent's reply distribution, soft CE    (target: go_opp_dist)
    plies_left  : plies until the game ends / 80, Huber     (target: go_plies_left)

Aux targets arrive in the training minibatch via the rollout-targets hook
(see PgxGoVecEnv.process_rollout_targets and a2c_common.play_steps); heads
whose targets are absent contribute nothing. Aux weights anneal linearly to
`aux_anneal_to` x initial over the run (driven by set_train_progress).

Used with model `discrete_a2c` — the standard masked-categorical model; this
module only supplies (logits, value) plus the aux losses through the generic
`get_aux_loss` hook.

Config:

    network:
      name: go_resnet
      blocks: 6
      channels: 64
      gpool_every: 2          # a global-pooling bias layer every k blocks
      value_units: 128
      aux_heads: [ownership, score_dist, opp_policy, plies_left]
      aux_weights: {ownership: 1.0, score_dist: 0.25, opp_policy: 0.15, plies_left: 0.05}
      aux_anneal_to: 0.5

No normalization layers by design: the flax twin (go_flax.py) must match this
computation bit-for-bit up to float error, and norm-free keeps the converter
and the jitted opponent step trivial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch.network_builder import NetworkBuilder

DEFAULT_AUX_WEIGHTS = {
    'ownership': 1.0,
    'score_dist': 0.25,
    'opp_policy': 0.15,
    'plies_left': 0.05,
}

SCORE_BINS = 41  # score diff in [-20, +20], tails clamped


class GlobalPoolBias(nn.Module):
    """KataGo-style: (mean, max) pool over the board -> linear -> channel bias."""

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(2 * channels, channels)

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return x + self.fc(pooled)[:, :, None, None]


class ResBlock(nn.Module):
    """Pre-activation residual block, norm-free (last conv zero-init)."""

    def __init__(self, channels, gpool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gpool = GlobalPoolBias(channels) if gpool else None
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        y = self.conv1(F.relu(x))
        if self.gpool is not None:
            y = self.gpool(y)
        y = self.conv2(F.relu(y))
        return x + y


class NestedBottleneckBlock(nn.Module):
    """KataGo-style nested bottleneck (the 'nbt' in b18c384nbt): 1x1 down to
    `mid` channels, two nested residual pairs at mid, 1x1 back up, outer
    residual. Pre-activation, norm-free (up-projection zero-init)."""

    def __init__(self, channels, mid, gpool=False):
        super().__init__()
        self.down = nn.Conv2d(channels, mid, 1)
        self.inner1 = ResBlock(mid)
        self.inner2 = ResBlock(mid, gpool=gpool)
        self.up = nn.Conv2d(mid, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.down(F.relu(x))
        y = self.inner1(y)
        y = self.inner2(y)
        y = self.up(F.relu(y))
        return x + y


class GoResNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)
            input_shape = kwargs.pop('input_shape')  # (H, W, planes) NHWC
            actions_num = kwargs.pop('actions_num')
            self.value_size = kwargs.pop('value_size', 1)

            assert len(input_shape) == 3, f'expected board obs, got {input_shape}'
            self.size = input_shape[0]
            self.n_points = self.size * self.size
            assert actions_num == self.n_points + 1
            in_planes = input_shape[2]

            self.blocks_num = params.get('blocks', 6)
            self.channels = params.get('channels', 64)
            gpool_every = params.get('gpool_every', 2)
            value_units = params.get('value_units', 128)
            self.block_type = params.get('block_type', 'res')
            self.bottleneck_channels = params.get(
                'bottleneck_channels', self.channels // 2)
            self.aux_head_names = list(params.get(
                'aux_heads', ['ownership', 'score_dist', 'opp_policy', 'plies_left']))
            weights = dict(DEFAULT_AUX_WEIGHTS)
            weights.update(params.get('aux_weights', {}) or {})
            self.aux_weights = weights
            self.aux_anneal_to = params.get('aux_anneal_to', 0.5)
            self._aux_scale = 1.0
            self._aux_losses = None

            c = self.channels
            self.stem = nn.Conv2d(in_planes, c, 3, padding=1)
            if self.block_type == 'nbt':
                self.blocks = nn.ModuleList([
                    NestedBottleneckBlock(
                        c, self.bottleneck_channels,
                        gpool=(gpool_every > 0 and (i + 1) % gpool_every == 0))
                    for i in range(self.blocks_num)
                ])
            else:
                self.blocks = nn.ModuleList([
                    ResBlock(c, gpool=(gpool_every > 0 and (i + 1) % gpool_every == 0))
                    for i in range(self.blocks_num)
                ])

            # policy: per-point logits from a 1x1 conv, pass logit from the pool
            self.policy_conv = nn.Conv2d(c, 1, 1)
            self.policy_pass = nn.Linear(2 * c, 1)
            # value: pooled trunk -> mlp -> scalar
            self.value_fc1 = nn.Linear(2 * c, value_units)
            self.value_fc2 = nn.Linear(value_units, self.value_size)
            self.value = self.value_fc2  # for get_value_layer()

            if 'ownership' in self.aux_head_names:
                self.own_conv = nn.Conv2d(c, 1, 1)
            if 'score_dist' in self.aux_head_names:
                self.score_fc1 = nn.Linear(2 * c, value_units)
                self.score_fc2 = nn.Linear(value_units, SCORE_BINS)
            if 'opp_policy' in self.aux_head_names:
                self.opp_conv = nn.Conv2d(c, 1, 1)
                self.opp_pass = nn.Linear(2 * c, 1)
            if 'plies_left' in self.aux_head_names:
                self.plies_fc = nn.Linear(2 * c, 1)

        def set_train_progress(self, epoch_num, max_epochs):
            if max_epochs and max_epochs > 0:
                p = min(1.0, max(0.0, epoch_num / max_epochs))
                self._aux_scale = 1.0 + (self.aux_anneal_to - 1.0) * p

        @staticmethod
        def _masked_mean(per_sample, valid):
            valid = valid.float().reshape(-1)
            denom = valid.sum().clamp(min=1.0)
            return (per_sample.reshape(-1) * valid).sum() / denom

        def _compute_aux(self, obs_dict, trunk, pooled):
            losses = {}
            b = trunk.shape[0]
            if 'ownership' in self.aux_head_names and 'go_ownership' in obs_dict:
                pred = torch.tanh(self.own_conv(trunk).reshape(b, -1))
                target = obs_dict['go_ownership']
                per = F.mse_loss(pred, target, reduction='none').mean(-1)
                losses['ownership'] = self._masked_mean(per, obs_dict['go_terminal_valid'])
            if 'score_dist' in self.aux_head_names and 'go_score' in obs_dict:
                logits = self.score_fc2(F.relu(self.score_fc1(pooled)))
                half = (SCORE_BINS - 1) // 2
                idx = (obs_dict['go_score'].round().clamp(-half, half) + half).long()
                per = F.cross_entropy(logits, idx, reduction='none')
                losses['score_dist'] = self._masked_mean(per, obs_dict['go_terminal_valid'])
            if 'opp_policy' in self.aux_head_names and 'go_opp_dist' in obs_dict:
                point_logits = self.opp_conv(trunk).reshape(b, -1)
                pass_logit = self.opp_pass(pooled)
                logits = torch.cat([point_logits, pass_logit], dim=1)
                target = obs_dict['go_opp_dist']
                per = -(target * F.log_softmax(logits, dim=-1)).sum(-1)
                losses['opp_policy'] = self._masked_mean(per, obs_dict['go_opp_valid'])
            if 'plies_left' in self.aux_head_names and 'go_plies_left' in obs_dict:
                pred = self.plies_fc(pooled).reshape(-1)
                target = obs_dict['go_plies_left'] / 80.0
                per = F.huber_loss(pred, target, reduction='none')
                losses['plies_left'] = self._masked_mean(per, obs_dict['go_terminal_valid'])

            self._aux_losses = {
                k: v * self.aux_weights.get(k, 1.0) * self._aux_scale
                for k, v in losses.items()
            } or None

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            x = obs.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            x = self.stem(x)
            for block in self.blocks:
                x = block(x)
            trunk = F.relu(x)
            pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)

            point_logits = self.policy_conv(trunk).reshape(trunk.shape[0], -1)
            pass_logit = self.policy_pass(pooled)
            logits = torch.cat([point_logits, pass_logit], dim=1)
            value = self.value_fc2(F.relu(self.value_fc1(pooled)))

            self._aux_losses = None
            if obs_dict.get('is_train', False):
                self._compute_aux(obs_dict, trunk, pooled)
            return logits, value, None

        def get_aux_loss(self):
            return self._aux_losses

    def build(self, name, **kwargs):
        return GoResNetBuilder.Network(self.params, **kwargs)
