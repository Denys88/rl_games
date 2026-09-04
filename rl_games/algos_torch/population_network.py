"""Population actor-critic: N independent MLP policies in one rl_games network.

Every parameter carries a leading slot axis (weights (N, in, out), biases
(N, out), heads, sigma (N, A)). A row's slot is the argmax of the LAST N
observation dims (a one-hot appended by EnvpoolSoccerVecEnv in population
mode). forward() groups rows by slot, pads to (N, G, D), runs the MLP with
batched matmuls and gathers the outputs back into row order, so the standard
PPO update trains all N policies at once and gradients never cross slots.

Config, under `network`:

    name: population_actor_critic
    population_size: 8
    seed: 0                      # slot k is initialised under seed + 100*k
    mlp: {units: [512, 256, 128], activation: elu}
    space: {continuous: {fixed_sigma: True, sigma_init: {name: const_initializer, val: 0}}}

Not supported: rnn, cnn, discrete actions, separate critic, dict observations.
"""

import numpy as np
import torch
import torch.nn as nn

from rl_games.algos_torch import network_builder


def slot_of(obs, population_size):
    """Slot index per row from the trailing one-hot.

    The network sees the observation AFTER rl_games' running-mean-std, but the
    argmax survives it: a one-hot dim's running mean lies in [0, 1], so its
    normalised "on" value (1 - m) / s is >= 0 while every "off" value -m / s
    is <= 0."""
    return obs[:, -population_size:].argmax(dim=1)


class PopulationBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return PopulationBuilder.Network(self.params, **kwargs)

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            kwargs.pop('num_seqs', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.N = int(params['population_size'])
            self.seed = int(params.get('seed', 0))
            mlp = params['mlp']
            units = list(mlp['units'])
            if not units:
                raise ValueError('population_actor_critic needs a non-empty mlp')
            self.activation = self.activations_factory.create(mlp.get('activation', 'elu'))
            space = params['space']['continuous']
            if not space.get('fixed_sigma', True):
                raise NotImplementedError('population_actor_critic supports fixed_sigma only')
            self.obs_dim = int(np.prod(input_shape)) - self.N
            self.actions_num = actions_num
            self._init_calls = 0
            sizes = [self.obs_dim] + units
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            for i in range(len(units)):
                w, b = self._stacked_linear(sizes[i], sizes[i + 1])
                self.weights.append(w)
                self.biases.append(b)
            self.mu_w, self.mu_b = self._stacked_linear(units[-1], actions_num)
            self.value_w, self.value_b = self._stacked_linear(units[-1], self.value_size)
            sigma_val = float(space.get('sigma_init', {}).get('val', 0.0))
            self.sigma = nn.Parameter(torch.full((self.N, actions_num), sigma_val))

        def _stacked_linear(self, in_size, out_size):
            """(N, in, out) weights + (N, out) biases, slot k under its own seed
            (nn.Linear default init, same as rl_games' 'default' initializer)."""
            ws, bs = [], []
            for k in range(self.N):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.seed + 100 * k + self._init_calls)
                    lin = nn.Linear(in_size, out_size)
                ws.append(lin.weight.detach().t().clone())
                bs.append(lin.bias.detach().clone())
            self._init_calls += 1
            return nn.Parameter(torch.stack(ws)), nn.Parameter(torch.stack(bs))

        def load(self, params):
            self.params = params

        def is_rnn(self):
            return False

        def is_separate_critic(self):
            return False

        def get_default_rnn_state(self):
            return None

        def get_value_layer(self):
            return None

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            B = obs.shape[0]
            slot = slot_of(obs, self.N)
            x = obs[:, :self.obs_dim]
            order = torch.argsort(slot, stable=True)
            counts = torch.bincount(slot, minlength=self.N)
            G = int(counts.max().item())
            starts = torch.cumsum(counts, 0) - counts
            sorted_slot = slot[order]
            pos = torch.arange(B, device=obs.device) - starts[sorted_slot]
            idx = torch.zeros(self.N, G, dtype=torch.long, device=obs.device)
            idx[sorted_slot, pos] = order
            mask = torch.zeros(self.N, G, dtype=torch.bool, device=obs.device)
            mask[sorted_slot, pos] = True
            h = x[idx]                                   # (N, G, D); pads duplicate row 0
            for w, b in zip(self.weights, self.biases):
                h = self.activation(torch.baddbmm(b.unsqueeze(1), h, w))
            mu_g = torch.baddbmm(self.mu_b.unsqueeze(1), h, self.mu_w)
            v_g = torch.baddbmm(self.value_b.unsqueeze(1), h, self.value_w)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(B, device=obs.device)
            mu = mu_g[mask][inv]                         # rows in sorted order -> original order
            value = v_g[mask][inv]
            logstd = self.sigma[slot]
            return mu, logstd, value, None


def extract_slot(state_dict, k, base_obs_dim):
    """Slice slot k out of a population checkpoint into a standard
    `actor_critic` (continuous_a2c_logstd) state dict with `base_obs_dim`
    inputs, so soccer_play.py / soccer_eval.py load it unchanged."""
    sd = {key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key: v
          for key, v in state_dict.items()}
    out = {}
    n_layers = len([key for key in sd if key.startswith('a2c_network.weights.')])
    for i in range(n_layers):
        out[f'a2c_network.actor_mlp.{2 * i}.weight'] = sd[f'a2c_network.weights.{i}'][k].t().contiguous()
        out[f'a2c_network.actor_mlp.{2 * i}.bias'] = sd[f'a2c_network.biases.{i}'][k].clone()
    out['a2c_network.mu.weight'] = sd['a2c_network.mu_w'][k].t().contiguous()
    out['a2c_network.mu.bias'] = sd['a2c_network.mu_b'][k].clone()
    out['a2c_network.value.weight'] = sd['a2c_network.value_w'][k].t().contiguous()
    out['a2c_network.value.bias'] = sd['a2c_network.value_b'][k].clone()
    out['a2c_network.sigma'] = sd['a2c_network.sigma'][k].clone()
    for key, v in sd.items():
        if key.startswith('running_mean_std.'):
            out[key] = v[:base_obs_dim].clone() if v.dim() == 1 else v.clone()
        elif key.startswith('value_mean_std.'):
            out[key] = v.clone()
    return out
