"""actor_critic_state_aux: standard actor-critic plus an auxiliary head that
regresses the privileged `states` vector from the actor trunk's last hidden
layer. No teacher: the privileged information is used only as a prediction
target, so it can be compared with frozen-teacher distillation
(rl_games/common/distillation.py) on the same pixel student.

Config, under `network`:

    name: actor_critic_state_aux
    state_aux:
      coef: 1.0            # weight of the MSE (on running-normalised targets)
      units: [256]         # hidden layers of the head
      indices: null        # optional list of state dims to predict (default all)
    ... (cnn / mlp / space as for actor_critic)

The agent must store `states` (algo config `state_aux: {}` turns that on; the
env returns {'obs': ..., 'states': ...}) and passes `state_shape` in the
build config. The loss is picked up through the model's get_aux_loss().
"""

import torch
import torch.nn as nn

from rl_games.algos_torch import network_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class StateAuxBuilder(network_builder.A2CBuilder):
    def build(self, name, **kwargs):
        return StateAuxBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            state_shape = kwargs.pop('state_shape', None)
            super().__init__(params, **kwargs)
            if self.separate:
                raise NotImplementedError('actor_critic_state_aux needs a shared trunk (separate: False)')
            if state_shape is None:
                raise ValueError("actor_critic_state_aux needs 'state_shape' in the build config "
                                 "(set algo config `state_aux: {}` and an env that returns states)")
            cfg = params.get('state_aux', {})
            self.aux_coef = float(cfg.get('coef', 1.0))
            indices = cfg.get('indices', None)
            self.register_buffer('aux_indices',
                                 torch.as_tensor(indices, dtype=torch.long) if indices is not None else torch.empty(0, dtype=torch.long))
            out_dim = len(indices) if indices is not None else int(state_shape[0])
            units = list(cfg.get('units', [256]))
            latent = self.units[-1] if len(self.units) else self._calc_input_size(self.actor_cnn, kwargs.get('input_shape'))
            layers, size = [], latent
            for u in units:
                layers += [nn.Linear(size, u), nn.ELU()]
                size = u
            layers.append(nn.Linear(size, out_dim))
            self.aux_head = nn.Sequential(*layers)
            self.aux_target_norm = RunningMeanStd((out_dim,))
            self._latent = None
            self._aux_loss = None
            self.actor_mlp.register_forward_hook(lambda m, i, o: setattr(self, '_latent', o))

        def forward(self, obs_dict):
            out = super().forward(obs_dict)
            states = obs_dict.get('states', None)
            if states is not None and obs_dict.get('is_train', True) and self._latent is not None:
                target = states.float()
                if self.aux_indices.numel():
                    target = target[:, self.aux_indices]
                with torch.no_grad():
                    target = self.aux_target_norm(target)          # updates stats in train mode
                pred = self.aux_head(self._latent)
                self._aux_loss = self.aux_coef * torch.mean((pred - target) ** 2)
            return out

        def get_aux_loss(self):
            loss, self._aux_loss = self._aux_loss, None
            return {'state_aux': loss} if loss is not None else None
