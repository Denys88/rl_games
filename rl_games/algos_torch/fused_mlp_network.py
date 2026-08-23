"""Custom network: fused actor-critic MLP running on single Triton kernels.

Registered as network name ``fused_mlp_actor_critic``. Drop-in replacement for
the standard ``actor_critic`` MLP network for the common GPU case:

    network:
      name: fused_mlp_actor_critic
      space:
        continuous:
          fixed_sigma: True
          sigma_init: {name: const_initializer, val: 0}
      mlp:
        units: [256, 128, 64]   # exactly 3 layers
        activation: elu

The whole trunk + mu/value heads execute as ONE Triton forward kernel and ONE
analytic backward kernel (see rl_games/triton_kernels/mlp_kernel.py). Together
with the fused PPO loss the minibatch update becomes a 4-kernel pipeline.

Supported: flat observations, ELU, exactly 3 mlp layers, fixed_sigma,
separate=False, continuous space. Input/value normalization work unchanged
(they live in the model wrapper). Anything else -> use ``actor_critic``.

Weights are stored transposed ([in, out]) relative to nn.Linear; this network
defines its own state_dict layout, so checkpoints are self-consistent.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rl_games.algos_torch import network_builder


def _next_pow2(x):
    import triton
    return max(16, triton.next_power_of_2(x))


class _FusedMLPFunction(torch.autograd.Function):
    """Autograd binding: 1 kernel forward, 1 kernel backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, obs, w1, b1, w2, b2, w3, b3, wm, bm, wv, bv,
                actions_num, value_size, ieee):
        import triton
        from rl_games.triton_kernels.mlp_kernel import _ac_mlp_fwd_kernel

        obs = obs.contiguous()
        n, d = obs.shape
        h1, h2, h3 = w1.shape[1], w2.shape[1], w3.shape[1]
        a_pad, v_pad = wm.shape[1], wv.shape[1]
        d_pad = _next_pow2(d)
        block_k = min(32, d_pad)

        dev = obs.device
        h1s = torch.empty(n, h1, device=dev)
        h2s = torch.empty(n, h2, device=dev)
        h3s = torch.empty(n, h3, device=dev)
        mu = torch.empty(n, actions_num, device=dev)
        value = torch.empty(n, value_size, device=dev)

        grid = (triton.cdiv(n, 64),)
        _ac_mlp_fwd_kernel[grid](
            obs, w1, b1, w2, b2, w3, b3, wm, bm, wv, bv,
            h1s, h2s, h3s, mu, value,
            n, d, d_pad, h1, h2, h3,
            actions_num, a_pad, value_size, v_pad,
            64, block_k, ieee,
            num_warps=8, num_stages=1,
        )

        ctx.save_for_backward(obs, w2, w3, wm, wv, h1s, h2s, h3s,
                              w1, b1, b2, b3, bm, bv)
        ctx.cfg = (n, d, d_pad, h1, h2, h3, actions_num, a_pad,
                   value_size, v_pad, ieee)
        return mu, value

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dmu, dvalue):
        import triton
        from rl_games.triton_kernels.mlp_kernel import _ac_mlp_bwd_kernel

        (obs, w2, w3, wm, wv, h1s, h2s, h3s,
         w1, b1, b2, b3, bm, bv) = ctx.saved_tensors
        (n, d, d_pad, h1, h2, h3, actions_num, a_pad,
         value_size, v_pad, ieee) = ctx.cfg

        dw1 = torch.zeros_like(w1)
        db1 = torch.zeros_like(b1)
        dw2 = torch.zeros_like(w2)
        db2 = torch.zeros_like(b2)
        dw3 = torch.zeros_like(w3)
        db3 = torch.zeros_like(b3)
        dwm = torch.zeros_like(wm)
        dbm = torch.zeros_like(bm)
        dwv = torch.zeros_like(wv)
        dbv = torch.zeros_like(bv)
        dummy = dmu  # unused sigma slot

        grid = (triton.cdiv(n, 32),)
        _ac_mlp_bwd_kernel[grid](
            obs, h1s, h2s, h3s,
            w2, w3, wm, wv,
            dmu.contiguous(), dvalue.contiguous(), dummy,
            dw1, db1, dw2, db2, dw3, db3,
            dwm, dbm, dwv, dbv, dummy,
            n, d, d_pad, h1, h2, h3,
            actions_num, a_pad, value_size, v_pad,
            32, 64, ieee, False,
            num_warps=4, num_stages=1,
        )

        return (None, dw1, db1, dw2, db2, dw3, db3, dwm, dbm, dwv, dbv,
                None, None, None)


class FusedMLPA2CBuilder(network_builder.NetworkBuilder):

    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return FusedMLPA2CBuilder.Network(self.params, **kwargs)

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)

            input_shape = kwargs.pop('input_shape')
            self.actions_num = kwargs.pop('actions_num')
            self.value_size = kwargs.pop('value_size', 1)

            assert len(input_shape) == 1, \
                'fused_mlp_actor_critic requires flat observations'
            assert not params.get('separate', False), \
                'fused_mlp_actor_critic requires separate: False'
            mlp = params['mlp']
            units = mlp['units']
            assert len(units) == 3, \
                'fused_mlp_actor_critic requires exactly 3 mlp layers'
            assert mlp.get('activation', 'elu') == 'elu', \
                'fused_mlp_actor_critic requires elu activation'
            space = params['space']['continuous']
            assert space.get('fixed_sigma', True), \
                'fused_mlp_actor_critic requires fixed_sigma: True'

            self.d_in = input_shape[0]
            self.units = list(units)
            self.a_pad = _next_pow2(self.actions_num)
            self.v_pad = _next_pow2(self.value_size)
            self.ieee = bool(params.get('ieee_precision', False))

            def make_linear(d_in, d_out):
                # nn.Linear default init (= rl_games 'default' initializer),
                # stored transposed [in, out] for the kernels
                lin = nn.Linear(d_in, d_out)
                w = nn.Parameter(lin.weight.detach().t().contiguous())
                b = nn.Parameter(lin.bias.detach().clone())
                return w, b

            h1, h2, h3 = units
            self.w1, self.b1 = make_linear(self.d_in, h1)
            self.w2, self.b2 = make_linear(h1, h2)
            self.w3, self.b3 = make_linear(h2, h3)
            self.wm, self.bm = make_linear(h3, self.actions_num)
            self.wv, self.bv = make_linear(h3, self.value_size)

            sigma_init = space.get('sigma_init', {})
            sigma_val = float(sigma_init.get('val', 0.0))
            self.logstd = nn.Parameter(
                torch.full((self.actions_num,), sigma_val, dtype=torch.float32))

        @torch.compiler.disable
        def forward(self, input_dict):
            obs = input_dict['obs']
            # pad head params on the out-dim for tl.dot (F.pad is differentiable,
            # so padded-lane grads are sliced away by autograd)
            wm = F.pad(self.wm, (0, self.a_pad - self.actions_num))
            bm = F.pad(self.bm, (0, self.a_pad - self.actions_num))
            wv = F.pad(self.wv, (0, self.v_pad - self.value_size))
            bv = F.pad(self.bv, (0, self.v_pad - self.value_size))

            mu, value = _FusedMLPFunction.apply(
                obs, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                wm, bm, wv, bv, self.actions_num, self.value_size, self.ieee)

            logstd = mu * 0.0 + self.logstd
            return mu, logstd, value, None

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def get_aux_loss(self):
            return None

        def get_value_layer(self):
            return None
