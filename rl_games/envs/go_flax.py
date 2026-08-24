"""Flax twin of the Go ResNet (policy + value heads only).

Opponents and (later) search run inside the jitted pgx env loop, which is JAX —
so the torch training net (rl_games/envs/go_network.py) has a flax mirror here,
plus `params_from_torch` to convert a checkpoint state_dict, and
`make_flax_opponent` producing the jit-able opponent callable the env expects:

    (params, obs, mask, rng) -> (action, dist)

Aux heads are intentionally not mirrored: the jitted opponent step only needs
policy logits (and search will need value). The parity test
(tests/test_flax_parity.py) asserts torch and flax agree to <1e-4 on random
boards; it must stay green for every architecture change on either side.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class GoResNetFlax(nn.Module):
    """Mirror of go_network.GoResNetBuilder.Network (policy/value only)."""

    blocks: int = 6
    channels: int = 64
    gpool_every: int = 2
    value_units: int = 128

    @nn.compact
    def __call__(self, obs):
        # obs: (B, size, size, planes) float32, NHWC
        c = self.channels
        x = nn.Conv(c, (3, 3), name='stem')(obs)
        for i in range(self.blocks):
            has_gpool = self.gpool_every > 0 and (i + 1) % self.gpool_every == 0
            residual = x
            y = nn.Conv(c, (3, 3), name=f'b{i}_conv1')(jax.nn.relu(x))
            if has_gpool:
                pooled = jnp.concatenate([y.mean(axis=(1, 2)), y.max(axis=(1, 2))], axis=-1)
                y = y + nn.Dense(c, name=f'b{i}_gpool')(pooled)[:, None, None, :]
            y = nn.Conv(c, (3, 3), name=f'b{i}_conv2')(jax.nn.relu(y))
            x = residual + y
        trunk = jax.nn.relu(x)
        pooled = jnp.concatenate([trunk.mean(axis=(1, 2)), trunk.max(axis=(1, 2))], axis=-1)

        point_logits = nn.Conv(1, (1, 1), name='policy_conv')(trunk)
        point_logits = point_logits.reshape(point_logits.shape[0], -1)
        pass_logit = nn.Dense(1, name='policy_pass')(pooled)
        logits = jnp.concatenate([point_logits, pass_logit], axis=-1)

        v = nn.Dense(self.value_units, name='value_fc1')(pooled)
        v = nn.Dense(1, name='value_fc2')(jax.nn.relu(v))
        return logits, v[..., 0]


def _clean_key(key):
    for prefix in ('_orig_mod.', 'a2c_network.', 'model.'):
        key = key.replace(prefix, '')
    return key


def params_from_torch(state_dict, blocks=6, gpool_every=2, **_):
    """torch state_dict (from the model or the bare network) -> flax params.

    Handles the 'a2c_network.' model prefix and torch.compile's '_orig_mod.'.
    Conv weights OIHW -> HWIO, linear (out,in) -> (in,out). Aux-head and
    normalizer entries are ignored.
    """
    flat = {}
    for k, v in state_dict.items():
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        flat[_clean_key(k)] = np.asarray(v, dtype=np.float32)

    def conv(name):
        return {'kernel': flat[name + '.weight'].transpose(2, 3, 1, 0),
                'bias': flat[name + '.bias']}

    def dense(name):
        return {'kernel': flat[name + '.weight'].T, 'bias': flat[name + '.bias']}

    params = {
        'stem': conv('stem'),
        'policy_conv': conv('policy_conv'),
        'policy_pass': dense('policy_pass'),
        'value_fc1': dense('value_fc1'),
        'value_fc2': dense('value_fc2'),
    }
    for i in range(blocks):
        params[f'b{i}_conv1'] = conv(f'blocks.{i}.conv1')
        params[f'b{i}_conv2'] = conv(f'blocks.{i}.conv2')
        if gpool_every > 0 and (i + 1) % gpool_every == 0:
            params[f'b{i}_gpool'] = dense(f'blocks.{i}.gpool.fc')
    return jax.tree_util.tree_map(jnp.asarray, params)


def init_flax_params(net, seed=0, size=9, planes=17):
    dummy = jnp.zeros((1, size, size, planes), dtype=jnp.float32)
    return net.init(jax.random.PRNGKey(seed), dummy)['params']


def make_flax_opponent(blocks=6, channels=64, gpool_every=2, value_units=128,
                       temperature=1.0):
    """Jit-able opponent for PgxGoVecEnv: samples the flax policy.

    Operates on a single (unbatched) board — the env vmaps it. `dist` is the
    post-mask, post-temperature distribution actually sampled from, which is
    exactly what the opp_policy aux target wants.
    """
    net = GoResNetFlax(blocks=blocks, channels=channels,
                       gpool_every=gpool_every, value_units=value_units)

    def opponent_fn(params, obs, mask, rng):
        logits, _ = net.apply({'params': params}, obs[None].astype(jnp.float32))
        logits = logits[0] / jnp.maximum(temperature, 1e-6)
        logits = jnp.where(mask, logits, -jnp.inf)
        action = jax.random.categorical(rng, logits)
        dist = jax.nn.softmax(logits)
        return action, dist

    return opponent_fn, net
