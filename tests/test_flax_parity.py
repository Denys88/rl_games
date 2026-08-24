"""torch <-> flax parity for the Go ResNet.

The flax twin (opponents, search) must match the torch training net to float
error on the policy logits and value. Runs on random boards; must stay green
for every architecture change on either side and on every checkpoint export.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from rl_games.envs.go_network import GoResNetBuilder
from rl_games.envs.go_flax import (GoResNetFlax, params_from_torch,
                                   make_flax_opponent, init_flax_params)

ARCH = dict(blocks=6, channels=64, gpool_every=2, value_units=128)


def _build_torch_net(seed=0):
    torch.manual_seed(seed)
    builder = GoResNetBuilder()
    builder.load(dict(ARCH))
    net = builder.build('go_resnet', input_shape=(9, 9, 17), actions_num=82)
    # conv2 layers are zero-init; randomize them so parity is a real test
    with torch.no_grad():
        for name, p in net.named_parameters():
            if 'conv2' in name:
                p.normal_(0, 0.05)
    return net.eval()


@pytest.fixture(scope='module')
def nets():
    tnet = _build_torch_net()
    fnet = GoResNetFlax(**ARCH)
    fparams = params_from_torch(tnet.state_dict(), **ARCH)
    return tnet, fnet, fparams


def test_parity_on_random_boards(nets):
    tnet, fnet, fparams = nets
    rng = np.random.RandomState(0)
    obs = (rng.rand(256, 9, 9, 17) < 0.3).astype(np.float32)

    with torch.no_grad():
        t_logits, t_value, _ = tnet({'obs': torch.from_numpy(obs)})
    # full f32 (JAX defaults to TF32 convs on Ampere+, ~1e-3 noise)
    with jax.default_matmul_precision('float32'):
        f_logits, f_value = fnet.apply({'params': fparams}, jnp.asarray(obs))

    dl = np.abs(t_logits.numpy() - np.asarray(f_logits)).max()
    dv = np.abs(t_value.numpy()[:, 0] - np.asarray(f_value)).max()
    assert dl < 1e-4, f'policy logits diverge: {dl}'
    assert dv < 1e-4, f'value diverges: {dv}'


def test_parity_with_model_prefixes(nets):
    tnet, fnet, _ = nets
    # simulate algo.get_weights(): a2c_network. prefix + unrelated entries
    sd = {'a2c_network.' + k: v for k, v in tnet.state_dict().items()}
    sd['value_mean_std.running_mean'] = torch.zeros(1)
    fparams = params_from_torch(sd, **ARCH)
    obs = np.zeros((4, 9, 9, 17), dtype=np.float32)
    with torch.no_grad():
        t_logits, _, _ = tnet({'obs': torch.from_numpy(obs)})
    with jax.default_matmul_precision('float32'):
        f_logits, _ = fnet.apply({'params': fparams}, jnp.asarray(obs))
    assert np.abs(t_logits.numpy() - np.asarray(f_logits)).max() < 1e-4


def test_flax_opponent_samples_legal(nets):
    tnet, _, fparams = nets
    opponent_fn, _ = make_flax_opponent(**ARCH)
    rng = np.random.RandomState(1)
    obs = (rng.rand(9, 9, 17) < 0.3).astype(np.float32)
    mask = np.zeros(82, dtype=bool)
    legal = [3, 17, 40, 81]
    mask[legal] = True
    for i in range(20):
        a, dist = opponent_fn(fparams, jnp.asarray(obs), jnp.asarray(mask),
                              jax.random.PRNGKey(i))
        assert int(a) in legal
        d = np.asarray(dist)
        assert abs(d.sum() - 1.0) < 1e-5
        assert d[~mask].max() == 0.0


def test_export_speed(nets):
    import time
    tnet, _, _ = nets
    sd = tnet.state_dict()
    t0 = time.perf_counter()
    p = params_from_torch(sd, **ARCH)
    jax.block_until_ready(p)
    dt = time.perf_counter() - t0
    assert dt < 1.0, f'export too slow: {dt*1000:.0f} ms'
