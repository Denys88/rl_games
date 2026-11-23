"""Regression tests for network, normalization and logging defects found in a code
review (reproduced on torch 2.15 / numpy 2.5 / Python 3.13):

- LayerNorm2d used batch statistics with frozen gamma/beta (beta == 1); the conv1d layer norm raised
"""

import numpy as np
import pytest
import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder


def _base_network():
    return NetworkBuilder.BaseNetwork()


def _ppo_agent(tmp_path, **config_overrides):
    """Tiny CPU PPO agent on the shared fake env, able to run train_epoch()."""
    from tests.test_ppo_masking import make_ppo_agent

    agent, _ = make_ppo_agent(train_dir=str(tmp_path), **config_overrides)
    agent.vec_env.set_train_info = lambda frame, algo: None   # the shared fake env has no such hook
    return agent


# ---------------------------------------------------------------------------
# LayerNorm2d is a per-sample layer norm with learnable affine parameters
# ---------------------------------------------------------------------------

CONVS = [{'filters': 8, 'kernel_size': 3, 'strides': 1, 'padding': 1},
         {'filters': 6, 'kernel_size': 3, 'strides': 1, 'padding': 1}]


def _mixed_scale_batch(*shape):
    """Samples with very different magnitudes: batch statistics and per-sample
    statistics disagree strongly on such a batch."""
    torch.manual_seed(0)
    x = torch.randn(*shape)
    scale = torch.tensor([0.1, 1.0, 10.0, 100.0])[:shape[0]]
    return x * scale.view(-1, *([1] * (len(shape) - 1))) + scale.view(-1, *([1] * (len(shape) - 1)))


def test_layer_norm2d_output_does_not_depend_on_the_rest_of_the_batch():
    """The old layer normalized with mean/std over (batch, H, W) per channel and
    kept no running estimate, so a sample's output depended on its batch mates:
    a rollout batch and a training minibatch saw different functions."""
    ln = torch_ext.LayerNorm2d(3)
    x = _mixed_scale_batch(4, 3, 5, 5)

    full = ln(x)
    for i in range(x.size(0)):
        alone = ln(x[i:i + 1])
        assert torch.allclose(full[i:i + 1], alone, atol=1e-5), f'sample {i} depends on the batch'

    # per-sample statistics over (C, H, W): zero mean (the old beta of ones gave 1.0), unit std
    flat = full.flatten(1)
    assert torch.allclose(flat.mean(dim=1), torch.zeros(4), atol=1e-4)
    assert torch.allclose(flat.std(dim=1), torch.ones(4), atol=1e-2)
    # and no train/eval difference
    assert torch.equal(ln.eval()(x), full)


def test_layer_norm2d_affine_parameters_are_learnable():
    """gamma/beta were registered as buffers: never touched by the optimizer."""
    ln = torch_ext.LayerNorm2d(3)
    params = dict(ln.named_parameters())
    assert set(params) == {'weight', 'bias'}
    assert all(p.requires_grad for p in params.values())
    assert torch.equal(params['weight'].detach(), torch.ones(3))
    assert torch.equal(params['bias'].detach(), torch.zeros(3))
    assert dict(ln.named_buffers()) == {}

    (ln(_mixed_scale_batch(4, 3, 5, 5)) * torch.randn(4, 3, 5, 5)).sum().backward()
    assert params['weight'].grad is not None and params['weight'].grad.abs().sum() > 0
    assert params['bias'].grad is not None and params['bias'].grad.abs().sum() > 0


def test_cnn2d_layer_norm_is_built_from_layer_norm2d():
    cnn = _base_network()._build_cnn2d(
        input_shape=(3, 8, 8), convs=CONVS, activation='relu', norm_func_name='layer_norm')
    norms = [m for m in cnn if isinstance(m, torch_ext.LayerNorm2d)]
    assert [m.num_channels for m in norms] == [8, 6]

    x = _mixed_scale_batch(4, 3, 8, 8)
    full = cnn(x)
    assert full.shape == (4, 6, 8, 8)
    assert torch.allclose(full[2:3], cnn(x[2:3]), atol=1e-5)


def test_layer_norm2d_loads_checkpoints_with_legacy_gamma_beta_buffers():
    """Checkpoints written by the old layer hold 'gamma'/'beta' buffers of
    shape [C, 1, 1] and no 'weight'/'bias'. They must still load (strict), with
    the affine terms carried over: gamma -> weight, beta -> bias."""
    def build():
        return _base_network()._build_cnn2d(
            input_shape=(3, 8, 8), convs=CONVS, activation='relu', norm_func_name='layer_norm')

    torch.manual_seed(1)
    source = build()
    legacy = {}
    for key, value in source.state_dict().items():
        module = source[int(key.split('.')[0])]
        if isinstance(module, torch_ext.LayerNorm2d):
            channels = module.num_channels
            # distinct values so a swapped or dropped mapping cannot pass
            # (the old layer always stored ones for both)
            legacy[key.replace('weight', 'gamma').replace('bias', 'beta')] = (
                torch.full((channels, 1, 1), 2.0 if key.endswith('weight') else 3.0))
        else:
            legacy[key] = value.clone()
    assert any(k.endswith('gamma') for k in legacy) and not any(
        k.endswith(('2.weight', '2.bias', '5.weight', '5.bias')) for k in legacy)

    target = build()
    snapshot = dict(legacy)
    with pytest.warns(UserWarning, match='LayerNorm2d'):
        result = target.load_state_dict(legacy)             # strict
    assert not result.missing_keys and not result.unexpected_keys
    assert legacy.keys() == snapshot.keys()                 # caller's dict is left alone

    for module in target:
        if isinstance(module, torch_ext.LayerNorm2d):
            assert torch.equal(module.weight.detach(), torch.full((module.num_channels,), 2.0))
            assert torch.equal(module.bias.detach(), torch.full((module.num_channels,), 3.0))
    for i in (0, 3):                                        # conv weights came through untouched
        assert torch.equal(target[i].weight, source[i].weight)

    # checkpoints written by the new layer round-trip without a warning
    fresh = build()
    fresh.load_state_dict(target.state_dict())
    assert torch.equal(fresh[2].bias, target[2].bias)


def test_cnn1d_layer_norm_normalizes_channels_for_any_length():
    """torch.nn.LayerNorm(C) on a [B, C, L] tensor normalizes the LAST dim and
    raised unless L == C."""
    cnn = _base_network()._build_cnn1d(
        input_shape=(4, 20), convs=CONVS, activation='relu', norm_func_name='layer_norm')
    x = _mixed_scale_batch(4, 4, 20)

    full = cnn(x)                                           # pre-fix: RuntimeError (L=20, C=8)
    assert full.shape == (4, 6, 20)
    for i in range(x.size(0)):
        assert torch.allclose(full[i:i + 1], cnn(x[i:i + 1]), atol=1e-5)
    norm = cnn[2]
    assert norm.weight.shape == (8,) and norm.weight.requires_grad


