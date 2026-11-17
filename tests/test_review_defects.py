"""Regression tests for defects found in a code review (reproduced on
torch 2.15 / numpy 2.5 / Python 3.13). One section per defect, numbered as in
the review:

1. _build_sequential_mlp: stale in_size with norm_only_first_layer
"""

import threading

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
# 1. _build_sequential_mlp: in_size must advance on every layer
# ---------------------------------------------------------------------------

UNITS = [256, 128, 64]
CHAINED = [(10, 256), (256, 128), (128, 64)]
MLP_NORMS = (torch.nn.LayerNorm, torch.nn.BatchNorm1d)


def _linear_sizes(seq):
    return [(m.in_features, m.out_features) for m in seq if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize('norm', ['layer_norm', 'batch_norm'])
def test_mlp_norm_only_first_layer_chains_non_uniform_units(norm):
    """`in_size = unit` sat after `if not need_norm: continue`, so once the
    first layer had been normalized every later Linear kept the first layer's
    width as its input: units [256, 128, 64] built Linear(256, 64) and the
    forward raised 'mat1 and mat2 shapes cannot be multiplied (3x128 and
    256x64)'. Uniform units hid it."""
    mlp = _base_network()._build_sequential_mlp(
        10, UNITS, 'relu', torch.nn.Linear,
        norm_only_first_layer=True, norm_func_name=norm)

    assert _linear_sizes(mlp) == CHAINED
    assert mlp(torch.randn(3, 10)).shape == (3, 64)
    # the option still does what it says: one norm, right after the first activation
    norms = [i for i, m in enumerate(mlp) if isinstance(m, MLP_NORMS)]
    assert norms == [2]


@pytest.mark.parametrize('norm_only_first_layer', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
def test_mlp_layer_sizes_chain_for_every_norm_setting(norm_only_first_layer, norm):
    """The settings that already worked keep their layer layout."""
    mlp = _base_network()._build_sequential_mlp(
        10, UNITS, 'relu', torch.nn.Linear,
        norm_only_first_layer=norm_only_first_layer, norm_func_name=norm)

    assert _linear_sizes(mlp) == CHAINED
    assert mlp(torch.randn(3, 10)).shape == (3, 64)
    expected_norms = 0 if norm is None else (1 if norm_only_first_layer else len(UNITS))
    assert sum(isinstance(m, MLP_NORMS) for m in mlp) == expected_norms


