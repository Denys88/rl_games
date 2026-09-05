"""Masked diagnostics regressions (diagnostics-only paths, use_diagnostics)."""

import torch

from rl_games.algos_torch.torch_ext import explained_variance, policy_clip_fraction


def test_masked_diagnostics_match_unmasked_on_valid_rows():
    # masked explained_variance used Var(y_pred) where Var(y) belongs, and
    # masked clip fraction returned a length-N vector scaled by 1/N instead
    # of the scalar fraction -- both produced nonsense values
    torch.manual_seed(2)
    n_valid = 96
    y = torch.randn(128, 1) * 3 + 1
    y_pred = y + torch.randn(128, 1) * 0.5
    masks = torch.zeros(128)
    masks[:n_valid] = 1.0
    ev_masked = explained_variance(y_pred, y, masks)
    ev_direct = explained_variance(y_pred[:n_valid], y[:n_valid])
    assert torch.allclose(ev_masked, ev_direct, atol=1e-4), (ev_masked, ev_direct)

    new_nl = torch.randn(128)
    old_nl = new_nl + torch.randn(128) * 0.3
    cf_masked = policy_clip_fraction(new_nl, old_nl, 0.2, masks)
    cf_direct = policy_clip_fraction(new_nl[:n_valid], old_nl[:n_valid], 0.2)
    assert cf_masked.ndim == 0, 'clip fraction must be a scalar'
    assert torch.allclose(cf_masked, cf_direct, atol=1e-6), (cf_masked, cf_direct)
