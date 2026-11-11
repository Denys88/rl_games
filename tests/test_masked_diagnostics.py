"""Masked diagnostics regressions (diagnostics-only paths, use_diagnostics)."""

import torch

from rl_games.algos_torch.torch_ext import explained_variance, policy_clip_fraction


def _row_mask(n, n_valid):
    masks = torch.zeros(n)
    masks[:n_valid] = 1.0
    return masks


def test_masked_diagnostics_match_unmasked_on_valid_rows():
    # masked explained_variance used Var(y_pred) where Var(y) belongs, and
    # masked clip fraction returned a length-N vector scaled by 1/N instead
    # of the scalar fraction -- both produced nonsense values
    torch.manual_seed(2)
    n_valid = 96
    y = torch.randn(128, 1) * 3 + 1
    # Var(y_pred) must differ from Var(y) by O(1), or the old Var(y_pred)
    # code passes this test by accident
    y_pred = 0.3 * y + torch.randn(128, 1) * 0.5
    masks = _row_mask(128, n_valid)
    ev_masked = explained_variance(y_pred, y, masks)
    ev_direct = explained_variance(y_pred[:n_valid], y[:n_valid])
    assert torch.allclose(ev_masked, ev_direct, atol=1e-4), (ev_masked, ev_direct)

    new_nl = torch.randn(128)
    old_nl = new_nl + torch.randn(128) * 0.3
    cf_masked = policy_clip_fraction(new_nl, old_nl, 0.2, masks)
    cf_direct = policy_clip_fraction(new_nl[:n_valid], old_nl[:n_valid], 0.2)
    assert cf_masked.ndim == 0, 'clip fraction must be a scalar'
    assert torch.allclose(cf_masked, cf_direct, atol=1e-6), (cf_masked, cf_direct)


def test_masked_explained_variance_multi_value_matches_direct():
    # value_size > 1: the (N, 1) row mask broadcasts over V columns, so the
    # valid-element count is n_valid * V, not n_valid -- with distinct column
    # means the old row-count denominator reported EV > 1
    torch.manual_seed(3)
    n_valid = 200
    y = torch.randn(256, 2) + torch.tensor([0.0, 5.0])
    y_pred = 0.3 * y + torch.randn(256, 2) * 0.5
    masks = _row_mask(256, n_valid)
    ev_masked = explained_variance(y_pred, y, masks)
    ev_direct = explained_variance(y_pred[:n_valid], y[:n_valid])
    assert torch.allclose(ev_masked, ev_direct, atol=1e-4), (ev_masked, ev_direct)


def test_masked_clip_fraction_finite_on_all_invalid_mask():
    # every sibling masked mean clamps its denominator (apply_masks,
    # get_mean_var_with_masks, the KL reductions); an all-invalid minibatch
    # must not inject NaN into PpoDiagnostics' epoch mean
    torch.manual_seed(4)
    new_nl = torch.randn(64)
    old_nl = new_nl + torch.randn(64)
    cf = policy_clip_fraction(new_nl, old_nl, 0.2, torch.zeros(64))
    assert torch.isfinite(cf), cf
    assert cf.item() == 0.0
