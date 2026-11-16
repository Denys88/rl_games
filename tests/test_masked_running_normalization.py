"""Masked normalization tracks per-feature moments and valid-row counts."""

import pytest
import torch

from rl_games.algos_torch.running_mean_std import RunningMeanStd


SHAPES = [((4,), False), ((2, 3), False), ((3, 4), True), ((2, 3, 4), True)]


def normalizer(shape, per_channel, scripted):
    module = RunningMeanStd(shape, per_channel=per_channel, init_count=7)
    return torch.jit.script(module) if scripted else module


def assert_statistics_equal(actual, reference):
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value)


@pytest.mark.parametrize("shape,per_channel", SHAPES)
@pytest.mark.parametrize("scripted", [False, True])
@pytest.mark.parametrize("column_mask", [False, True])
@pytest.mark.parametrize("float_mask", [False, True])
def test_masked_statistics_match_valid_rows_per_feature(
        shape, per_channel, scripted, column_mask, float_mask):
    torch.manual_seed(29)
    masked = normalizer(shape, per_channel, scripted)
    reference = normalizer(shape, per_channel, scripted)
    valid_counts = 0
    for valid in [torch.tensor([True, False, True, True, False, False, True, False]),
                  torch.tensor([False, True, True, False, True, True, False, True])]:
        batch = torch.randn(8, *shape) + torch.arange(shape[0]).reshape(
            1, shape[0], *([1] * (len(shape) - 1))) * 10
        batch[~valid] = float("nan")  # Padding must never contaminate persistent statistics.
        mask = valid.reshape(-1, 1) if column_mask else valid
        if float_mask:
            mask = mask.float()
        result = masked(batch, mask=mask)
        expected = reference(batch[valid])
        valid_counts += int(valid.sum())
        assert masked.count.item() == 7 + valid_counts
        assert_statistics_equal(masked, reference)
        torch.testing.assert_close(result[valid], expected)
        assert torch.isfinite(masked.running_mean).all()
        assert torch.isfinite(masked.running_var).all()
    masked.eval()
    reference.eval()
    query = torch.randn(3, *shape)
    torch.testing.assert_close(masked(query), reference(query))
    torch.testing.assert_close(masked(query, denorm=True), reference(query, denorm=True))
    restored = normalizer(shape, per_channel, scripted)
    restored.load_state_dict(masked.state_dict())
    restored.eval()
    torch.testing.assert_close(restored(query), masked(query))


@pytest.mark.parametrize("shape,per_channel", [SHAPES[0], SHAPES[-1]])
@pytest.mark.parametrize("scripted", [False, True])
@pytest.mark.parametrize("valid_count", [0, 1])
def test_degenerate_row_masks_preserve_valid_statistics(shape, per_channel, scripted, valid_count):
    masked = normalizer(shape, per_channel, scripted)
    reference = normalizer(shape, per_channel, scripted)
    values = torch.full((6, *shape), float("nan"))
    mask = torch.arange(6) < valid_count
    values[mask] = 2
    masked(values, mask=mask)
    if valid_count:
        reference(values[mask])
    assert_statistics_equal(masked, reference)
    assert masked.count.item() == 7 + valid_count
    next_batch = torch.randn(4, *shape)
    torch.testing.assert_close(masked(next_batch), reference(next_batch))
    assert_statistics_equal(masked, reference)
