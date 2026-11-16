"""Coordinate channels share one cached spatial grid across batch sizes."""

import pytest
import torch
import torch.nn.functional as F

from rl_games.algos_torch.torch_ext import CoordConv2d


@pytest.fixture(autouse=True)
def isolated_coordinate_cache():
    previous = CoordConv2d.pool
    CoordConv2d.pool = {}
    yield
    CoordConv2d.pool = previous


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (1, 5)])
@pytest.mark.parametrize("batch", [1, 4])
def test_coordinate_convolution_matches_explicit_grid_and_gradients(dtype, shape, batch):
    height, width = shape
    torch.manual_seed(7)
    layer = CoordConv2d(2, 3, kernel_size=3, padding=1).to(dtype=dtype)
    x = torch.randn(batch, 2, height, width, dtype=dtype, requires_grad=True)
    actual = layer(x)
    actual_grads = torch.autograd.grad(actual.square().sum(), (x, layer.weight, layer.bias))

    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
    grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)
    grid = grid.permute(0, 3, 1, 2).repeat(batch, 1, 1, 1).to(dtype=dtype)
    reference_x = x.detach().clone().requires_grad_()
    weight = layer.weight.detach().clone().requires_grad_()
    bias = layer.bias.detach().clone().requires_grad_()
    expected = F.conv2d(torch.cat([reference_x, grid], dim=1), weight, bias, padding=1)
    expected_grads = torch.autograd.grad(expected.square().sum(), (reference_x, weight, bias))
    torch.testing.assert_close(actual, expected)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 5), (84, 84)])
def test_coordinate_storage_is_independent_of_batch_size(dtype, shape):
    height, width = shape
    expected_bytes = 2 * height * width * torch.empty((), dtype=dtype).element_size()
    pointers = []
    for batch in (1, 8, 32):
        x = torch.zeros(1, 1, height, width, dtype=dtype).expand(batch, -1, -1, -1)
        coordinates = CoordConv2d.get_coord(x)
        assert coordinates.shape == (batch, 2, height, width)
        assert coordinates.untyped_storage().nbytes() == expected_bytes
        pointers.append(coordinates.untyped_storage().data_ptr())
        assert len(CoordConv2d.pool) == 1
    assert len(set(pointers)) == 1


def test_large_rollout_coordinates_do_not_allocate_one_grid_per_actor():
    x = torch.zeros(1, 1, 84, 84).expand(1024, -1, -1, -1)
    coordinates = CoordConv2d.get_coord(x)
    assert coordinates.shape == (1024, 2, 84, 84)
    assert coordinates.untyped_storage().nbytes() == 56_448


def test_coordinate_cache_keeps_dtype_and_device_separate():
    for dtype in (torch.float32, torch.float64):
        x = torch.zeros(4, 1, 3, 5, dtype=dtype)
        coordinates = CoordConv2d.get_coord(x)
        assert coordinates.dtype == x.dtype
        assert coordinates.device == x.device
    assert len(CoordConv2d.pool) == 2
