import pytest
import torch

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.spatial_softmax import SpatialSoftArgmax


@pytest.mark.parametrize("height,width", [(2, 5), (5, 2), (3, 3)])
@pytest.mark.parametrize("normalize", [False, True])
def test_spatial_softargmax_returns_row_column_peak_coordinates(height, width, normalize):
    features = torch.full((2, 2, height, width), -100.0)
    peaks = torch.tensor([[[0, 1], [height - 1, 0]], [[1, width - 1], [0, 0]]])
    for batch in range(2):
        for channel in range(2):
            row, column = peaks[batch, channel]
            features[batch, channel, row, column] = 100.0

    result = SpatialSoftArgmax(normalize=normalize)(features)

    expected = peaks.float()
    if normalize:
        expected = 2.0 * expected / torch.tensor([height - 1, width - 1]) - 1.0
    torch.testing.assert_close(result, expected.reshape(2, 4))


@pytest.mark.parametrize("normalize", [False, True])
def test_spatial_softargmax_matches_rectangular_weighted_mean(normalize):
    weights = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    features = weights.log().reshape(1, 1, 2, 3).requires_grad_()
    expected = torch.tensor([
        weights[1].sum() / weights.sum(),
        (weights[:, 1].sum() + 2 * weights[:, 2].sum()) / weights.sum(),
    ])
    if normalize:
        expected = 2.0 * expected / torch.tensor([1, 2]) - 1.0

    result = SpatialSoftArgmax(normalize=normalize)(features)

    torch.testing.assert_close(result, expected.reshape(1, 2))
    result.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0


def test_network_builder_spatial_softargmax_accepts_rectangular_features():
    network = NetworkBuilder.BaseNetwork()._build_conv(
        "conv2d_spatial_softargmax",
        input_shape=(1, 2, 3),
        convs=[],
        activation="relu",
    )
    features = torch.full((1, 1, 2, 3), -100.0)
    features[0, 0, 1, 1] = 100.0

    torch.testing.assert_close(network(features), torch.tensor([[1.0, 0.0]]))
