import pytest
import torch
from torch import nn

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("separate", [False, True])
def test_batch_normalized_cnn_construction_does_not_train_on_shape_probe(
        dimensions, kernel_size, separate):
    params = {
        "separate": separate,
        "space": {"discrete": {}},
        "normalization": "batch_norm",
        "mlp": {"units": [8], "activation": "relu", "initializer": {"name": "default"}},
        "cnn": {
            "type": f"conv{dimensions}d", "activation": "relu",
            "initializer": {"name": "default"}, "permute_input": False,
            "convs": [{"filters": 4, "kernel_size": kernel_size, "strides": 1, "padding": 0}],
        },
    }
    shape = (1,) + (3,) * dimensions
    builder = A2CBuilder()
    builder.load(params)
    network = builder.build("a2c", actions_num=2, input_shape=shape, num_seqs=4)
    norms = [m for m in network.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for norm in norms:
        assert norm.training
        assert norm.num_batches_tracked.item() == 0
        torch.testing.assert_close(norm.running_mean, torch.zeros_like(norm.running_mean))
        torch.testing.assert_close(norm.running_var, torch.ones_like(norm.running_var))

    observations = torch.randn(4, *shape, requires_grad=True)
    logits, values, _ = network({"obs": observations})
    assert logits.shape == (4, 2)
    assert values.shape == (4, 1)
    (logits.square().mean() + values.square().mean()).backward()
    assert torch.isfinite(observations.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert all(norm.num_batches_tracked.item() == 1 for norm in norms)


def test_shape_probe_preserves_mixed_module_training_modes():
    layers = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout())
    layers[0].eval()
    states = [m.training for m in layers.modules()]
    assert NetworkBuilder.BaseNetwork()._calc_input_size((2, 3), layers) == 6
    assert [m.training for m in layers.modules()] == states
    assert layers[0].num_batches_tracked.item() == 0


def test_shape_probe_restores_training_modes_after_an_invalid_shape():
    layers = nn.Sequential(nn.BatchNorm1d(2), nn.Flatten(), nn.Linear(7, 3))
    states = [m.training for m in layers.modules()]
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes"):
        NetworkBuilder.BaseNetwork()._calc_input_size((2, 3), layers)
    assert [m.training for m in layers.modules()] == states
    assert layers[0].num_batches_tracked.item() == 0
