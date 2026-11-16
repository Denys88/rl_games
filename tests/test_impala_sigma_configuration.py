"""Continuous Impala policies honor the configured sigma transform."""

import pytest
import torch

from rl_games.algos_torch.model_builder import ModelBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("parametrization", ["default", "exp", "softplus", "linear", "scalar"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("raw", [-4.0, 0.25, 4.0])
@pytest.mark.parametrize("capped", [False, True])
def test_impala_sigma_configuration_controls_distribution(parametrization, fixed, raw, capped):
    space = {
        "mu_activation": "None", "sigma_activation": "None",
        "mu_init": {"name": "default"},
        "sigma_init": {"name": "const_initializer", "val": raw},
        "fixed_sigma": fixed, "min_sigma": 0.2, "max_sigma": 2.0 if capped else 0.0,
    }
    if parametrization != "default":
        space.update(sigma_parametrization=parametrization, logstd_bounds=[-2.0, 1.0])
    params = {
        "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "resnet_actor_critic", "space": {"continuous": space},
            "cnn": {"conv_depths": [4], "permute_input": False},
            "mlp": {"units": [8], "activation": "relu", "initializer": {"name": "default"}},
        },
    }
    model = ModelBuilder().load(params).build({
        "actions_num": 2, "input_shape": (3, 8, 8), "num_seqs": 6, "value_size": 1,
        "normalize_value": False, "normalize_input": False,
    })
    inputs = {"obs": torch.randn(6, 3, 8, 8), "is_train": True,
              "prev_actions": torch.zeros(6, 2), "rnn_states": None}
    result = model(dict(inputs))
    raw_reference = torch.full_like(result["sigmas"], raw, requires_grad=True)
    if parametrization == "softplus":
        expected = torch.nn.functional.softplus(raw_reference) + 0.2
    elif parametrization in ("linear", "scalar"):
        expected = 0.2 + torch.nn.functional.softplus(raw_reference - 0.2)
    else:
        logstd = raw_reference if parametrization == "default" else raw_reference.clamp(-2.0, 1.0)
        expected = logstd.exp() + 0.2
    if capped:
        excess = expected - 0.2
        expected = 0.2 + 1.8 * excess / (1.8 + excess)
    torch.testing.assert_close(result["sigmas"], expected)
    distribution = torch.distributions.Normal(result["mus"], expected)
    torch.testing.assert_close(result["prev_neglogp"], -distribution.log_prob(inputs["prev_actions"]).sum(-1))
    torch.testing.assert_close(result["entropy"], distribution.entropy().sum(-1))

    result["sigmas"].sum().backward()
    expected.sum().backward()
    sigma = model.a2c_network.sigma
    actual_gradient = sigma.grad if fixed else sigma.bias.grad
    torch.testing.assert_close(actual_gradient, raw_reference.grad.sum(0))
    model.zero_grad()
    trained = model(dict(inputs))
    (trained["prev_neglogp"].mean() + trained["values"].square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
