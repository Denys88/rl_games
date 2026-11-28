"""Omitted recurrent states behave like explicit zero initialization."""
import pytest
import torch
from torch import nn

from rl_games.algos_torch.network_builder import A2CBuilder, A2CResnetBuilder


@pytest.fixture(scope="module", autouse=True)
def small_network_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _network(architecture, recurrent, before_mlp=False):
    params = {
        "separate": architecture == "separate", "space": {"discrete": {}},
        "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
        "rnn": {"name": recurrent, "units": 8, "layers": 2, "before_mlp": before_mlp},
    }
    shape = (5,)
    builder = A2CBuilder()
    if architecture == "impala":
        params["cnn"] = {"conv_depths": [4], "permute_input": False}
        shape = (3, 8, 8)
        builder = A2CResnetBuilder()
    builder.load(params)
    return builder.build("a2c", actions_num=3, input_shape=shape, num_seqs=2), shape


@pytest.mark.parametrize("architecture", ["shared", "separate", "impala"])
@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
@pytest.mark.parametrize("with_dones", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_omitted_state_matches_zero_state_and_gradients(architecture, recurrent, with_dones, dtype):
    torch.manual_seed(7)
    network, shape = _network(architecture, recurrent, before_mlp=with_dones)
    network = network.to(dtype=dtype)
    # The runtime batch can differ from the num_seqs used to construct a policy.
    num_seqs, seq_length = 3, 4 if with_dones else 1
    observations = torch.randn(num_seqs * seq_length, *shape, dtype=dtype, requires_grad=True)
    inputs = {"obs": observations, "seq_length": seq_length}
    if with_dones:
        dones = torch.zeros(num_seqs, seq_length, dtype=dtype)
        dones[0, 0] = dones[1, 2] = 1
        inputs.update(dones=dones.flatten(), rnn_states=None)
    states = tuple(observations.new_zeros((state.shape[0], num_seqs, state.shape[2]))
                   for state in network.get_default_rnn_state())
    expected = network(dict(inputs, rnn_states=states))
    actual = network(inputs)
    torch.testing.assert_close(actual, expected)

    targets = tuple(network.parameters()) + (observations,)
    actual_grads = torch.autograd.grad(actual[0].square().mean() + actual[1].square().mean(), targets)
    expected_grads = torch.autograd.grad(expected[0].square().mean() + expected[1].square().mean(), targets)
    torch.testing.assert_close(actual_grads, expected_grads)

    # The returned state must remain usable by the next inference call.
    next_obs = torch.randn(num_seqs, *shape, dtype=dtype)
    continued = network({"obs": next_obs, "rnn_states": actual[2]})
    reference = network({"obs": next_obs, "rnn_states": expected[2]})
    torch.testing.assert_close(continued, reference)


class StatelessPolicyTrace(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        logits, values, _ = self.policy({"obs": observations})
        return logits, values


@pytest.mark.parametrize("architecture", ["shared", "separate", "impala"])
@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
def test_policy_tracing_initializes_state_for_the_runtime_batch(architecture, recurrent):
    network, shape = _network(architecture, recurrent)
    wrapper = StatelessPolicyTrace(network).eval()
    example = torch.randn(2, *shape)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
        observations = torch.randn(3, *shape)
        torch.testing.assert_close(traced(observations), wrapper(observations))
