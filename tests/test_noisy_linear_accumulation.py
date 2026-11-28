import pytest
import torch

from rl_games.algos_torch.layers import NoisyLinear


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_shape", [(4, 3), (2, 4, 3)])
def test_each_forward_keeps_its_noise_for_accumulated_gradients(dtype, bias, input_shape):
    layer = NoisyLinear(3, 2, bias=bias).to(dtype)
    first_input = torch.arange(1, 1 + torch.tensor(input_shape).prod().item(), dtype=dtype).reshape(input_shape)
    second_input = first_input.flip(-1) / 3
    torch.manual_seed(1783)

    first = layer(first_input)
    first_weight_noise = layer.epsilon_weight.clone()
    first_bias_noise = layer.epsilon_bias.clone() if bias else None
    second = layer(second_input)
    second_weight_noise = layer.epsilon_weight.clone()
    second_bias_noise = layer.epsilon_bias.clone() if bias else None
    first_grad = torch.arange(1, first.numel() + 1, dtype=dtype).reshape_as(first) / 7
    second_grad = -first_grad.flip(-1) / 2

    (first.mul(first_grad).sum() + second.mul(second_grad).sum()).backward()

    first_weight_grad = first_grad.reshape(-1, 2).T @ first_input.reshape(-1, 3)
    second_weight_grad = second_grad.reshape(-1, 2).T @ second_input.reshape(-1, 3)
    torch.testing.assert_close(layer.weight.grad, first_weight_grad + second_weight_grad)
    torch.testing.assert_close(
        layer.sigma_weight.grad,
        first_weight_grad * first_weight_noise + second_weight_grad * second_weight_noise,
    )
    if bias:
        first_bias_grad = first_grad.reshape(-1, 2).sum(0)
        second_bias_grad = second_grad.reshape(-1, 2).sum(0)
        torch.testing.assert_close(layer.bias.grad, first_bias_grad + second_bias_grad)
        torch.testing.assert_close(
            layer.sigma_bias.grad,
            first_bias_grad * first_bias_noise + second_bias_grad * second_bias_noise,
        )


@pytest.mark.parametrize("bias", [False, True])
def test_inference_does_not_replace_noise_saved_for_backward(bias):
    layer = NoisyLinear(3, 2, bias=bias).double()
    inputs = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=torch.float64)
    torch.manual_seed(93)
    output = layer(inputs)
    weight_noise = layer.epsilon_weight.clone()
    bias_noise = layer.epsilon_bias.clone() if bias else None

    with torch.no_grad():
        layer(inputs * 2)
    output.sum().backward()

    torch.testing.assert_close(layer.sigma_weight.grad, inputs.sum(0) * weight_noise)
    if bias:
        torch.testing.assert_close(layer.sigma_bias.grad, inputs.shape[0] * bias_noise)
