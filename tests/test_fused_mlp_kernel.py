"""Correctness tests for the experimental fused actor-critic MLP kernels.

Verifies single-kernel forward and single-kernel analytic backward against
PyTorch autograd on an identical network (IEEE fp32 dot precision).

Run: pytest tests/test_fused_mlp_kernel.py -v   (requires CUDA + triton)
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    triton_available = True
except ImportError:
    triton_available = False

pytestmark = pytest.mark.skipif(
    not (cuda_available and triton_available), reason='requires CUDA and triton')


@pytest.mark.parametrize('n', [64, 4097, 32768])
@pytest.mark.parametrize('d_in,actions,value_size', [(36, 8, 1), (30, 12, 1), (48, 6, 3)])
def test_fwd_bwd_parity(n, d_in, actions, value_size):
    from rl_games.triton_kernels.mlp_kernel import FusedACMLP

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        net = FusedACMLP(d_in, [256, 128, 64], actions, value_size, ieee=True, seed=n)
        ref = net.torch_reference()
        x = torch.randn(n, d_in, device='cuda')

        mu_f, v_f = net.forward(x)
        mu_r, v_r = ref(x)
        torch.testing.assert_close(mu_f, mu_r, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(v_f, v_r, atol=1e-5, rtol=1e-4)

        dmu = torch.randn_like(mu_r)
        dv = torch.randn_like(v_r)
        ((mu_r * dmu).sum() + (v_r * dv).sum()).backward()
        g = net.backward(x, dmu, dv)

        a, v = actions, value_size
        pairs = [
            (ref.l1.weight.grad.t(), g['w1']), (ref.l1.bias.grad, g['b1']),
            (ref.l2.weight.grad.t(), g['w2']), (ref.l2.bias.grad, g['b2']),
            (ref.l3.weight.grad.t(), g['w3']), (ref.l3.bias.grad, g['b3']),
            (ref.mu.weight.grad.t(), g['wm'][:, :a]), (ref.mu.bias.grad, g['bm'][:a]),
            (ref.value.weight.grad.t(), g['wv'][:, :v]), (ref.value.bias.grad, g['bv'][:v]),
        ]
        # atomics change reduction order; scale tolerance with batch size
        atol = 1e-4 * max(1.0, n / 4096)
        for r, f in pairs:
            torch.testing.assert_close(f, r, atol=atol, rtol=1e-3)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_logstd_grad_passthrough():
    from rl_games.triton_kernels.mlp_kernel import FusedACMLP

    n, a = 2048, 8
    net = FusedACMLP(36, [256, 128, 64], a, 1, ieee=True)
    x = torch.randn(n, 36, device='cuda')
    net.forward(x)
    dmu = torch.zeros(n, a, device='cuda')
    dv = torch.zeros(n, 1, device='cuda')
    dsigma = torch.randn(n, a, device='cuda')
    g = net.backward(x, dmu, dv, dsigma=dsigma)
    torch.testing.assert_close(g['logstd'], dsigma.sum(0), atol=1e-3, rtol=1e-4)
