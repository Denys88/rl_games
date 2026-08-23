"""Drop-in parity: fused_mlp_actor_critic vs the standard actor_critic network.

Builds both networks from the same config, copies weights across, and checks
outputs (mu, logstd, value) and every parameter gradient match.
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


NET_PARAMS = {
    'name': 'actor_critic',
    'separate': False,
    'space': {'continuous': {
        'mu_activation': 'None', 'sigma_activation': 'None',
        'mu_init': {'name': 'default'},
        'sigma_init': {'name': 'const_initializer', 'val': 0.3},
        'fixed_sigma': True}},
    'mlp': {'units': [256, 128, 64], 'activation': 'elu',
            'initializer': {'name': 'default'}},
}

BUILD_KWARGS = dict(input_shape=(36,), actions_num=8, value_size=1, num_seqs=4)


def build_pair():
    from rl_games.algos_torch.network_builder import A2CBuilder
    from rl_games.algos_torch.fused_mlp_network import FusedMLPA2CBuilder

    ref_b = A2CBuilder()
    ref_b.load({**NET_PARAMS, 'name': 'actor_critic'})
    ref = ref_b.build('a2c', **BUILD_KWARGS).cuda()

    fused_b = FusedMLPA2CBuilder()
    fused_b.load({**NET_PARAMS, 'name': 'fused_mlp_actor_critic',
                  'ieee_precision': True})
    fused = fused_b.build('a2c', **BUILD_KWARGS).cuda()

    # copy ref weights into fused (transposed storage)
    linears = [m for m in ref.actor_mlp if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    with torch.no_grad():
        for (w, b), lin in zip([(fused.w1, fused.b1), (fused.w2, fused.b2),
                                (fused.w3, fused.b3)], linears):
            w.copy_(lin.weight.t())
            b.copy_(lin.bias)
        fused.wm.copy_(ref.mu.weight.t())
        fused.bm.copy_(ref.mu.bias)
        fused.wv.copy_(ref.value.weight.t())
        fused.bv.copy_(ref.value.bias)
        fused.logstd.copy_(ref.sigma)
    return ref, fused


def test_forward_parity():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref, fused = build_pair()
        obs = torch.randn(4097, 36, device='cuda')
        mu_r, logstd_r, v_r, _ = ref({'obs': obs})
        mu_f, logstd_f, v_f, st = fused({'obs': obs})
        assert st is None
        torch.testing.assert_close(mu_f, mu_r, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(v_f, v_r, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(logstd_f, logstd_r, atol=1e-6, rtol=1e-6)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_grad_parity():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref, fused = build_pair()
        obs = torch.randn(4097, 36, device='cuda')
        gmu = torch.randn(4097, 8, device='cuda')
        gv = torch.randn(4097, 1, device='cuda')
        gs = torch.randn(4097, 8, device='cuda')

        mu_r, logstd_r, v_r, _ = ref({'obs': obs})
        ((mu_r * gmu).sum() + (v_r * gv).sum() + (logstd_r * gs).sum()).backward()
        mu_f, logstd_f, v_f, _ = fused({'obs': obs})
        ((mu_f * gmu).sum() + (v_f * gv).sum() + (logstd_f * gs).sum()).backward()

        linears = [m for m in ref.actor_mlp if isinstance(m, torch.nn.Linear)]
        pairs = [
            ('w1', linears[0].weight.grad.t(), fused.w1.grad),
            ('b1', linears[0].bias.grad, fused.b1.grad),
            ('w2', linears[1].weight.grad.t(), fused.w2.grad),
            ('b2', linears[1].bias.grad, fused.b2.grad),
            ('w3', linears[2].weight.grad.t(), fused.w3.grad),
            ('b3', linears[2].bias.grad, fused.b3.grad),
            ('wm', ref.mu.weight.grad.t(), fused.wm.grad),
            ('bm', ref.mu.bias.grad, fused.bm.grad),
            ('wv', ref.value.weight.grad.t(), fused.wv.grad),
            ('bv', ref.value.bias.grad, fused.bv.grad),
            ('logstd', ref.sigma.grad, fused.logstd.grad),
        ]
        for name, r, f in pairs:
            torch.testing.assert_close(f, r, atol=2e-4, rtol=1e-3,
                                       msg=lambda m, n=name: f'{n}: {m}')
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_model_builder_registration():
    from rl_games.algos_torch.model_builder import ModelBuilder

    params = {
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {**NET_PARAMS, 'name': 'fused_mlp_actor_critic'},
    }
    model = ModelBuilder().load(params)
    net = model.build({'input_shape': (36,), 'actions_num': 8, 'value_size': 1,
                       'num_seqs': 4, 'normalize_value': True,
                       'normalize_input': True}).cuda()
    obs = torch.randn(512, 36, device='cuda')
    out = net({'is_train': True, 'obs': obs,
               'prev_actions': torch.randn(512, 8, device='cuda')})
    assert out['mus'].shape == (512, 8)
    assert out['values'].shape == (512, 1)
    assert out['prev_neglogp'].shape == (512,)
    assert torch.isfinite(out['prev_neglogp']).all()
