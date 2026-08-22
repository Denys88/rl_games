"""Numerical parity tests: fused Triton PPO loss vs the eager rl_games path.

The reference below is built from the exact production code
(common_losses.actor_loss / critic_loss, torch_ext.apply_masks,
torch_ext.policy_kl and the model's neglogp / Normal entropy), so passing
these tests means the fused kernel is a drop-in replacement.

Run: pytest tests/test_fused_ppo_loss.py -v   (requires CUDA + triton)
"""

import itertools
import math

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

DEVICE = 'cuda'


def make_inputs(n=4096, num_actions=8, value_size=1, with_mask=False, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, device=DEVICE)

    # realistic PPO regime: actions sampled from the old policy, new policy a
    # small perturbation of it (a few gradient steps away)
    old_mu = rand(n, num_actions)
    old_sigma = (0.3 * rand(n, num_actions)).exp()
    actions = old_mu + old_sigma * rand(n, num_actions)
    mu = (old_mu + 0.1 * rand(n, num_actions)).requires_grad_(True)
    sigma = (old_sigma * (0.1 * rand(n, num_actions)).exp()).clone().requires_grad_(True)
    values = rand(n, value_size).requires_grad_(True)
    with torch.no_grad():
        old_neglogp = (0.5 * (((actions - old_mu) / old_sigma) ** 2).sum(-1)
                       + 0.5 * math.log(2 * math.pi) * num_actions
                       + old_sigma.log().sum(-1))
    advantage = rand(n)
    old_values = values.detach() + 0.2 * rand(n, value_size)
    returns = old_values + rand(n, value_size)
    rnn_masks = (torch.rand(n, generator=g, device=DEVICE) > 0.3).float() if with_mask else None
    return dict(mu=mu, sigma=sigma, values=values, actions=actions,
                old_neglogp=old_neglogp, advantage=advantage,
                old_mu=old_mu, old_sigma=old_sigma,
                old_values=old_values, returns=returns, rnn_masks=rnn_masks)


def reference_loss(inp, e_clip, critic_coef, entropy_coef, bounds_coef,
                   is_ppo, use_smooth_clamp, clip_value, bound_loss_type,
                   has_value_loss):
    """Mirror of A2CAgent.calc_losses + models neglogp/entropy + policy_kl."""
    from rl_games.common import common_losses
    from rl_games.algos_torch import torch_ext

    mu, sigma, values = inp['mu'], inp['sigma'], inp['values']
    actions = inp['actions']

    neglogp = (0.5 * (((actions - mu) / sigma) ** 2).sum(dim=-1)
               + 0.5 * math.log(2 * math.pi) * actions.size(-1)
               + sigma.log().sum(dim=-1))
    distr = torch.distributions.Normal(mu, sigma, validate_args=False)
    entropy = distr.entropy().sum(dim=-1)

    loss_func = common_losses.smoothed_actor_loss if use_smooth_clamp else common_losses.actor_loss
    a_loss = loss_func(inp['old_neglogp'], neglogp, inp['advantage'], is_ppo, e_clip)
    if has_value_loss:
        c_loss = common_losses.default_critic_loss(
            inp['old_values'], values, e_clip, inp['returns'], clip_value)
    else:
        c_loss = torch.zeros(1, device=DEVICE)
    if bound_loss_type == 'regularisation':
        b_loss = (mu * mu).sum(dim=-1)
    elif bound_loss_type == 'bound':
        soft_bound = 1.1
        b_loss = (torch.clamp_min(mu - soft_bound, 0.0) ** 2
                  + torch.clamp_max(mu + soft_bound, 0.0) ** 2).sum(dim=-1)
    else:
        b_loss = torch.zeros(mu.size(0), device=DEVICE)

    losses, sum_mask = torch_ext.apply_masks(
        [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
        inp['rnn_masks'])
    a_loss, c_loss, entropy, b_loss = losses
    loss = a_loss + 0.5 * c_loss * critic_coef - entropy * entropy_coef + b_loss * bounds_coef

    with torch.no_grad():
        kl = torch_ext.policy_kl(mu.detach(), sigma.detach(),
                                 inp['old_mu'], inp['old_sigma'],
                                 inp['rnn_masks'] is None)
        if inp['rnn_masks'] is not None:
            kl = (kl * inp['rnn_masks']).sum() / inp['rnn_masks'].numel()

    return loss, a_loss, c_loss, entropy, b_loss, kl


def run_both(inp, **cfg):
    from rl_games.triton_kernels.ppo_loss_kernel import fused_ppo_loss

    # reference
    ref_out = reference_loss(inp, **cfg)
    ref_loss = ref_out[0]
    grads_ref = torch.autograd.grad(ref_loss, [inp['mu'], inp['sigma'], inp['values']],
                                    allow_unused=True)

    # fused
    f = fused_ppo_loss(
        inp['mu'], inp['sigma'], inp['values'], inp['actions'],
        inp['old_neglogp'], inp['advantage'], inp['old_mu'], inp['old_sigma'],
        inp['old_values'], inp['returns'], inp['rnn_masks'],
        cfg['e_clip'], cfg['critic_coef'], cfg['entropy_coef'], cfg['bounds_coef'],
        is_ppo=cfg['is_ppo'], use_smooth_clamp=cfg['use_smooth_clamp'],
        clip_value=cfg['clip_value'], bound_loss_type=cfg['bound_loss_type'],
        has_value_loss=cfg['has_value_loss'])
    fused_scalars = f[:6]
    grads_fused = torch.autograd.grad(f[0], [inp['mu'], inp['sigma'], inp['values']],
                                      allow_unused=True)
    return ref_out, fused_scalars, grads_ref, grads_fused


def assert_close(ref, fused, name, atol=1e-5, rtol=1e-4):
    if ref is None and fused is None:
        return
    if ref is None:
        ref = torch.zeros_like(fused)
    if fused is None:
        fused = torch.zeros_like(ref)
    torch.testing.assert_close(fused, ref.reshape(fused.shape), atol=atol, rtol=rtol,
                               equal_nan=True, msg=lambda m: f'{name}: {m}')


CONFIG_MATRIX = list(itertools.product(
    [False, True],                       # with_mask
    [False, True],                       # clip_value
    ['none', 'regularisation', 'bound'], # bound_loss_type
    [False, True],                       # use_smooth_clamp
))


@pytest.mark.parametrize('with_mask,clip_value,bound_type,smooth', CONFIG_MATRIX)
def test_parity_feature_matrix(with_mask, clip_value, bound_type, smooth):
    inp = make_inputs(n=4096, num_actions=8, with_mask=with_mask,
                      seed=hash((with_mask, clip_value, bound_type, smooth)) % 2**31)
    cfg = dict(e_clip=0.2, critic_coef=2.0, entropy_coef=0.01,
               bounds_coef=0.005 if bound_type != 'none' else 0.0,
               is_ppo=True, use_smooth_clamp=smooth, clip_value=clip_value,
               bound_loss_type=bound_type, has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)

    names = ['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl']
    for name, r, fv in zip(names, ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)


def test_not_ppo():
    inp = make_inputs(n=2048, num_actions=4, seed=7)
    cfg = dict(e_clip=0.2, critic_coef=1.0, entropy_coef=0.0, bounds_coef=0.0,
               is_ppo=False, use_smooth_clamp=False, clip_value=True,
               bound_loss_type='none', has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)


def test_no_value_loss():
    inp = make_inputs(n=2048, num_actions=4, seed=11)
    cfg = dict(e_clip=0.2, critic_coef=4.0, entropy_coef=0.01, bounds_coef=0.0,
               is_ppo=True, use_smooth_clamp=False, clip_value=True,
               bound_loss_type='none', has_value_loss=False)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    assert torch.all(gfused[2] == 0), 'value grads must be zero when has_value_loss=False'


def test_multi_value_heads():
    inp = make_inputs(n=2048, num_actions=6, value_size=3, seed=13)
    cfg = dict(e_clip=0.2, critic_coef=2.0, entropy_coef=0.005, bounds_coef=0.0001,
               is_ppo=True, use_smooth_clamp=False, clip_value=True,
               bound_loss_type='bound', has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)


def test_multi_value_heads_masked():
    inp = make_inputs(n=2048, num_actions=6, value_size=3, with_mask=True, seed=17)
    cfg = dict(e_clip=0.2, critic_coef=2.0, entropy_coef=0.005, bounds_coef=0.0,
               is_ppo=True, use_smooth_clamp=False, clip_value=True,
               bound_loss_type='none', has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)


def test_odd_sizes():
    # non power-of-two everything, exercises padding/masking
    inp = make_inputs(n=1000 + 37, num_actions=5, value_size=1, seed=23)
    cfg = dict(e_clip=0.1, critic_coef=1.0, entropy_coef=0.02, bounds_coef=0.01,
               is_ppo=True, use_smooth_clamp=False, clip_value=False,
               bound_loss_type='regularisation', has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)


def make_discrete_inputs(n=4096, value_size=1, with_mask=False, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, device=DEVICE)

    old_neglogp = (rand(n) * 0.5 + 2.0).abs()
    neglogp = (old_neglogp + 0.1 * rand(n)).clone().requires_grad_(True)
    entropy = (rand(n) * 0.3 + 1.0).clone().requires_grad_(True)
    values = rand(n, value_size).requires_grad_(True)
    advantage = rand(n)
    old_values = values.detach() + 0.2 * rand(n, value_size)
    returns = old_values + rand(n, value_size)
    rnn_masks = (torch.rand(n, generator=g, device=DEVICE) > 0.3).float() if with_mask else None
    return dict(neglogp=neglogp, entropy=entropy, values=values,
                old_neglogp=old_neglogp, advantage=advantage,
                old_values=old_values, returns=returns, rnn_masks=rnn_masks)


def reference_loss_discrete(inp, e_clip, critic_coef, entropy_coef,
                            is_ppo, use_smooth_clamp, clip_value, has_value_loss):
    """Mirror of DiscreteA2CAgent.calc_gradients loss section."""
    from rl_games.common import common_losses
    from rl_games.algos_torch import torch_ext

    loss_func = common_losses.smoothed_actor_loss if use_smooth_clamp else common_losses.actor_loss
    a_loss = loss_func(inp['old_neglogp'], inp['neglogp'], inp['advantage'], is_ppo, e_clip)
    if has_value_loss:
        c_loss = common_losses.default_critic_loss(
            inp['old_values'], inp['values'], e_clip, inp['returns'], clip_value)
    else:
        c_loss = torch.zeros(1, device=DEVICE)
    losses, _ = torch_ext.apply_masks(
        [a_loss.unsqueeze(1), c_loss, inp['entropy'].unsqueeze(1)], inp['rnn_masks'])
    a_loss, c_loss, entropy = losses
    loss = a_loss + 0.5 * c_loss * critic_coef - entropy * entropy_coef
    with torch.no_grad():
        kl = 0.5 * ((inp['old_neglogp'] - inp['neglogp']) ** 2)
        if inp['rnn_masks'] is not None:
            kl = (kl * inp['rnn_masks']).sum() / inp['rnn_masks'].numel()
        else:
            kl = kl.mean()
    return loss, a_loss, c_loss, entropy, kl


@pytest.mark.parametrize('with_mask', [False, True])
@pytest.mark.parametrize('clip_value', [False, True])
@pytest.mark.parametrize('smooth', [False, True])
def test_discrete_parity(with_mask, clip_value, smooth):
    from rl_games.triton_kernels.ppo_loss_kernel import fused_ppo_loss_discrete

    inp = make_discrete_inputs(n=4096, with_mask=with_mask,
                               seed=hash((with_mask, clip_value, smooth)) % 2**31)
    cfg = dict(e_clip=0.2, critic_coef=2.0, entropy_coef=0.01,
               is_ppo=True, use_smooth_clamp=smooth, clip_value=clip_value,
               has_value_loss=True)

    ref = reference_loss_discrete(inp, **cfg)
    grads_ref = torch.autograd.grad(ref[0], [inp['neglogp'], inp['entropy'], inp['values']])

    f = fused_ppo_loss_discrete(
        inp['neglogp'], inp['entropy'], inp['values'],
        inp['old_neglogp'], inp['advantage'],
        inp['old_values'], inp['returns'], inp['rnn_masks'],
        cfg['e_clip'], cfg['critic_coef'], cfg['entropy_coef'],
        is_ppo=True, use_smooth_clamp=smooth, clip_value=clip_value,
        has_value_loss=True)
    grads_fused = torch.autograd.grad(f[0], [inp['neglogp'], inp['entropy'], inp['values']])

    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'kl'], ref, f[:5]):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_neglogp', 'grad_entropy', 'grad_values'],
                           grads_ref, grads_fused):
        assert_close(r, fv, name)


def test_single_action_dim():
    inp = make_inputs(n=512, num_actions=1, seed=29)
    cfg = dict(e_clip=0.2, critic_coef=1.0, entropy_coef=0.0, bounds_coef=0.0,
               is_ppo=True, use_smooth_clamp=False, clip_value=True,
               bound_loss_type='none', has_value_loss=True)
    ref, fused, gref, gfused = run_both(inp, **cfg)
    for name, r, fv in zip(['loss', 'a_loss', 'c_loss', 'entropy', 'b_loss', 'kl'], ref, fused):
        assert_close(r, fv, name)
    for name, r, fv in zip(['grad_mu', 'grad_sigma', 'grad_values'], gref, gfused):
        assert_close(r, fv, name)
