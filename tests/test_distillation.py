"""Tests for rl_games/common/distillation.py (frozen-teacher distillation)."""

import numpy as np
import pytest
import torch


def test_piecewise_linear_schedule():
    from rl_games.common.distillation import piecewise_linear
    spec = [[0, 1.0], [10, 1.0], [20, 0.0]]
    assert piecewise_linear(spec, -5) == 1.0
    assert piecewise_linear(spec, 5) == 1.0
    assert abs(piecewise_linear(spec, 15) - 0.5) < 1e-9
    assert piecewise_linear(spec, 30) == 0.0
    assert piecewise_linear(0.3, 7) == 0.3            # scalar spec = constant
    assert piecewise_linear(None, 7) == 0.0


class _Teacher(torch.nn.Module):
    """Stand-in teacher: logits = W states (discrete) or mus = W states (continuous)."""

    def __init__(self, in_dim, out_dim, discrete=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.discrete = discrete

    def forward(self, d):
        h = self.lin(d['obs'])
        if self.discrete:
            return {'logits': h}
        return {'mus': h, 'sigmas': torch.full_like(h, 0.5)}


def _distill(teacher, discrete, **cfg):
    from rl_games.common.distillation import TeacherDistillation
    base = {'coef': 1.0, 'ppo_coef': 1.0, 'beta': 0.0, 'loss': 'kl'}
    base.update(cfg)
    return TeacherDistillation(base, teacher, is_discrete=discrete, device='cpu')


def test_discrete_kl_matches_manual_and_is_zero_for_identical():
    t = _Teacher(4, 5)
    d = _distill(t, True)
    states = torch.randn(16, 4)
    student_logits = torch.randn(16, 5)
    with torch.no_grad():
        p = torch.softmax(t.lin(states), -1)
    manual = (p * (torch.log(p) - torch.log_softmax(student_logits, -1))).sum(-1).mean()
    got = d.loss({'logits': student_logits}, states)
    assert torch.allclose(got, manual, atol=1e-6)
    with torch.no_grad():
        same = d.loss({'logits': t.lin(states)}, states)
    assert same.abs() < 1e-6
    masks = torch.zeros(16, 1)
    masks[:4] = 1
    masked = d.loss({'logits': student_logits}, states, rnn_masks=masks)
    manual_m = (p * (torch.log(p) - torch.log_softmax(student_logits, -1))).sum(-1)[:4].mean()
    assert torch.allclose(masked, manual_m, atol=1e-6)


def test_gaussian_kl_and_mse():
    t = _Teacher(4, 3, discrete=False)
    states = torch.randn(8, 4)
    mus = torch.randn(8, 3)
    sig = torch.full((8, 3), 0.8)
    with torch.no_grad():
        tm = t.lin(states)
        ts = torch.full_like(tm, 0.5)
    kl = (torch.log(sig / ts) + (ts ** 2 + (tm - mus) ** 2) / (2 * sig ** 2) - 0.5).sum(-1).mean()
    assert torch.allclose(_distill(t, False).loss({'mus': mus, 'sigmas': sig}, states), kl, atol=1e-6)
    mse = ((tm - mus) ** 2).sum(-1).mean()
    assert torch.allclose(_distill(t, False, loss='mse').loss({'mus': mus, 'sigmas': sig}, states), mse, atol=1e-6)


def test_mix_actions_substitutes_beta_rows_with_valid_neglogp():
    torch.manual_seed(1)
    t = _Teacher(4, 5)
    d = _distill(t, True, beta=[[0, 1.0], [10, 0.0]])
    N = 2000
    states = torch.randn(N, 4)
    logits = torch.randn(N, 5)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    res = {'logits': logits, 'actions': actions.clone(), 'neglogpacs': -dist.log_prob(actions)}
    out = d.mix_actions(res, states, epoch=5)                       # beta 0.5
    changed = (out['actions'] != actions).float().mean().item()
    assert 0.25 < changed < 0.55                                     # ~half the rows re-drawn from the teacher
    assert torch.allclose(out['neglogpacs'], -dist.log_prob(out['actions']), atol=1e-5)
    fresh = {'logits': logits, 'actions': actions.clone(), 'neglogpacs': -dist.log_prob(actions)}
    untouched = d.mix_actions(fresh, states, epoch=20)              # beta 0
    assert torch.equal(untouched['actions'], actions)
    # continuous: substituted rows carry the student's Gaussian neglogp
    tc = _Teacher(4, 3, discrete=False)
    dc = _distill(tc, False, beta=1.0)
    mus = torch.randn(N, 3)
    sig = torch.full((N, 3), 0.7)
    resc = {'mus': mus, 'sigmas': sig, 'actions': mus.clone(), 'neglogpacs': torch.zeros(N)}
    outc = dc.mix_actions(resc, states, epoch=0)
    ref = -torch.distributions.Normal(mus, sig).log_prob(outc['actions']).sum(-1)
    assert torch.allclose(outc['neglogpacs'], ref, atol=1e-4)
    assert not torch.allclose(outc['actions'], mus)


def test_warm_start_copies_matching_tensors_only():
    teacher = torch.nn.ModuleDict({'trunk': torch.nn.Linear(4, 8), 'value': torch.nn.Linear(8, 1),
                                   'logits': torch.nn.Linear(8, 5)})
    cv = torch.nn.ModuleDict({'trunk': torch.nn.Linear(4, 8), 'value': torch.nn.Linear(8, 1),
                              'extra': torch.nn.Linear(2, 2)})
    d = _distill(teacher, True)
    copied, skipped = d.warm_start_central_value(cv)
    assert copied == 4 and skipped == 2                           # extra.* skipped
    assert torch.equal(cv['trunk'].weight, teacher['trunk'].weight)
    assert torch.equal(cv['value'].bias, teacher['value'].bias)


def test_teacher_mu_clip_bounds_continuous_targets():
    t = _Teacher(4, 3, discrete=False, seed=3)
    with torch.no_grad():
        t.lin.weight.mul_(10.0)                       # teacher means far outside [-1, 1]
    states = torch.randn(8, 4)
    mus = torch.zeros(8, 3)
    sig = torch.full((8, 3), 0.5)
    with torch.no_grad():
        tm = t.lin(states)
    assert tm.abs().max() > 1.5
    clipped = _distill(t, False, loss='mse', teacher_mu_clip=1.0).loss({'mus': mus, 'sigmas': sig}, states)
    expected = (tm.clamp(-1.0, 1.0) ** 2).sum(-1).mean()
    assert torch.allclose(clipped, expected, atol=1e-6)
    unclipped = _distill(t, False, loss='mse').loss({'mus': mus, 'sigmas': sig}, states)
    assert unclipped > clipped
