"""Opt-in policy telemetry must expose outliers without counting masked rows."""

import pytest
import torch

from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics


def _batch(sigma, mu=None, advantages=None, logratio=None, masks=None):
    sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)
    n = sigma.shape[0]
    old_neglogp = torch.arange(n, dtype=torch.float32, requires_grad=True)
    logratio = torch.zeros(n) if logratio is None else torch.tensor(logratio)
    return {
        'values': torch.arange(n, dtype=torch.float32).reshape(n, 1),
        'returns': torch.arange(n, dtype=torch.float32).reshape(n, 1) + 1,
        'new_neglogp': old_neglogp - logratio,
        'old_neglogp': old_neglogp,
        'masks': None if masks is None else torch.tensor(masks),
        'sigma': sigma,
        'mu': sigma * 2 if mu is None else torch.tensor(mu, dtype=torch.float32),
        'advantages': torch.ones(n) if advantages is None else torch.tensor(advantages),
    }


def _metrics(diag, mini_epoch=0):
    diag.mini_epoch(None, mini_epoch)
    prefix = 'diagnostics/policy/'
    suffix = f'/{mini_epoch}'
    return {key[len(prefix):-len(suffix)]: value for key, value in diag.diag_dict.items()
            if key.startswith(prefix) and key.endswith(suffix)}


def test_epoch_extrema_and_sigma_mean_include_all_action_elements():
    diag = PpoDiagnostics()
    batches = [
        _batch([[.2, .4], [.6, 12.]], advantages=[-9., 2.], logratio=[-.8, .3]),
        _batch([[.1, .3]], mu=[[-30., 1.]], advantages=[3.], logratio=[.1]),
    ]
    for batch in batches:
        diag.mini_batch(None, batch, .2, 0)
    # Aggregates are detached scalar snapshots, not references to the graph.
    for stats in diag.policy_stats:
        for value in stats.values():
            if torch.is_tensor(value):
                assert value.ndim == 0 and value.grad_fn is None
                assert not value.requires_grad
    metrics = _metrics(diag)
    expected = {'sigma_min': .1, 'sigma_max': 12., 'sigma_mean': 13.6 / 6,
                'mu_abs_max': 30., 'advantage_abs_max': 9., 'logratio_abs_max': .8}
    assert set(metrics) == set(expected)
    for name, value in expected.items():
        assert metrics[name].item() == pytest.approx(value)
    assert not diag.policy_stats


@pytest.mark.parametrize('masks', [[1., 0., 1.], [[1.], [0.], [1.]]])
def test_invalid_rows_do_not_hide_valid_extrema(masks):
    diag = PpoDiagnostics()
    batch = _batch([[.2, .4], [float('nan'), float('inf')], [.6, .8]],
                   mu=[[-2., 1.], [float('nan'), float('inf')], [3., 4.]],
                   advantages=[-5., float('nan'), 2.],
                   logratio=[-.7, float('inf'), .2], masks=masks)
    diag.mini_batch(None, batch, .2, 0)
    metrics = _metrics(diag)
    for name, expected in {'sigma_min': .2, 'sigma_max': .8, 'sigma_mean': .5,
                           'mu_abs_max': 4., 'advantage_abs_max': 5.,
                           'logratio_abs_max': .7}.items():
        assert metrics[name].item() == pytest.approx(expected)


def test_no_valid_rows_omit_metrics_and_clear_previous_epoch_values():
    diag = PpoDiagnostics()
    diag.mini_batch(None, _batch([[.2], [.4]]), .2, 0)
    assert _metrics(diag)
    diag.mini_batch(None, _batch([[float('nan')], [float('inf')]], masks=[0., 0.]), .2, 0)
    assert _metrics(diag) == {}


def test_discrete_batches_need_no_continuous_policy_fields():
    diag = PpoDiagnostics()
    batch = _batch([[.2], [.4]], logratio=[.1, -.9])
    for key in ('sigma', 'mu', 'advantages'):
        batch.pop(key)
    diag.mini_batch(None, batch, .2, 0)
    metrics = _metrics(diag)
    assert set(metrics) == {'logratio_abs_max'}
    assert metrics['logratio_abs_max'].item() == pytest.approx(.9)


def test_disabled_diagnostics_never_read_or_compute_on_batch():
    class UnreadableBatch:
        def __getitem__(self, key):
            raise AssertionError('disabled diagnostics accessed a tensor')

    diag = DefaultDiagnostics()
    diag.mini_batch(None, UnreadableBatch(), .2, 0)
    diag.mini_epoch(None, 0)
    assert not vars(diag)


def test_continuous_agent_supplies_learner_policy_and_rollout_logprobs(tmp_path):
    from tests.test_ppo_masking import make_ppo_agent, _rollout_batch

    agent, _ = make_ppo_agent(use_diagnostics=True, train_dir=str(tmp_path))
    try:
        batch = _rollout_batch(agent)
        agent.prepare_dataset(batch)
        minibatch = agent.dataset[0]
        # Simulate a likelihood change relative to the immutable rollout data.
        minibatch['old_logp_actions'] = minibatch['old_logp_actions'] + 2.
        agent.train_actor_critic(minibatch)
        metrics = _metrics(agent.diagnostics)
        assert metrics['logratio_abs_max'].item() == pytest.approx(2., abs=1e-5)
        assert metrics['sigma_min'].item() == pytest.approx(1.)
        assert metrics['sigma_max'].item() == pytest.approx(1.)
        valid = minibatch['rnn_masks'].bool()
        assert metrics['advantage_abs_max'].item() == pytest.approx(
            minibatch['advantages'][valid].abs().max().item())
        # post-step forward: the optimizer step moved the policy, so the
        # matched-sample KLs are strictly positive, finite, and max >= mean
        for name in ('kl_step', 'kl_post_ref', 'kl_post_ref_max'):
            assert torch.isfinite(metrics[name]), name
            assert metrics[name].item() > 0., name
        assert metrics['kl_post_ref_max'].item() >= metrics['kl_post_ref'].item()
    finally:
        agent.writer.close()
