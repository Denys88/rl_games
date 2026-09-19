import torch

from rl_games.common.diagnostics import PpoDiagnostics


def _batch(actions, mu, sigma, adv, **extra):
    n = len(adv)
    b = {'values': torch.zeros(n, 1), 'returns': torch.arange(n, dtype=torch.float32).unsqueeze(1),
         'new_neglogp': torch.zeros(n), 'old_neglogp': torch.zeros(n), 'masks': None,
         'actions': torch.tensor(actions), 'mu': torch.tensor(mu), 'sigma': torch.tensor(sigma),
         'advantages': torch.tensor(adv)}
    b.update(extra)
    return b


def test_signed_sigma_score_direction():
    diag = PpoDiagnostics()
    # near-mean samples (z ~ 0): negative advantage pushes sigma UP, positive pushes it DOWN
    near = _batch([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.2, 0.2], [0.2, 0.2]], [-1.0, 1.0])
    diag.mini_batch(None, near, 0.2, 0)
    diag.mini_epoch(None, 0)
    assert float(diag.diag_dict['diagnostics/policy/sigma_score_neg/0']) > 0
    assert float(diag.diag_dict['diagnostics/policy/sigma_score_pos/0']) < 0
    assert float(diag.diag_dict['diagnostics/policy/tail_frac/0']) == 0.0
    # tail samples (|z| = 4): negative advantage pushes sigma DOWN; tail fraction counts them
    tail = _batch([[0.8, 0.0]], [[0.0, 0.0]], [[0.2, 0.2]], [-1.0])
    diag.mini_batch(None, tail, 0.2, 0)
    diag.mini_epoch(None, 1)
    assert float(diag.diag_dict['diagnostics/policy/sigma_score_neg/1']) < 0
    assert float(diag.diag_dict['diagnostics/policy/tail_frac/1']) == 1.0


def test_matched_sample_kl_logged_when_present():
    diag = PpoDiagnostics()
    b = _batch([[0.0]], [[0.0]], [[0.2]], [0.5], kl_step=torch.tensor([0.01]), kl_post_ref=torch.tensor([0.03]))
    diag.mini_batch(None, b, 0.2, 0)
    diag.mini_epoch(None, 0)
    assert abs(float(diag.diag_dict['diagnostics/policy/kl_step/0']) - 0.01) < 1e-6
    assert abs(float(diag.diag_dict['diagnostics/policy/kl_post_ref_max/0']) - 0.03) < 1e-6
