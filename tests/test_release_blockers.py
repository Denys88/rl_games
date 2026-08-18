"""Regression tests for the 2.0 release-blocker batch (silent-wrong-behavior
class): seed 0, legacy seq_len, SAC multi_gpu guard, adaptive-LR resume."""

import copy

import numpy as np
import pytest
import torch

from rl_games.torch_runner import Runner
from tests.test_critical_fixes import (_load_params, CARTPOLE_YAML,
                                       make_cartpole_agent)


def _load_runner_with_seed(seed):
    # seed is a TOP-LEVEL params key (not config), so build params directly
    params = _load_params(CARTPOLE_YAML)
    params['seed'] = seed
    r = Runner()
    r.load({'params': copy.deepcopy(params)})
    return r


def test_seed_zero_actually_seeds():
    # seed: 0 is a conventional choice; `if self.seed:` treated it as falsy and
    # silently skipped ALL seeding while logging that seed 0 was in effect
    draws = []
    for _ in range(2):
        _load_runner_with_seed(0)
        draws.append((torch.rand(3).tolist(), np.random.rand(3).tolist()))
    assert draws[0] == draws[1], 'seed 0 did not seed torch/numpy'
    # and seed 0 is genuinely seed 0, not some other stream
    torch.manual_seed(0)
    np.random.seed(0)
    expected = (torch.rand(3).tolist(), np.random.rand(3).tolist())
    assert draws[0] == expected


def test_legacy_seq_len_falls_back_instead_of_silently_dropping():
    # the deprecation warning implied the old key still worked; it was
    # discarded and the RNN silently trained with seq_length=4 (a shipped
    # config, ppo_pacman_torch_rnn.yaml, hit this)
    # (batch is 16 in the tiny factory: keep seq values divisible)
    agent = make_cartpole_agent(seq_len=8)
    assert agent.seq_length == 8
    agent = make_cartpole_agent(seq_len=4, seq_length=8)   # new key wins
    assert agent.seq_length == 8


def test_sac_rejects_multi_gpu():
    # SAC reads RANK only to gate the writer -- no DDP wrapper, no gradient
    # collective. torchrun silently trained N independent agents racing on
    # the same checkpoint files. Refuse loudly until real support exists.
    from tests.test_sac_correctness import make_fake_env_sac_agent
    with pytest.raises(NotImplementedError, match='multi_gpu is not supported for SAC'):
        make_fake_env_sac_agent(multi_gpu=True)


def test_adaptive_lr_state_survives_resume():
    # the optimizer LR was checkpointed but last_lr was not: after restore the
    # next scheduler.update restarted the adaptive walk from the config LR
    # (up to ~30x too high late in training -- KL spike on resume)
    src = make_cartpole_agent()
    src.last_lr = 3.7e-5
    src.entropy_coef = 0.0123
    state = src.get_full_state_weights()
    dst = make_cartpole_agent()
    assert dst.last_lr != 3.7e-5
    dst.set_full_state_weights(state)
    assert dst.last_lr == 3.7e-5
    assert dst.entropy_coef == 0.0123
    # old checkpoints without the keys keep config-derived values
    for k in ('last_lr', 'entropy_coef'):
        state.pop(k)
    legacy = make_cartpole_agent()
    before = (legacy.last_lr, legacy.entropy_coef)
    legacy.set_full_state_weights(state)
    assert (legacy.last_lr, legacy.entropy_coef) == before
