"""Cross-rank RunningMeanStd synchronization: merge math, resume seeding,
and a real 2-process gloo all-reduce roundtrip."""

import os

import numpy as np
import pytest
import torch

from rl_games.common.a2c_common import merge_rank_stats, seed_stats_sync_snapshot
from rl_games.algos_torch.running_mean_std import RunningMeanStd


def _allreduce_two_identical_ranks(t):
    # SUM all-reduce where both ranks hold identical tensors
    # (int multiplier: count tensors are integer dtype, like real NCCL SUM)
    t.mul_(2)


def test_moment_merge_math_matches_concatenated_data():
    torch.manual_seed(3)
    a = torch.randn(500, 4) * 2 + 1
    b = torch.randn(300, 4) * 5 - 2
    n1, n2 = float(len(a)), float(len(b))
    m1, v1 = a.mean(0), a.var(0, unbiased=False)
    m2, v2 = b.mean(0), b.var(0, unbiased=False)
    n = n1 + n2
    mean = (m1 * n1 + m2 * n2) / n
    var = ((v1 + m1 ** 2) * n1 + (v2 + m2 ** 2) * n2) / n - mean ** 2
    full = torch.cat([a, b])
    assert torch.allclose(mean, full.mean(0), atol=1e-5)
    assert torch.allclose(var, full.var(0, unbiased=False), atol=1e-4)


def test_fresh_start_merge_sums_local_histories():
    torch.manual_seed(5)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(400, 3) * 2 + 1)
    c0, mean0, var0 = m.count.clone(), m.running_mean.clone(), m.running_var.clone()
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    # fresh start: each rank's history is genuinely local data -> counts add
    assert torch.allclose(m.count.float(), 2 * c0.float(), rtol=1e-6)
    assert torch.allclose(m.running_mean, mean0, atol=1e-5)
    assert torch.allclose(m.running_var, var0, atol=1e-4)


def test_delta_merge_does_not_double_weight_shared_history():
    # second sync with no new data must be a no-op (deltas are zero)
    torch.manual_seed(8)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(200, 3) + 4)
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    c1, mean1 = m.count.clone(), m.running_mean.clone()
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    assert torch.equal(m.count, c1)
    assert torch.allclose(m.running_mean, mean1, atol=0, rtol=0)


def test_seeded_snapshot_prevents_resume_count_inflation():
    torch.manual_seed(6)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(400, 3) + 2)
    # emulate checkpoint restore: every rank loads IDENTICAL stats (shared
    # history) -- the restore path seeds the snapshot so the first sync
    # must not re-sum that history across ranks
    seed_stats_sync_snapshot(m)
    c0, mean0 = m.count.clone(), m.running_mean.clone()
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    assert torch.allclose(m.count.float(), c0.float(), rtol=1e-6)  # NOT x world_size
    assert torch.allclose(m.running_mean, mean0, atol=1e-6)
    # data collected AFTER the resume is fresh per-rank: only the delta sums
    m(torch.randn(100, 3) - 1)
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    assert torch.allclose(m.count.float(), c0.float() + 200.0, rtol=1e-5)


def _gloo_worker(rank, world_size, port, results):
    import torch.distributed as dist
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    torch.manual_seed(100 + rank)
    m = RunningMeanStd((2,))
    m.train()
    # DISJOINT per-rank data: rank 0 ~ N(0,1), rank 1 ~ N(10,4)
    data = torch.randn(300, 2) * (1 + 3 * rank) + 10 * rank
    m(data)
    merge_rank_stats(m, lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM))
    results[rank] = (m.count.clone(), m.running_mean.clone(),
                     m.running_var.clone(), data)
    dist.destroy_process_group()


def test_two_process_gloo_merge_matches_pooled_data():
    import torch.multiprocessing as mp
    port = 29517 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_gloo_worker, args=(2, port, results), nprocs=2, join=True)
        (c0, mean0, var0, d0), (c1, mean1, var1, d1) = results[0], results[1]
    # both ranks converge to identical merged stats
    assert torch.equal(c0, c1)
    assert torch.allclose(mean0, mean1, atol=0, rtol=0)
    # and those stats equal the EXACT prior-weighted pooled statistics:
    # each rank's RunningMeanStd starts with count 1, mean 0, var 1 (the init
    # prior), so the merged totals include 2 prior samples
    pooled = torch.cat([d0, d1]).to(mean0.dtype)
    n_prior = c0.float() - len(pooled)          # prior pseudo-counts (2)
    exp_mean = pooled.sum(0) / c0.float()       # prior mean is 0
    exp_var = (pooled.pow(2).sum(0) + n_prior * 1.0) / c0.float() - exp_mean ** 2
    assert torch.allclose(mean0, exp_mean, atol=1e-6)
    assert torch.allclose(var0, exp_var, atol=1e-5)


def test_unknown_sync_mode_raises():
    from rl_games.common.a2c_common import resolve_stats_sync_mode
    assert resolve_stats_sync_mode('pooled') == 'pooled'
    assert resolve_stats_sync_mode('broadcast') == 'broadcast'
    with pytest.raises(ValueError, match='multi_gpu_sync_stats_mode'):
        resolve_stats_sync_mode('all_reduce')


def test_broadcast_is_stateless_and_idempotent():
    from rl_games.common.a2c_common import broadcast_rank_stats
    torch.manual_seed(11)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(250, 3) * 3 + 5)
    c0, mean0, var0 = m.count.clone(), m.running_mean.clone(), m.running_var.clone()
    identity = lambda t: t   # rank 0's own broadcast is a no-op
    broadcast_rank_stats(m, identity)
    broadcast_rank_stats(m, identity)
    assert torch.equal(m.count, c0)
    assert torch.equal(m.running_mean, mean0)
    assert torch.equal(m.running_var, var0)
    # stateless: no snapshot bookkeeping is created
    assert not hasattr(m, '_stats_sync_snapshot')


def _gloo_broadcast_worker(rank, world_size, port, results):
    import torch.distributed as dist
    from rl_games.common.a2c_common import broadcast_rank_stats
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    torch.manual_seed(200 + rank)
    m = RunningMeanStd((2,))
    m.train()
    # deliberately DIFFERENT per-rank streams
    m(torch.randn(300, 2) * (1 + 3 * rank) + 10 * rank)
    pre = (m.count.clone(), m.running_mean.clone(), m.running_var.clone())
    broadcast_rank_stats(m, lambda t: dist.broadcast(t, src=0))
    results[rank] = (pre, (m.count.clone(), m.running_mean.clone(), m.running_var.clone()))
    dist.destroy_process_group()


def test_two_process_gloo_broadcast_makes_ranks_byte_identical():
    import torch.multiprocessing as mp
    port = 30517 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_gloo_broadcast_worker, args=(2, port, results), nprocs=2, join=True)
        (pre0, post0), (pre1, post1) = results[0], results[1]
    # ranks disagreed before...
    assert not torch.allclose(pre0[1], pre1[1])
    # ...and are BYTE-identical to rank 0's pre-broadcast state after
    for a, b in zip(post0, pre0):
        assert torch.equal(a, b)      # rank 0 unchanged
    for a, b in zip(post1, pre0):
        assert torch.equal(a, b)      # rank 1 adopted rank 0 exactly
