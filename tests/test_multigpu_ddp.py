"""Multi-GPU gradient sync: DDP-vs-flat-allreduce gradient parity on a real
2-process gloo group, the unused-value-head (0-coef term) config, and the
state_dict invariance of the lazily created DDP wrapper."""

import os

import pytest
import torch
import torch.nn as nn

from rl_games.algos_torch.torch_ext import wrap_model_ddp, flat_allreduce_grads


class _TwoHeadNet(nn.Module):
    """Tiny actor-critic stand-in: shared trunk, policy head, value head."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.trunk = nn.Linear(4, 8)
        self.policy = nn.Linear(8, 2)
        self.value = nn.Linear(8, 1)

    def forward(self, x):
        h = torch.relu(self.trunk(x))
        return self.policy(h), self.value(h)


def _rank_data(rank):
    torch.manual_seed(100 + rank)
    return torch.randn(16, 4)


def _parity_worker(rank, world_size, port, results):
    import torch.distributed as dist
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    x = _rank_data(rank)

    # path A: DDP averages during backward
    ddp_net = wrap_model_ddp(_TwoHeadNet(), 'cpu')
    pi, v = ddp_net(x)
    (pi.sum() + v.sum()).backward()
    ddp_grads = [p.grad.clone() for p in ddp_net.module.parameters()]

    # path B: local backward + flat all-reduce
    flat_net = _TwoHeadNet()
    pi, v = flat_net(x)
    (pi.sum() + v.sum()).backward()
    flat_allreduce_grads(flat_net, world_size)
    flat_grads = [p.grad.clone() for p in flat_net.parameters()]

    results[rank] = (ddp_grads, flat_grads)
    dist.destroy_process_group()


def test_ddp_and_flat_allreduce_grads_match_across_two_ranks():
    import torch.multiprocessing as mp
    port = 29617 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_parity_worker, args=(2, port, results), nprocs=2, join=True)
        for rank in (0, 1):
            ddp_grads, flat_grads = results[rank]
            for g_ddp, g_flat in zip(ddp_grads, flat_grads):
                assert torch.allclose(g_ddp, g_flat, atol=1e-6)
        # both ranks hold the same averaged gradients
        for g0, g1 in zip(results[0][0], results[1][0]):
            assert torch.allclose(g0, g1, atol=0, rtol=0)


def _unused_head_worker(rank, world_size, port, results):
    import torch.distributed as dist
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    net = wrap_model_ddp(_TwoHeadNet(), 'cpu')
    ok_steps = 0
    for step in range(3):
        x = _rank_data(rank) + step
        pi, v = net(x)
        # has_value_loss=False shape: value head excluded from the objective,
        # kept in the graph via the 0-coef term (a2c calc_losses pattern)
        loss = pi.sum() + 0.0 * v.sum()
        loss.backward()
        value_grads_zero = all(
            torch.count_nonzero(p.grad) == 0 for p in net.module.value.parameters())
        net.module.zero_grad(set_to_none=True)
        ok_steps += int(value_grads_zero)
    results[rank] = ok_steps
    dist.destroy_process_group()


def test_zero_coef_value_head_survives_ddp_bucket_accounting():
    # without the 0-coef term DDP raises "Expected to have finished reduction"
    # on the second forward; with it, several steps run and value grads are 0
    import torch.multiprocessing as mp
    port = 29717 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_unused_head_worker, args=(2, port, results), nprocs=2, join=True)
        assert results[0] == 3 and results[1] == 3


def test_ddp_wrapper_stays_out_of_module_state_dict():
    # CentralValueTrain assigns its wrapper via self.__dict__ so that
    # nn.Module.__setattr__ never registers it as a child; mirror that
    # pattern and assert state_dict keys are unchanged
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TwoHeadNet()
            self.__dict__['_ddp_model'] = None

    import torch.distributed as dist
    port = 29817 + os.getpid() % 1000
    dist.init_process_group(
        'gloo', rank=0, world_size=1, init_method=f'tcp://127.0.0.1:{port}')
    try:
        holder = Holder()
        keys_before = set(holder.state_dict().keys())
        holder.__dict__['_ddp_model'] = wrap_model_ddp(holder.model, 'cpu')
        keys_after = set(holder.state_dict().keys())
        assert keys_before == keys_after
        assert not any(k.startswith('_ddp_model') for k in keys_after)
        # a fresh instance strict-restores from the wrapped one's checkpoint
        fresh = Holder()
        fresh.load_state_dict(holder.state_dict(), strict=True)
    finally:
        dist.destroy_process_group()
