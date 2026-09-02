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


# ---- bypassed-training-forward guard ----

class _GuardAgent:
    """Just enough of A2CBase for trancate_gradients_and_step."""

    def __init__(self, model, ddp_model):
        self.multi_gpu = True
        self.multi_gpu_grad_sync = 'ddp'
        self._ddp_model = ddp_model
        self.model = model
        self.truncate_grads = False
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)


def _bypass_guard_worker(rank, world_size, port, results):
    import torch.distributed as dist
    from rl_games.common.a2c_common import A2CBase
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    net = _TwoHeadNet()
    ddp_net = wrap_model_ddp(net, 'cpu')
    agent = _GuardAgent(net, ddp_net)
    x = _rank_data(rank)

    def raw_step_raises():
        # forward through the RAW model: DDP hooks never fire, step must raise
        pi, v = net(x)
        (pi.sum() + v.sum()).backward()
        try:
            A2CBase.trancate_gradients_and_step(agent)
            return False
        except RuntimeError as e:
            return 'bypassed the DDP wrapper' in str(e)
        finally:
            net.zero_grad(set_to_none=True)

    raised = raw_step_raises()

    # forward through the wrapper: step passes and the flag clears after it
    pi, v = ddp_net(x)
    (pi.sum() + v.sum()).backward()
    A2CBase.trancate_gradients_and_step(agent)
    cleared = ddp_net.forward_seen is False

    # a no-grad forward through the wrapper must NOT arm the guard: it
    # installs no reduction hooks, so a subsequent raw-model training
    # forward/backward would step with unsynchronized gradients
    with torch.no_grad():
        ddp_net(x)
    armed_by_nograd = ddp_net.forward_seen
    nograd_guarded = not armed_by_nograd and raw_step_raises()

    # same for a no_sync() forward: DDP skips the reduction hooks there too
    with ddp_net.no_sync():
        ddp_net(x)
    armed_by_nosync = ddp_net.forward_seen
    nosync_guarded = not armed_by_nosync and raw_step_raises()

    results[rank] = (raised, cleared, nograd_guarded, nosync_guarded)
    dist.destroy_process_group()


def test_bypassed_training_forward_raises_and_flag_clears():
    import torch.multiprocessing as mp
    port = 31017 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_bypass_guard_worker, args=(1, port, results), nprocs=1, join=True)
        raised, cleared, nograd_guarded, nosync_guarded = results[0]
    assert raised, 'bypassed forward must raise an actionable RuntimeError'
    assert cleared, 'forward_seen must clear after the optimizer step'
    assert nograd_guarded, 'a no-grad wrapper forward must not arm the guard'
    assert nosync_guarded, 'a no_sync() wrapper forward must not arm the guard'


def _cv_dead_head_worker(rank, world_size, port, results):
    import torch.distributed as dist
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    from rl_games.envs.test_network import TestNet

    def two_iters(module, sample):
        ddp = wrap_model_ddp(module, 'cpu')
        opt = torch.optim.SGD(module.parameters(), lr=0.01)
        for _ in range(2):
            out = ddp(sample())
            out = out[0] if isinstance(out, tuple) else out
            out.sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    # the shipped central-value TestNet must train under DEFAULT DDP
    # (find_unused_parameters=False): no dead actor head anymore
    net = TestNet({'central_value': True},
                  actions_num=3, input_shape={'pos': (2,), 'info': (2,)})
    sample = lambda: {'obs': {'pos': torch.randn(5, 2), 'info': torch.randn(5, 2)}}
    two_iters(net, sample)

    # a genuinely dead head trains only with the plumbed knob
    class DeadHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Linear(4, 1)
            self.dead = torch.nn.Linear(4, 3)

        def forward(self, x):
            self.dead(x)  # computed, discarded: no grads for self.dead
            return self.value(x)

    dead = DeadHead()
    ddp = wrap_model_ddp(dead, 'cpu', find_unused_parameters=True)
    opt = torch.optim.SGD(dead.parameters(), lr=0.01)
    for _ in range(2):
        ddp(torch.randn(5, 4)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    results[rank] = True
    dist.destroy_process_group()


def test_central_value_network_trains_under_default_ddp():
    import torch.multiprocessing as mp
    port = 32017 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_cv_dead_head_worker, args=(1, port, results), nprocs=1, join=True)
        assert results[0] is True


# ---- ddp_find_unused_parameters plumbing ----

class _CvNet(_TwoHeadNet):
    def is_rnn(self):
        return False


class _StubNetwork:
    """CentralValueTrain network stand-in: build() returns a plain module."""

    def build(self, cfg):
        return _CvNet()


def _make_cv(top_level, cv_key=None, multi_gpu=False):
    """CentralValueTrain given the agent's top-level ddp_find_unused_parameters
    and, when cv_key is not None, the central_value_config key."""
    from rl_games.algos_torch.central_value import CentralValueTrain

    config = {
        'mini_epochs': 1, 'normalize_input': False, 'learning_rate': 1e-3,
        'clip_value': False, 'mixed_precision': False, 'lr_schedule': None,
        'minibatch_size': 4,
    }
    if cv_key is not None:
        config['ddp_find_unused_parameters'] = cv_key
    return CentralValueTrain(
        state_shape=(4,), value_size=1, ppo_device='cpu', num_agents=1,
        horizon_length=2, num_actors=2, num_actions=2, seq_length=2,
        normalize_value=False, network=_StubNetwork(), config=config,
        writter=None, max_epochs=1, multi_gpu=multi_gpu, zero_rnn_on_done=True,
        ddp_find_unused_parameters=top_level)


def test_ddp_find_unused_parameters_resolution():
    """central_value_config key > top-level key > False; A2CBase reads the
    top-level key from its config."""
    assert _make_cv(top_level=False).ddp_find_unused_parameters is False
    assert _make_cv(top_level=True).ddp_find_unused_parameters is True
    assert _make_cv(top_level=True, cv_key=False).ddp_find_unused_parameters is False
    assert _make_cv(top_level=False, cv_key=True).ddp_find_unused_parameters is True

    from tests.test_ppo_masking import make_ppo_agent
    agent, _ = make_ppo_agent(ddp_find_unused_parameters=True)
    assert agent.ddp_find_unused_parameters is True
    agent, _ = make_ppo_agent()
    assert agent.ddp_find_unused_parameters is False


def _setup_multi_gpu_worker(rank, world_size, port, results):
    import types
    import torch.distributed as dist
    from rl_games.common.a2c_common import A2CBase
    dist.init_process_group(
        'gloo', rank=rank, world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}')
    # CPU-only: setup_multi_gpu pins the rank's GPU before anything else
    torch.cuda.set_device = lambda *_: None
    out = {}
    for top_level in (False, True):
        agent = types.SimpleNamespace(
            multi_gpu=True, local_rank=0, ppo_device='cpu',
            multi_gpu_grad_sync='ddp', model=_TwoHeadNet(),
            has_central_value=True, ddp_find_unused_parameters=top_level,
            # the CV key carries the opposite value and must win there
            central_value_net=_make_cv(top_level, cv_key=not top_level,
                                       multi_gpu=True))
        A2CBase.setup_multi_gpu(agent)
        out[top_level] = (agent._ddp_model.find_unused_parameters,
                          agent.central_value_net._ddp_model.find_unused_parameters)
    results[rank] = out
    dist.destroy_process_group()


def test_setup_multi_gpu_plumbs_find_unused_parameters_to_both_wraps():
    import torch.multiprocessing as mp
    port = 33017 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.dict()
        mp.spawn(_setup_multi_gpu_worker, args=(1, port, results), nprocs=1, join=True)
        out = results[0]
    # (agent-side DDP, central-value DDP)
    assert out[True] == (True, False)
    assert out[False] == (False, True)
