"""Data-parallel SAC (multi_gpu) on a real 2-process gloo group on CPU: every rank steps its own envs into its
own replay and samples its own batches, yet after training all ranks hold byte-identical networks,
temperatures, batch-norm statistics and reward scales, and frames count both ranks."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_sac_flash import flash_network, make_agent

from rl_games.envs.test.point_mass_tracking import PointMassTracking


DEFAULT_NETWORK = {'name': 'soft_actor_critic', 'separate': True, 'space': {'continuous': {}},
                   'mlp': {'units': [32, 32], 'activation': 'relu', 'initializer': {'name': 'default'}},
                   'log_std_bounds': [-5, 2]}


def _digest(agent):
    state = agent.model.state_dict()
    tensors = [v.double().flatten() for v in state.values() if torch.is_tensor(v) and v.is_floating_point()]
    tensors.append(agent.log_alpha.detach().double().flatten())
    if agent.reward_scaler is not None:
        tensors += [agent.reward_scaler.var.double(), agent.reward_scaler.abs_max.double()]
    return torch.cat(tensors)


def _worker(rank, world_size, port, results, network, config):
    os.environ.update({'RANK': str(rank), 'LOCAL_RANK': str(rank), 'WORLD_SIZE': str(world_size),
                       'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port)})
    torch.set_num_threads(2)
    import torch.distributed as dist
    config = dict(config)
    env = PointMassTracking(16, num_groups=config.pop('num_groups', 2), seed=10 + rank)
    agent = make_agent(env, network, seed=rank, multi_gpu=True, multi_gpu_backend='gloo', **config)
    agent.train()
    rb = agent.replay_buffer
    obs = rb.obs if hasattr(rb, 'obs') else rb.obses
    results[rank] = {'digest': _digest(agent), 'updates': agent.update_counter, 'frame': agent.frame,
                     'lr_decay_steps': agent.lr_decay_steps,
                     'replay_obs': obs[:64].float().clone()}
    dist.destroy_process_group()


_RESULTS = {}


def _run(replay_memory_efficient, overlap):
    """Rank results of a 2-rank run (cached: the tests below compare runs)."""
    key = (replay_memory_efficient, overlap)
    if key not in _RESULTS:
        import torch.multiprocessing as mp
        port = 29700 + os.getpid() % 1000 + 2 * int(replay_memory_efficient) + int(overlap)
        config = {'max_frames': 960, 'max_epochs': -1, 'replay_memory_efficient': replay_memory_efficient,
                  'lr_schedule': 'cosine',
                  'mixed_precision': False, 'multi_gpu_overlap': overlap}
        with mp.Manager() as manager:
            results = manager.dict()
            mp.spawn(_worker, args=(2, port, results, flash_network(), config), nprocs=2, join=True)
            _RESULTS[key] = (results[0], results[1])
    return _RESULTS[key]


@pytest.mark.parametrize('replay_memory_efficient,overlap', [(False, True), (True, True), (True, False)])
def test_ranks_stay_identical(replay_memory_efficient, overlap):
    r0, r1 = _run(replay_memory_efficient, overlap)
    assert torch.equal(r0['digest'], r1['digest'])
    assert r0['updates'] == r1['updates'] > 0
    assert r0['frame'] == r1['frame'] == 960  # max_frames counts both ranks: 3 epochs x 16 envs x 10 steps x 2
    assert not torch.equal(r0['replay_obs'], r1['replay_obs'])  # own envs, own replay
    assert r0['lr_decay_steps'] == 3 * 10 * 2  # the cosine spans each rank's 3 epochs x 10 steps x 2 updates


def test_overlapped_and_flat_allreduce_agree():
    """The per-parameter all-reduce during backward gives the same mean gradients as one flat one after it."""
    assert torch.equal(_run(True, True)[0]['digest'], _run(True, False)[0]['digest'])


def test_default_network_with_obs_normalizer_stays_identical():
    """Plain SAC (MLP networks, no batch norm) with normalize_input: the obs statistics are pooled each epoch."""
    import torch.multiprocessing as mp
    port = 29720 + os.getpid() % 1000
    config = {'max_epochs': 3, 'normalize_input': True, 'normalize_reward': False, 'held_noise': False,
              'num_groups': 1, 'n_step': 1, 'mixed_precision': False}
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(2, port, results, DEFAULT_NETWORK, config), nprocs=2, join=True)
        r0, r1 = results[0], results[1]
    assert torch.equal(r0['digest'], r1['digest'])
    assert r0['updates'] == r1['updates'] > 0


def _nccl_worker(rank, world_size, port, results, config):
    os.environ.update({'RANK': str(rank), 'LOCAL_RANK': str(rank), 'WORLD_SIZE': str(world_size),
                       'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port)})
    torch.cuda.set_device(rank)
    env = PointMassTracking(64, num_groups=2, seed=10 + rank, device=f'cuda:{rank}')
    agent = make_agent(env, flash_network(actor_units=32, critic_units=64, blocks=2), seed=rank, device='cuda',
                       multi_gpu=True, batch_size=128, replay_memory_efficient=True, lr_schedule='cosine', **config)
    agent.train()
    results[rank] = {'digest': _digest(agent).cpu(), 'updates': agent.update_counter,
                     'device': str(agent.model.sac_network.critic.out_bias.device),
                     'captured': sorted(agent.graph_step.graphs) if agent.graph_step is not None else []}
    # no destroy_process_group: with captured NCCL collectives it can block; process exit frees the communicator


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='needs 2 GPUs')
@pytest.mark.parametrize('config', [
    {'mixed_precision': 'fp16', 'update_cuda_graph': True},
    {'mixed_precision': False, 'multi_gpu_overlap': False},
    {'mixed_precision': 'fp16', 'multi_gpu_grad_dtype': 'bfloat16'},
], ids=['fp16_graph_overlap', 'fp32_flat', 'fp16_bf16_traffic'])
def test_nccl_two_gpus_stay_identical(config):
    """Real NCCL on two GPUs (overlapped all-reduce, CUDA graph capture): each rank on its own GPU, same networks."""
    import torch.multiprocessing as mp
    port = 29740 + os.getpid() % 1000 + len(str(config))
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_nccl_worker, args=(2, port, results, dict(config, max_epochs=4)), nprocs=2, join=True)
        r0, r1 = results[0], results[1]
    assert (r0['device'], r1['device']) == ('cuda:0', 'cuda:1')
    assert r0['updates'] == r1['updates'] > 0
    assert torch.equal(r0['digest'], r1['digest'])
    if config.get('update_cuda_graph'):
        assert r0['captured'] == [False, True]
