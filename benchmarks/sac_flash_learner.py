"""FlashSAC learner throughput at the size of a large bimanual tracking run, without an env.

Builds SAC with the flash_sac network from env_info alone (obs 3088 with a 780-column future window,
36 actions in two groups: actor 0.65M per group, critic 77.8M), fills the memory-efficient fp16 replay
with random rows (772 stored columns plus an int aux, the other 2316 rebuilt at sample time), then
times gradient updates of batch 4096. Under torchrun every rank fills its own replay and the updates
are data-parallel (multi_gpu).

  python benchmarks/sac_flash_learner.py [--rows 1000000] [--compile default] [--set update_cuda_graph=true]
  torchrun --standalone --nproc_per_node=4 benchmarks/sac_flash_learner.py --compile default --set ...
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

OBS, ACTIONS, STORED, WINDOW = 3088, 36, 772, 780


class RandomRebuild:
    """Keeps the first STORED obs columns; rebuilds the rest from a fixed table indexed by the aux ints."""
    compact_dim, aux_dim = STORED, 2

    def __init__(self, device):
        self.table = torch.randn(4096, OBS - STORED, device=device)

    def compact(self, full):
        return full[:, :STORED]

    def expand(self, compact, aux):
        idx = (aux[:, 0].long() * 7 + aux[:, 1].long()) % self.table.shape[0]
        return torch.cat([compact.float(), self.table[idx] * compact[:, :1].float()], dim=1)


def build(args, device, world):
    import gymnasium as gym
    from rl_games.torch_runner import Runner, _compile_sac_submodules
    env_info = {'observation_space': gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
                'action_space': gym.spaces.Box(-1.0, 1.0, (ACTIONS,), np.float32), 'agents': 1}
    config = {
        'name': 'sac_flash_learner', 'env_name': 'none', 'env_info': env_info, 'device': device,
        'multi_gpu': world > 1, 'num_actors': 2048, 'max_frames': 500_000_000, 'reward_shaper': {},
        'action_groups': [['0:18'], ['18:36']], 'gamma': 0.98, 'n_step': 5, 'critic_tau': 0.01,
        'batch_size': 4096, 'num_updates_per_step': 2, 'policy_frequency': 2, 'critic_grad_clip': 0,
        'replay_buffer_size': args.rows, 'replay_memory_efficient': True, 'replay_obs_dtype': 'float16',
        'replay_obs_compactor': RandomRebuild(device), 'init_alpha': 0.01, 'learnable_temperature': True,
        'min_alpha': 0.0, 'target_entropy_sigma': 0.15, 'actor_lr': 3e-4, 'critic_lr': 3e-4, 'alpha_lr': 3e-4,
        'lr_schedule': 'cosine', 'normalize_reward': True, 'mixed_precision': 'fp16', 'print_stats': False,
        'torch_compile': args.compile or False, 'train_dir': os.environ.get('TMPDIR', '/tmp') + '/sac_flash_learner',
    }
    for kv in args.set:
        key, value = kv.split('=', 1)
        config[key] = yaml.safe_load(value)
    network = {'name': 'flash_sac', 'actor': {'units': 128, 'blocks': 2},
               'critic': {'units': 1024, 'blocks': 2, 'num_bins': 101, 'v_min': -5.0, 'v_max': 5.0},
               'obs_window': {'size': WINDOW, 'steps': 10}}
    runner = Runner()
    runner.load({'params': {'seed': 42, 'algo': {'name': 'sac'}, 'model': {'name': 'soft_actor_critic'},
                            'network': network, 'config': config}})
    agent = runner.algo_factory.create('sac', base_name='bench', params=runner.params)
    if args.compile:
        _compile_sac_submodules(agent.model.sac_network, args.compile, args.compile,
                                False if config.get('update_cuda_graph') else None)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=1_000_000)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--compile', default='', help="torch_compile mode for the networks, e.g. 'default'")
    p.add_argument('--set', nargs='*', default=[], help='config overrides key=value')
    args = p.parse_args()
    rank, world = int(os.environ.get('LOCAL_RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    agent = build(args, device, world)
    agent._setup_multi_gpu()
    gen = torch.Generator(device=device).manual_seed(rank)
    n = agent.num_actors
    for _ in range(-(-args.rows // n) + agent.n_step + 1):
        obs = torch.randn(n, OBS, device=device, generator=gen)
        aux = torch.randint(0, 1 << 20, (n, 2), device=device, generator=gen, dtype=torch.int32)
        u = torch.rand(n, device=device, generator=gen)
        reward = torch.randn(n, 2, device=device, generator=gen)
        agent.replay_buffer.add(obs, torch.rand(n, ACTIONS, device=device) * 2 - 1, reward, obs,
                                (u < 1 / 300).unsqueeze(1), ((u > 1 - 1 / 300)).unsqueeze(1), obs_aux=aux,
                                next_obs_aux=aux)
        agent.reward_scaler.update(reward, u < 2 / 300)
    agent.set_train()
    for k in range(args.warmup):
        agent.update(k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(args.updates):
        agent.update(k)
        agent._head_stats.clear()
    torch.cuda.synchronize()
    ups = args.updates / (time.perf_counter() - t0)
    if rank == 0:
        print(json.dumps({'gpus': world, 'gpu': torch.cuda.get_device_name(), 'updates_per_s': round(ups, 2),
                          'ms_per_update': round(1000 / ups, 2), 'samples_per_s': round(world * ups * 4096),
                          'max_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)}))


if __name__ == '__main__':
    main()
