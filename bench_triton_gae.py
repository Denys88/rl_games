"""Benchmark: Triton GAE vs PyTorch GAE.

Compares the custom Triton kernel against the original Python loop
implementation from a2c_common.py for correctness and performance.
"""

import torch
import time
import argparse


def pytorch_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau):
    """Original PyTorch GAE from a2c_common.py (Python loop)."""
    horizon_length = mb_rewards.shape[0]
    mb_advs = torch.zeros_like(mb_rewards)
    lastgaelam = 0

    for t in reversed(range(horizon_length)):
        if t == horizon_length - 1:
            nextnonterminal = 1.0 - last_dones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_dones[t + 1]
            nextvalues = mb_values[t + 1]
        nextnonterminal = nextnonterminal.unsqueeze(1)

        delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam
    return mb_advs


def benchmark(fn, *args, warmup=10, iters=100, label=""):
    """Benchmark a function with CUDA sync."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000
    print(f"  {label:30s}: {avg_ms:8.3f} ms  ({iters} iters)")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton vs PyTorch GAE")
    parser.add_argument('--device', default='cuda', help='Device to benchmark on')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU (Triton won't work)")
        device = 'cpu'

    from rl_games.algos_torch.triton_kernels import triton_gae

    configs = [
        # (num_envs, horizon_length, value_size, label)
        (64,    32,  1, "Small  (64 envs, H=32)"),
        (256,   32,  1, "Medium (256 envs, H=32)"),
        (1024,  32,  1, "Large  (1024 envs, H=32)"),
        (4096,  32,  1, "XL     (4096 envs, H=32)"),
        (4096,  64,  1, "XL     (4096 envs, H=64)"),
        (16384, 32,  1, "XXL    (16384 envs, H=32)"),
        (4096,  24,  1, "MJLab  (4096 envs, H=24)"),
        (8192,  16,  1, "Isaac  (8192 envs, H=16)"),
    ]

    gamma = 0.99
    tau = 0.95

    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Correctness check first
    print("=" * 70)
    print("CORRECTNESS CHECK")
    print("=" * 70)
    for num_envs, horizon, value_size, label in configs[:3]:
        mb_rewards = torch.randn(horizon, num_envs, value_size, device=device)
        mb_values = torch.randn(horizon, num_envs, value_size, device=device)
        mb_dones = (torch.rand(horizon, num_envs, device=device) > 0.95).float()
        last_values = torch.randn(num_envs, value_size, device=device)
        last_dones = (torch.rand(num_envs, device=device) > 0.95).float()

        ref = pytorch_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau)
        tri = triton_gae(mb_rewards, mb_values, mb_dones, last_values, last_dones, gamma, tau)

        max_diff = (ref - tri).abs().max().item()
        mean_diff = (ref - tri).abs().mean().item()
        print(f"  {label}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}",
              "PASS" if max_diff < 1e-4 else "FAIL")

    print()
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    for num_envs, horizon, value_size, label in configs:
        print(f"\n{label}:")
        mb_rewards = torch.randn(horizon, num_envs, value_size, device=device)
        mb_values = torch.randn(horizon, num_envs, value_size, device=device)
        mb_dones = (torch.rand(horizon, num_envs, device=device) > 0.95).float()
        last_values = torch.randn(num_envs, value_size, device=device)
        last_dones = (torch.rand(num_envs, device=device) > 0.95).float()

        pt_ms = benchmark(pytorch_gae, mb_rewards, mb_values, mb_dones,
                          last_values, last_dones, gamma, tau, label="PyTorch (loop)")
        tri_ms = benchmark(triton_gae, mb_rewards, mb_values, mb_dones,
                           last_values, last_dones, gamma, tau, label="Triton (kernel)")

        speedup = pt_ms / tri_ms
        print(f"  {'Speedup':30s}: {speedup:8.2f}x")


if __name__ == '__main__':
    main()
