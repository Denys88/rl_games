# YAML Config Parameters

## Top-level Parameters (under `params:`)

### `torch_threads`

Controls the number of CPU threads used for **intra-op parallelism** in PyTorch (OpenMP/MKL threads for CPU-side tensor operations).

```yaml
params:
  torch_threads: 4  # Explicit thread count
  # torch_threads: 0  # Disable — use PyTorch default (os.cpu_count())
  # omit torch_threads for auto-detection (recommended)
```

**Default behavior (auto):** If not specified, computes `min(4, cpu_cores // world_size)`. This avoids CPU oversubscription in multi-GPU (DDP) setups while keeping enough threads for CPU-side work.

| Scenario | `torch_threads` | Effective Threads per Process |
|----------|----------------|-------------------------------|
| 1 GPU, 16 cores | auto | 4 |
| 2 GPUs (DDP), 16 cores | auto | 4 (min(4, 16/2=8)) |
| 8 GPUs (DDP), 16 cores | auto | 2 (min(4, 16/8=2)) |
| 1 GPU, 2 cores | auto | 2 (min(4, 2/1=2)) |
| Any setup | `0` | PyTorch default (all cores) |
| Any setup | `8` | 8 |

**When to adjust:**
- **GPU environments (Isaac Gym, Isaac Lab):** Default auto is fine. Most work happens on GPU; CPU threads handle data prep and tensor ops.
- **Ray-based environments:** Default auto is fine for the trainer process. Ray workers are separate processes with their own thread pools.
- **CPU-heavy training:** Set higher (e.g., `torch_threads: 8`) if your training loop has significant CPU tensor operations.
- **Disable:** Set `torch_threads: 0` to let PyTorch use all available cores (not recommended for multi-GPU).

**DDP note:** Each DDP process (launched via `torchrun`) runs `load_config` independently and gets its own thread pool. With auto-detection, each process accounts for `world_size` to avoid oversubscription.

**Ray note:** `torch.set_num_threads()` only affects the trainer process. Ray workers (`RayWorker`) are separate processes that use their own default thread count. This setting does NOT propagate to Ray workers.

### `multi_gpu_sync_stats` (config section)

**Type:** bool | **Default:** `True` | **Applies:** multi-GPU (torchrun) PPO runs

Synchronize obs/value running-normalization statistics across ranks each
epoch. Without it every rank's normalizers drift on their local shard, so
ranks train subtly different models whose averaged gradients conflict
(measured: envpool Pong, 2 ranks, 86.9 vs 94.8 mean reward at epoch 2000).
Enabled by default — this changes multi-GPU behavior relative to rl_games
< 2.0; set `multi_gpu_sync_stats: False` to restore the old (unsynced)
behavior.

### `multi_gpu_sync_stats_mode`

**Type:** str | **Default:** `'pooled'` | **Options:** `'pooled'`, `'broadcast'`

- `'pooled'`: moment-based merge of per-epoch deltas — every rank gets the
  statistics of the pooled global stream. Statistically exact at any world
  size; the default.
- `'broadcast'`: every rank adopts rank 0's statistics (standard DDP
  `broadcast_buffers` semantics). Stateless and idempotent, but estimator
  variance and within-update drift grow ~linearly with world size — fine
  at 2 ranks, prefer `'pooled'` at 8+.

A/B at 2 ranks (envpool Pong, 3 back-to-back seed pairs, 400 epochs):
parity — pooled 19.46 ± 0.38 vs broadcast 18.96 ± 0.28, paired p = 0.398.
