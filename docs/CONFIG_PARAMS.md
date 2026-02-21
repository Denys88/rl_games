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
