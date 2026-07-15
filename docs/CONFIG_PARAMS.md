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

## Adaptive LR (under `config:`)

### `schedule_type`

Stepping granularity for the KL-adaptive learning-rate scheduler
(`lr_schedule: adaptive`):

```yaml
config:
  lr_schedule: adaptive
  schedule_type: per_minibatch   # default; 'legacy' is a permanent alias
  kl_threshold: 0.008
  min_lr: 1e-5                   # ALWAYS set both bounds explicitly:
  max_lr: 1e-3                   # class defaults (1e-6 / 1e-2) are too wide
```

| Value | LR updates | KL input | When |
|-------|------------|----------|------|
| `per_minibatch` (alias `legacy`, **default**) | after every minibatch | that minibatch's KL | rl_games' original adaptive stepping (the old name marks its seniority; rsl-rl adopted the same mechanism) — tracks on-policy KL swings within a rollout. Requires reliable per-minibatch KL estimates: use large minibatches (16k+ on vectorized continuous control). |
| `standard` | once per mini-epoch | epoch-mean KL | Smoother; consider when minibatches are small (noisy KL estimates make per-minibatch stepping oscillate between the band edges). |
| `standard_epoch` | once per full epoch | epoch-mean KL | Coarsest. |

The practical failure modes to know: `per_minibatch` with *small* minibatches
rail-slams between `min_lr`/`max_lr` on KL-estimator noise (fix the minibatch
size, not the schedule); `standard` on tasks with fast on-policy KL swings
adapts too slowly and can leave measurable reward on the table.

Multi-GPU note: `per_minibatch` performs a KL all-reduce and an LR broadcast
per split per mini-epoch; `standard` does one per mini-epoch. On a single
node the difference is small; at multi-node latencies the extra collectives
compound — prefer `standard` there unless per-task evidence says otherwise.

## Sigma Parametrization (under `network: space: continuous:`)

### `sigma_parametrization`

How the sigma head's raw output `r` becomes the Gaussian policy's std.
Default `exp` (historical rl_games behavior, fully backward compatible).

| value | std | notes |
|---|---|---|
| `exp` | `exp(r)` | `r` is a log-std. Entropy bonus applies a **constant** upward force on `r` regardless of current sigma — on weak-reward tasks this can run away (sigma grows exponentially). |
| `softplus` | `softplus(r) + min_sigma` | smooth positive map with an additive floor. |
| `scalar` | `floor + softplus(r - floor)`, `floor = max(min_sigma, 1e-3)` | std-space: the head output *is* the std away from the floor (identity for `r >> floor`). Entropy pressure decays as `1/sigma`, so the same coefficient self-attenuates as sigma grows. Matches the reference `std_type="scalar"` distributions used by common locomotion recipes. |

The floor is a softplus, not a hard clamp, on purpose: a clamp has zero
gradient below the floor, stranding dimensions that drift under it (no
restoring gradient; observed to produce NaNs). The smooth floor keeps
`d(sigma)/dr > 0` everywhere.

**`sigma_init.val` units depend on the parametrization.** Under `exp` it is
a log-std (`val: 0.0` → σ₀ = 1). Under `scalar` it is in std units but the
floor shifts it: σ₀ = `floor + softplus(val - floor)` — e.g. `val: 1.0`,
`min_sigma: 0.05` gives σ₀ ≈ 1.29, not 1.0. To hit a target σ₀ exactly:
`val = floor + softplus_inverse(σ₀ - floor)` (`val: 0.511` → σ₀ ≈ 1.0 at
floor 0.05).

**Entropy coefficients do not transplant across parametrizations** — the
same number is a constant log-space force under `exp` and a `1/sigma`-
decaying std-space force under `scalar`. Retune when switching.
