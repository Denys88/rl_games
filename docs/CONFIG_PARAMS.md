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

## Precision (under `config:`)

### `mixed_precision`

Runs the training forward pass under bf16 autocast. Default: `False`. Applies to: PPO, the central value network, SAC.

With `False`, matmuls run in TF32 on GPUs that support it, because rl_games enables TF32 at startup.

Keep it off for continuous control with `lr_schedule: adaptive`. bf16 rounds the policy mean by up to 0.4 %. Once sigma drops below about 0.05, that rounding alone is a KL of 0.01 to 0.03 per update at any learning rate, and the adaptive schedule lowers the rate to `min_lr`. Rollouts run without autocast, so under bf16 the rollout policy and the updated policy also differ before the first gradient step.

| MicroDuck task, 4096 envs | bf16 | TF32 |
|---|---|---|
| Velocity, final return (2 seeds) | 136.1, 138.3 | 149.7, 148.1 |
| Ball walk, final return (seed 42) | 13.8 | 25.0 |
| Training throughput, frames/s | 130.1k | 133.4k |

Set it to `True` only when the network update dominates the iteration time, and check that the logged learning rate does not settle at `min_lr`.

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
| `per_minibatch` (alias `legacy`, **default**) | after every minibatch | that minibatch's KL | Responds within an optimization pass; noisy estimates can cause frequent rate changes. |
| `standard` | once per mini-epoch | epoch-mean KL | Smoother; consider when minibatches are small (noisy KL estimates make per-minibatch stepping oscillate between the band edges). |

The cadence trades responsiveness against noise. Neither setting is
universally better; compare them with the same minibatch geometry and KL
measurement. `lr_multiplier` (default `1.5`) controls each rate adjustment:
divide by it above `2 * kl_threshold`, multiply by it below
`0.5 * kl_threshold`, and otherwise leave the rate unchanged, subject to
`min_lr` / `max_lr`. A smaller multiplier such as `1.1` changes the rate more
gradually; it is a tuning choice, not a validated improvement for every task.

**Scope:** `schedule_type` applies to continuous PPO only. Discrete PPO always
steps its adaptive scheduler once per mini-epoch on the mean KL (i.e.
`standard`-like) and ignores this key; `per_minibatch` stepping for discrete is
a possible future addition.

Multi-GPU note: `per_minibatch` performs a KL all-reduce and an LR broadcast
per split per mini-epoch; `standard` does one per mini-epoch. On a single
node the difference is small; at multi-node latencies the extra collectives
compound — prefer `standard` there unless per-task evidence says otherwise.

### `kl_reference`

Continuous PPO can choose which Gaussian the reference KL compares against.
The KL direction remains `KL(current || reference)`. This is the scheduler
signal when `kl_schedule_source: reference` (the default).

| Value | Reference distribution |
|-------|------------------------|
| `previous_mini_epoch` (default) | The distribution last evaluated on each row; initially the rollout policy, then refreshed on each optimization pass. Preserves historical behavior. |
| `rollout` | The fixed policy that collected the batch, throughout all mini-epochs. Measures cumulative drift using the same reference as PPO's likelihood ratio. |

`rollout` can reduce the adaptive learning rate sooner; treat it as an
explicit training ablation rather than assuming the same learning curve.
This is a measurement/controller option, not a hard KL limit or early stop.
The scheduler cannot lower the rate below the configured `min_lr`.

### `kl_schedule_source`

Selects the continuous PPO KL measurement passed to the learning-rate
scheduler. `schedule_type` still determines when the scheduler changes the
rate; this option does not change PPO's fixed rollout log-probability
reference or its loss.

| Value | Scheduler measurement |
|-------|-----------------------|
| `reference` (default) | Mean `KL(current || reference)` from the pre-optimizer forward pass, using `kl_reference`. Preserves existing behavior. |
| `optimizer_step` | Mean `KL(post_step || pre_step)` on the same valid minibatch rows, with the same observation-normalization statistics for both distributions. Measures the change caused by that optimizer step. |

`optimizer_step` requires an extra forward pass without gradients on every
rank. It supports feedforward continuous PPO; recurrent policies and
discrete PPO do not support it. It measures the combined effect of all
losses updating the actor, including a shared auxiliary value loss. It does
not include observation-statistics changes made before the pre-step forward,
and does not constrain cumulative drift over a mini-epoch or rollout. The
measurement arrives after the step and adjusts subsequent updates; there is
no rollback or hard KL limit.

Using it: the step KL is a different scale from the reference KL, so
`kl_threshold` must be re-calibrated, not carried over. Read
`info/scheduler_kl` from a fixed-rate run of the same task and set the target
so that the dead band (`0.5 .. 2` times the target) contains the healthy
run's values in every phase; on the WujiHand recipe that is 0.002 (0.002
before takeoff, 0.003 after). Pair it with `schedule_type: standard`: the
per-minibatch step KL has a heavy upper tail, and per-minibatch stepping
brakes on every tail event while nothing ever triggers an increase, so the
rate ratchets down to `min_lr` (measured on WujiHand: 19.3 vs 20.1 reaches).

The actor's distribution parameters must be deterministic for a matched
input to isolate optimizer movement. Custom stochastic forwards (for example,
dropout) need matched randomness; this path does not replay RNG state. An
extra forward can also consume RNG in models that sample entropy estimates.
The Wuji Gaussian actor has neither behavior.

Calibrate `kl_threshold` for this measurement from a known working fixed-rate
run or saved-batch replay. A threshold tuned for rollout-relative or
previous-pass KL is not interchangeable with a single-step threshold.
The extra measurement also runs with a fixed learning rate when
`kl_schedule_source: optimizer_step`, so calibration does not require
turning on adaptation or `use_diagnostics`.
`schedule_type: standard` with `lr_multiplier: 1.1` is a candidate for slower
adaptation, to compare against the fixed-rate baseline after calibration.
Keep frames, minibatches, mini-epochs, seeds and instrumentation matched;
include the extra forward's cost in wall-clock comparisons. The shipped
WujiHand recipe remains fixed-rate.

`info/kl` retains the reference-KL meaning selected by `kl_reference`.
`info/scheduler_kl` reports the epoch mean of the selected scheduler signal.
`info/lr_mean`, `info/lr_min` and `info/lr_max` summarize the rates actually
used for minibatch updates; `info/lr_at_min_fraction` and
`info/lr_at_max_fraction` show the fractions of updates at the adaptive
bounds. These distinguish applied rates from the final rate scheduled for
the next update.

### Central critic and `use_experimental_cv`

With `central_value_config`, continuous PPO obtains rollout values from the
privileged central critic. By default, `use_experimental_cv: true` also
trains a value head in the actor model. With a shared actor trunk, this
auxiliary loss changes the policy outside PPO's surrogate clipping.
`use_experimental_cv: false` disables that auxiliary loss while retaining
the central critic. It does not disable value training when no central
critic is configured.

### Policy diagnostics

`use_diagnostics: true` adds per-mini-epoch metrics under
`diagnostics/policy/<name>/<mini_epoch>`, in two groups.

In-batch, from the learner's forward pass before the optimizer step:
`sigma_min`, `sigma_mean`, `sigma_max`, `mu_abs_max`, `advantage_abs_max`,
`logratio_abs_max` (uses the fixed rollout log probability regardless of
`kl_reference`), `sigma_score_pos`, `sigma_score_neg`, `sigma_score_mean`
(the signed log-std score `A * (z^2 - 1)` averaged over samples with positive,
negative and any advantage; positive means the surrogate pushes the std up on
those samples) and `tail_frac` (fraction of samples with `max_i |z_i| > 3`,
`z = (a - mu) / sigma`).

Post-step, from one extra no-grad forward on the same minibatch after
`optimizer.step()`, on rank 0 only, through the raw module with the
observation normaliser's statistics frozen: `kl_step` (KL(pre-step ||
post-step)), `kl_post_ref` (mean KL(post-step || reference), the reference
being the Gaussian `kl_reference` selects) and `kl_post_ref_max` (its
per-sample max). Not available for recurrent policies.

All metrics cover valid learner minibatch rows and are local to the writer's
rank under DDP. Diagnostics add reductions, device synchronizations and, for
the post-step group, one forward per minibatch, so account for this when
measuring throughput. They are disabled by default.

## Sigma Parametrization (under `network: space: continuous:`)

### `max_sigma`

Optional smooth ceiling on the policy std, applied after any parametrization
(`x = (sigma - min_sigma) / (max_sigma - min_sigma)`, `sigma = min_sigma + (max_sigma - min_sigma) * x / (1 + x)`).
Default `0` (off). The squash compresses the whole range, not only the top: with `min_sigma 0.2` and
`max_sigma 1.0` a raw std of 0.3 becomes 0.29, 0.5 becomes 0.42 and 0.8 becomes 0.54. It is exact only at
the floor, saturates at `max_sigma`, and keeps a (polynomially decaying) gradient above it. Read, like
`min_sigma`, by the `actor_critic` and `resnet_actor_critic` builders; `max_sigma <= min_sigma` is
rejected when the network is built. It bounds the exploration noise only; the policy mean is not
bounded. Use it when a state-dependent sigma head can extrapolate on rare states: for actions in
`[-1, 1]` a cap of `1.0` is a natural choice (see the MJLab WujiHand notes). With `min_sigma 0.2`,
`max_sigma 1.0` and `sigma_init -1.05` the initial std is 0.42 instead of 0.5.

### `sigma_parametrization`

How the sigma head's raw output `r` becomes the Gaussian policy's std.
Default `exp` (historical rl_games behavior, fully backward compatible).

| value | std | notes |
|---|---|---|
| `exp` | `exp(r)` | `r` is a log-std. Entropy bonus applies a **constant** upward force on `r` regardless of current sigma — on weak-reward tasks this can run away (sigma grows exponentially). |
| `softplus` | `softplus(r) + min_sigma` | smooth positive map with an additive floor. |
| `linear` (alias: `scalar`) | `floor + softplus(r - floor)`, `floor = max(min_sigma, 1e-3)` | std-space: the head output *is* the std away from the floor — asymptotically: the identity holds for `r >> floor`, with a maximum deviation of `ln 2 ≈ 0.693` at `r = floor` (e.g. `r = 1.0`, floor `0.05` gives sigma ≈ 1.29, not 1.0). `scalar` is the reference-compat alias (rsl-rl-lineage `std_type="scalar"` = std-space naming). Entropy pressure decays as `1/sigma`, so the same coefficient self-attenuates as sigma grows. |

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
decaying std-space force under `linear`. Retune when switching.
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
  statistics of the pooled global stream. Exact at any world size up to one
  startup artifact: each rank's mean-0/var-1 initialization prior is counted
  once, so a fresh merge carries `world_size` prior pseudo-samples instead
  of one (relative effect ~1e-5 against real per-epoch batches, decaying as
  1/epoch). The default.
- `'broadcast'`: every rank adopts rank 0's statistics (standard DDP
  `broadcast_buffers` semantics). Stateless and idempotent, but estimator
  variance and within-update drift grow ~linearly with world size (at fixed
  per-rank batch geometry) — fine at 2 ranks, prefer `'pooled'` at 8+.

A/B at 2 ranks (envpool Pong, 3 back-to-back seed pairs, 400 epochs):
parity — pooled 19.46 ± 0.38 vs broadcast 18.96 ± 0.28, paired p = 0.398.

### `multi_gpu_grad_sync`

**Type:** str | **Default:** `'ddp'` | **Options:** `'ddp'`, `'flat_allreduce'` | **Applies:** multi-GPU (torchrun) PPO runs

- `'ddp'`: gradients are averaged by `DistributedDataParallel` during
  backward (bucketed, overlapped with compute). The training forward runs
  through the DDP wrapper built once by `setup_multi_gpu()` at the start of
  `train()` and exposed via `train_model()`; rollout inference,
  checkpoints and attribute access keep using the raw model, so state_dict
  keys are unchanged. The default.
- `'flat_allreduce'`: the pre-2.x manual path — one flat all-reduce of all
  gradients after backward. Kept for one release as a compatibility escape
  hatch for downstream agents that override `calc_gradients()` and forward
  through `self.model` directly; scheduled for removal.

2-GPU A/B (Isaac Humanoid, 16k envs/rank): bit-identical gradients between
modes. Throughput with working GPU peer-to-peer: `'ddp'` is +1-3% step and
total fps at the default `[512, 256, 128]` net on GPU-pipeline sims, within
noise on CPU-bound sims and on the ~44 MB-gradient net, and never
meaningfully worse; not measured beyond 2 GPUs. Only on host-staged links
without P2P can `'flat_allreduce'` come out ~4% ahead on small nets. A
multi-GPU run that never routes its training forward through
`train_model()` under `'ddp'` raises at the optimizer step instead of
silently training rank-divergent.

### `ddp_find_unused_parameters`

**Type:** bool | **Default:** `False` | **Applies:** multi-GPU PPO runs under `multi_gpu_grad_sync: 'ddp'` (agent and central value net)

Passed through to `DistributedDataParallel(find_unused_parameters=...)` for
both DDP wrappers under `multi_gpu_grad_sync: 'ddp'`: the top-level key
(default `False`) applies to the PPO agent's model and is the fallback for the
central value net; `central_value_config.ddp_find_unused_parameters` overrides
it for the central value net alone. Set it to `True` when a network contains
heads whose outputs its forward discards — with the default, DDP's reducer
errors at the start of the second iteration because those parameters never
receive gradients. Leave it off otherwise: unused-parameter discovery costs a
graph walk every iteration.

```yaml
config:
  ddp_find_unused_parameters: True      # agent model; the central value net inherits it
  central_value_config:
    ddp_find_unused_parameters: False   # central value net only; overrides the top-level key
```

**Scope:** PPO agent and central value net under `multi_gpu_grad_sync: 'ddp'`;
the `central_value_config` key overrides the top-level one. No effect under
`'flat_allreduce'`.

### `multi_gpu_scheduler_kl`

**Type:** str | **Default:** `'global'` | **Options:** `'global'`, `'local'` | **Applies:** multi-GPU PPO runs (continuous: `schedule_type: per_minibatch`; discrete: its per-mini-epoch stepping)

- `'global'`: the adaptive-LR scheduler sees the cross-rank mean KL (one
  all-reduce per scheduler step). The default and historical behavior.
- `'local'`: the scheduler uses rank 0's local KL estimate, dropping one
  collective per scheduler step. The KL sample size the scheduler sees drops
  by `world_size`; lr is broadcast from rank 0 either way, so ranks never
  diverge. Worth trying when per-rank minibatches are large and profiling
  shows the update phase rendezvous-bound. For discrete PPO under `'local'`
  the logged per-mini-epoch KL is rank 0's local mean as well.

### `capability_manifest`

**Type:** any | **Default:** unset | **Applies:** PPO and SAC full-state checkpoints

Free-form metadata that travels with the policy: whatever this key holds is
saved into the checkpoint and restored on load. rl_games never interprets
it. Use it for anything a checkpoint consumer needs to know about the
policy, for example:

```yaml
config:
  capability_manifest:
    command_ranges: [{quantity: linear_velocity_x, min: -1.5, max: 1.5}]
    terrain_classes: [rigid]
```

On restore, a manifest already declared in the config takes precedence over
the checkpoint's (a warning is printed if they differ).

## Observation Normalization (under `config:`)

### `normalize_input`

Enables online observation normalization: a `RunningMeanStd` tracks per-dimension
mean/variance of the observations and the model normalizes every input with the
current stats. Stats update on each training minibatch forward (train mode) and
are frozen during rollouts and evaluation; they are part of the checkpoint.

```yaml
config:
  normalize_input: True
```

**Default:** `False`. Applies to the PPO agents (`a2c_continuous`,
`a2c_discrete`) and, when configured in `central_value_config`, to the central
value network. SAC has its own `normalize_input` handling and is not affected by
the warm-start parameter below.

### `normalize_input_init_count`

Seeds the sample count of the obs normalizer's zero-mean/unit-variance prior.
With the legacy count of 1, the first training minibatch outweighs the prior
thousands to one and the running stats jump straight to the batch stats; the
policy recomputed under the shifted stats diverges from the rollout policy that
collected the data (a large first-epoch KL — with `lr_schedule: adaptive` this
can floor the LR at `min_lr` in epoch 1 before learning starts). A larger count
turns that jump into a damped update.

```yaml
config:
  normalize_input: True
  normalize_input_init_count: 81920   # explicit prior weight (int or 8.2e4)
  # normalize_input_init_count: 1     # legacy cold start
  # omit or null                      # auto: one PPO epoch of samples (default)
```

**Default (`null`/absent):** for the agents' input normalizer,
`mini_epochs * horizon_length * num_actors * num_agents` — one PPO epoch of
*counted* samples (the normalizer updates on every training minibatch, so its
count accrues `mini_epochs`× the rollout size per epoch; the derivation matches
that accounting). The central value network derives its default from its **own**
geometry instead: `central_value_config.mini_epochs * horizon_length *
num_actors` (its state batch is per-env, not per-agent). The prior then fades
after roughly one epoch. Values below 1 are rejected (a 0 or negative count poisons
the running stats).

**Multi-GPU:** with pooled stats sync, the first merge sums every rank's seeded
prior, so the effective prior weight is exactly one *global* epoch — consistent
with the single-GPU behavior.

**Scope and follow-ups:** an explicit top-level value seeds the *input*
normalizer of the PPO agents **and** the central value network (as before);
`normalize_input_init_count` inside `central_value_config` overrides it for the
central value net alone. Only the *defaults* are derived per-network (agent and
CV geometry respectively, see above). `value_mean_std` (`normalize_value`) has the
same cold start and is a planned follow-up; SAC is unaffected. Note the warm
start damps the first-epoch stat jump but does not change the update scheme
itself — stats still drift within each epoch (updated per minibatch over the
same rollout data); collection-time freezing à la VecNormalize would remove
that drift entirely and remains a possible future change.
