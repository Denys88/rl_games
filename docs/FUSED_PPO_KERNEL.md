# Fused Triton PPO Loss Kernel

The PPO update step in rl_games traditionally executes ~30–60 small elementwise
and reduction CUDA kernels per minibatch (neglogp, ratio, clipped surrogate,
clipped value loss, entropy, bounds loss, masking, reductions, policy KL — each
with its autograd backward). On fast GPU-vectorized environments this launch
overhead is a measurable part of the update step.

`rl_games/triton_kernels/ppo_loss_kernel.py` replaces all of it with **one
fused Triton forward kernel and one analytic backward kernel**:

- **Continuous** (`a2c_continuous`): the kernel consumes the raw policy outputs
  (`mu`, `sigma`, `values`) and fuses the Gaussian neglogp, PPO clipped
  surrogate (hard or smooth clamp), clipped value loss, Gaussian entropy,
  bounds loss (`bound` / `regularisation`), RNN masking, all mean reductions
  and the Gaussian policy KL into a single launch. The backward kernel writes
  analytic `d loss / d mu`, `d sigma`, `d values` directly.
- **Discrete / multi-discrete** (`a2c_discrete`): the categorical
  neglogp/entropy stay in the model (they depend on action-space structure and
  action masking); the kernel fuses everything downstream — surrogate, value
  loss, entropy weighting, masking, reductions and the `0.5 * (old_nlp - nlp)^2`
  KL proxy.

## Usage

Enabled by default when `triton` is installed and training runs on CUDA.

```yaml
config:
  use_fused_ppo_kernel: True   # default; set False to use the eager path
```

Global kill-switch (also disables the Triton GAE kernel):

```bash
export RLG_NO_TRITON=1
```

On CPU/MPS or without triton, `fused_ppo_loss()` silently falls back to a
numerically identical eager PyTorch implementation.

## Feature parity

The fused path supports every option of the eager continuous/discrete loss:

| Feature | Config key |
|---|---|
| PPO clipped surrogate / plain PG | `ppo` |
| Smooth ratio clamp | `use_smooth_clamp` |
| Clipped / unclipped value loss | `clip_value` |
| Bounds loss (`bound`, `regularisation`, none) | `bound_loss_type`, `bounds_loss_coef` |
| Entropy bonus | `entropy_coef` |
| RNN sequence masks | (automatic with RNN nets) |
| Multi-head value (`value_size` > 1) | (automatic) |
| Central value / no value loss | `central_value_config` |
| Adaptive LR on policy KL | `lr_schedule: adaptive` (KL comes out of the same kernel launch) |
| Mixed precision | `mixed_precision` (kernel computes in fp32 via `torch.amp.custom_fwd`) |

Gradient conventions (subgradient of `max` at ties = 0.5/0.5, inclusive clamp
boundaries) follow PyTorch autograd exactly. `tests/test_fused_ppo_loss.py`
verifies losses, KL and gradients against the eager implementation across the
full feature matrix (38 parity tests).

## Performance

Loss + backward section only, RTX 5090, fp32 (eager path ≈ 60 kernel launches,
fused path 2 launches + one tiny reduction):

| Batch | Actions | Eager | Fused | Speedup |
|---|---|---|---|---|
| 8192 | 12 | 1.32 ms | 0.49 ms | 2.7x |
| 32768 | 12 | 1.32 ms | 0.51 ms | 2.6x |
| 65536 | 21 | 1.35 ms | 0.51 ms | 2.6x |

Both paths are launch-overhead-bound at these sizes; the fused kernel removes
that overhead, and the gain repeats `mini_epochs * num_minibatches` times per
training epoch. End-to-end gains depend on how large the loss section is
relative to network forward/backward and simulation.

## Experimental: fusing the network too

`rl_games/triton_kernels/mlp_kernel.py` extends the idea to the actor-critic
network itself: the standard 3-layer ELU MLP trunk plus `mu`/`value` heads run
as **one forward kernel** (inter-layer activations staged through L2-resident
scratch) and **one analytic backward kernel** (weight/bias grads accumulated
with fp32 atomics, dW tiles chunked to fit shared memory). Chained with the
fused loss, the whole PPO minibatch update — network forward, loss
forward/backward, network backward — is a **4-kernel pipeline** (optimizer
excluded; `fused=True` Adam is already a single kernel).

Measured on RTX 5090, TF32 (rl_games default), obs=36, units=[256,128,64],
actions=8, verified against autograd to ~1e-5 in IEEE mode
(`tests/test_fused_mlp_kernel.py`):

| Section | Batch | Eager | torch.compile (cudagraph) | Fused | Speedup |
|---|---|---|---|---|---|
| MLP fwd+bwd | 8192 | 0.69 ms | 0.58 ms | 0.22 ms | 3.1x / 2.6x |
| MLP fwd+bwd | 32768 | 0.71 ms | 0.59 ms | 0.57 ms | 1.2x / 1.0x |
| full update chain | 8192 | 1.82 ms | 1.83 ms | ~0.72 ms | ~2.5x |
| full update chain | 32768 | 1.82 ms | 1.74 ms | ~0.72 ms | ~2.5x |

At large batch the backward becomes atomic/compute-bound and the MLP-only gap
vs CUDA-graphed torch closes; the full-chain advantage (~2.5x) persists because
the loss section and inter-op gaps are gone entirely.

Why it is not wired into the agents yet:

- applies only to plain MLP actor-critic nets (`separate: False`, no
  CNN/RNN, flat observations); the current 3-layer trunk is a prototype
  constraint (generalizing layer count means one compiled variant per depth);
- `normalize_input` (RunningMeanStd) updates its statistics inside the train
  forward — that update has to be replicated or folded into layer 1;
- weights live transposed (`[in, out]`) relative to `nn.Linear`, so
  integration needs either synced transposed copies per optimizer step or
  checkpoint-compatible transposed storage;
- grad clipping / multi-GPU reduce operate on `param.grad`, so grads must be
  scattered back into the module parameters.

None of these is fundamental; it is engineering. The measured ceiling —
another ~2.5x on the update math on top of the fused loss — says the follow-up
is worth doing for GPU-vec-env workloads with high `mini_epochs`.

## Notes

- Advantage normalization and GAE are upstream of the minibatch loss; GAE has
  its own Triton kernel (`rl_games/triton_kernels/gae_kernel.py`).
- NaN/Inf propagation matches eager PyTorch (e.g. a fully-off-policy ratio
  overflowing to `inf` produces the same `NaN` gradients in both paths).
- `torch.compile` of the model composes with the fused kernel: the loss is
  outside the compiled model graph.
