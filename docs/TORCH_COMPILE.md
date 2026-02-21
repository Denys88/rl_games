# torch.compile Support in RL-Games

## Overview

RL-Games supports PyTorch's `torch.compile` for enhanced performance, providing 10-40% improvement in training throughput depending on model architecture, batch size, and hardware configuration.

## Configuration

Add `torch_compile` to your YAML config under `params.config`:

```yaml
# Enable with default mode - recommended for most cases
torch_compile: true

# Disable compilation
torch_compile: false

# Specify mode as string
torch_compile: "default"         # Kernel fusion + operator optimization (default)
torch_compile: "reduce-overhead" # Adds CUDA graph capture/replay (FF models only)
torch_compile: "max-autotune"    # Best runtime, longest compilation (FF models only)

# Advanced configuration (dict format)
torch_compile:
  mode: "reduce-overhead"        # Actor/main model mode
  critic_mode: "default"         # Central value model mode (optional, see Notes below)
```

## Performance Modes

- **`default`** (default): Kernel fusion, operator optimization, and graph-level optimizations. Recommended for all model types.
- **`reduce-overhead`**: Adds CUDA graph capture/replay on top of `default`. **Not compatible with RNN/LSTM models** (see warning below).
- **`max-autotune`**: Maximum runtime performance with extensive kernel tuning. First epoch will be significantly slower (~2-5x). **Not compatible with RNN/LSTM models** (see warning below).

### WARNING: RNN/LSTM Incompatibility

`reduce-overhead` and `max-autotune` modes use CUDA graphs, which are **incompatible with RNN/LSTM models**. Two issues arise:

1. **Rollout buffer corruption**: CUDA graph output tensors are graph-owned buffers that get overwritten on the next graph replay. In sequential RNN rollout loops, each timestep's outputs get corrupted by the next call.
2. **Backward pass failure**: LSTM internal allocations create tensors outside the CUDA graph memory pool, causing `RuntimeError: storage data ptrs are not allocated in pool` during `loss.backward()`.

**Use `default` mode for any config with RNN/LSTM** (including asymmetric actor-critic with LSTM central value networks). For feed-forward models, `reduce-overhead` works well.

## Compilation Overhead

The first epoch will be slower due to JIT compilation overhead:
- **`default`**: ~10-20% slower first epoch
- **`reduce-overhead`**: ~20-50% slower first epoch
- **`max-autotune`**: ~100-400% slower first epoch (tries many kernel variants)

CUDA Graph warnings during startup are normal and can be ignored.

## When to Use Each Mode

**Use `default` (recommended) when:**
- Training any model type (FF or RNN/LSTM)
- You want reliable performance across all configurations
- Iterating during development/tuning

**Use `reduce-overhead` when:**
- Training **feed-forward only** models (no RNN/LSTM)
- Training for moderate to long duration (100-10,000 epochs)
- You want the best performance for FF models

**Use `max-autotune` when:**
- Training **feed-forward only** models for very long runs (10,000+ epochs)
- You can afford 2-5x slower startup for maximum steady-state performance

## Advanced Configuration

### Separate Actor and Critic Modes

`critic_mode` only applies when using `central_value_config`, which enables asymmetric actor-critic training:

```yaml
torch_compile:
  mode: "reduce-overhead"        # Actor network
  critic_mode: "default"         # Critic network (compiles faster)
```

**When is central_value_config used?**
- **Asymmetric actor-critic**: Actor and critic see different observations (e.g., critic gets privileged state information)
- **Multi-agent centralized critic**: Critic sees all agents' observations + global state

For standard PPO/SAC, only the main model is compiled (which includes both actor and critic heads).

**Note:** `separate: true` in network config creates separate network towers for actor/critic, but they're still compiled together as one model unless using `central_value_config`.

## Performance Expectations

Typical speedup over non-compiled training:

| Mode | First Epoch | Steady State | Use Case |
|------|-------------|--------------|----------|
| `default` | 0.9x | 1.10-1.15x | All models (recommended) |
| `reduce-overhead` | 0.5-0.8x | 1.15-1.30x | FF models only |
| `max-autotune` | 0.2-0.5x | 1.25-1.40x | FF models, long training |

Actual speedup varies significantly based on:
- **Model architecture**: Larger models benefit more from compilation
- **Batch size**: Larger batches enable better kernel optimization
- **Hardware**: Newer GPUs (Ampere/Hopper) show larger improvements
- **Environment**: Faster environments show smaller relative speedup (more time in env.step)

## Troubleshooting

### Compilation Errors

If you encounter compilation errors:
1. Set `torch_compile: false` to disable compilation
2. Update to PyTorch 2.2 or newer
3. Check for custom operations that may not support torch.compile

### CUDA Graph Errors with RNN/LSTM

If you see errors like:
- `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`
- `RuntimeError: storage data ptrs are not allocated in pool`

Switch to `default` mode:
```yaml
torch_compile: "default"
```

### Performance Regression

If compiled code is slower:
- Ensure batch size is large enough (>256 recommended)
- Try `mode: "default"` for smaller models
- Profile to verify bottleneck is in RL update, not environment
