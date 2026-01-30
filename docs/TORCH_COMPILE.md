# torch.compile Support in RL-Games

## Overview

RL-Games supports PyTorch's `torch.compile` for enhanced performance, providing 10-40% improvement in training throughput depending on model architecture, batch size, and hardware configuration.

## Configuration

Add `torch_compile` to your YAML config under `params.config`:

```yaml
# Enable with default mode (reduce-overhead) - recommended for most cases
torch_compile: true

# Disable compilation
torch_compile: false

# Specify mode as string
torch_compile: "default"         # Fastest compilation, moderate speedup
torch_compile: "reduce-overhead" # Balanced performance and compilation time (default)
torch_compile: "max-autotune"    # Best runtime performance, longest compilation

# Advanced configuration (dict format)
torch_compile:
  mode: "reduce-overhead"        # Actor/main model mode
  critic_mode: "default"         # Central value model mode (optional, see Notes below)
```

## Performance Modes

- **`reduce-overhead`** (default): Best balance between compilation time and runtime performance. Recommended for most users.
- **`max-autotune`**: Maximum runtime performance with extensive kernel tuning. First epoch will be significantly slower (~2-5x) due to compilation overhead. Use when training for many epochs and maximum throughput is critical.
- **`default`**: Fastest compilation with moderate runtime improvement. Use for quick iterations during development.

## Compilation Overhead

The first epoch will be slower due to JIT compilation overhead:
- **`reduce-overhead`**: ~20-50% slower first epoch
- **`max-autotune`**: ~100-400% slower first epoch (tries many kernel variants)
- **`default`**: ~10-20% slower first epoch

CUDA Graph warnings during startup are normal and can be ignored.

## When to Use Each Mode

**Use `reduce-overhead` (default) when:**
- Training for moderate duration (100-10,000 epochs)
- Iterating during development/tuning
- You want good performance without excessive startup time

**Use `max-autotune` when:**
- Training for very long runs (10,000+ epochs)
- Running large-scale experiments where throughput is critical
- You can afford 2-5x slower startup for maximum steady-state performance

**Use `default` when:**
- Quick experiments and debugging
- Rapid prototyping with frequent code changes
- Compilation time is more important than runtime performance

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
| `default` | 0.9x | 1.10-1.15x | Development |
| `reduce-overhead` | 0.5-0.8x | 1.15-1.30x | Production (recommended) |
| `max-autotune` | 0.2-0.5x | 1.25-1.40x | Large-scale training |

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

### Performance Regression

If compiled code is slower:
- Ensure batch size is large enough (>256 recommended)
- Try `mode: "default"` for smaller models
- Profile to verify bottleneck is in RL update, not environment
