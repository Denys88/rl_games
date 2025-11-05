# torch.compile Support in RL-Games

## Overview

RL-Games now supports PyTorch's `torch.compile` for enhanced performance, providing 10-15% improvement in training throughput.

## Configuration

Add `torch_compile` to your yaml config under `params.config`:

```yaml
# Enable with default mode (max-autotune) - recommended
torch_compile: True

# Disable compilation
torch_compile: False

# Specify mode as string
torch_compile: "default"        # Good compatibility, moderate speedup
torch_compile: "reduce-overhead" # Balanced performance and compilation time  
torch_compile: "max-autotune"   # Best runtime performance (default)

# Advanced configuration (dict format)
torch_compile:
  mode: "max-autotune"          # Actor/main model mode
  critic_mode: "default"        # Central value model mode (see Notes below)
```

## Performance Modes

- **`max-autotune`** (default): Best runtime performance, longer initial compilation
- **`reduce-overhead`**: Good balance between compilation time and runtime performance
- **`default`**: Fastest compilation, moderate runtime improvement

## Notes

- First epoch will be slower due to compilation overhead
- CUDA Graph warnings during startup are normal
- `critic_mode` only applies when using `central_value_config`, which is used for:
  - Single-agent asymmetric actor-critic (actor and critic see different observations)
  - Multi-agent centralized critic (critic sees all agents' observations + privileged info)
- For standard PPO, the main model is compiled (includes both actor and critic)
- Note: `separate: true` in network config creates separate network towers, but they're compiled together as one model

## Performance

torch.compile typically provides 15-40% performance improvement in total training throughput. The exact speedup vary significantly by environment type and configuration and depends on your model architecture, batch size, and hardware configuration.
