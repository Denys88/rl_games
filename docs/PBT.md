# Population Based Training (PBT)

rl_games ships the PBT observer lineage of [DexPBT](https://arxiv.org/abs/2305.12127)
(previously maintained downstream in
[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and Isaac Lab's
`isaaclab_rl`), now available to any rl_games backend from `rl_games.common.pbt`.

## How it works

Each member of the population is an independent training process with a unique
`policy_idx`. Every `interval_steps` environment frames, each process:

1. saves a scored checkpoint into a shared workspace directory,
2. compares its objective against the population
   (leaders: score > max(mean + `threshold_std`·std, mean + `threshold_abs`);
   underperformers: the mirror image below the mean),
3. if it is an underperformer, re-execs itself from a random leader's checkpoint
   with the whitelisted hyperparameters mutated multiplicatively.

The objective is read from env infos at a dotted address (`objective`), e.g. a task
success rate — prefer a true task metric over raw reward when reward shaping is
non-stationary.

## Usage

Attach the observer when constructing the runner:

```python
from rl_games.common.pbt import PbtAlgoObserver, MultiObserver

observer = MultiObserver([my_other_observer, PbtAlgoObserver(params, args_cli)])
runner = Runner(algo_observer=observer)
```

with a `pbt` section in the params:

```yaml
pbt:
  enabled: True
  policy_idx: 0          # unique per process, 0..num_policies-1
  num_policies: 8
  directory: ./pbt_run
  interval_steps: 100000
  threshold_std: 0.10
  threshold_abs: 0.05
  mutation_rate: 0.25
  change_range: [1.1, 2.0]
  objective: episode.success
  mutation:
    agent.params.config.learning_rate: mutate_float
    agent.params.config.grad_norm: mutate_float
    agent.params.config.gamma: mutate_discount
```

## Restart mechanism and launchers

On replacement the process re-execs itself, rebuilding its original command line with
`--checkpoint=<leader checkpoint>` plus `key=value` overrides for the mutated
hyperparameters. This requires a train script whose CLI accepts Hydra-style
`key=value` overrides (as in Isaac Lab / IsaacGymEnvs `train.py`).

By default the restart uses `sys.executable` (plain Python). Isaac Sim workflows that
must go through a wrapper set it explicitly:

```yaml
pbt:
  launcher: /path/to/_isaac_sim/python.sh
```

`rl_games/runner.py` does not currently accept `key=value` overrides, so PBT with the
plain runner requires a thin Hydra-style entry point; native runner support is planned
alongside the config-management work.

## Mutation functions

- `mutate_float` — multiply or divide by a random factor in `change_range`.
- `mutate_discount` — mutate `(1 - x)` conservatively (for gamma-like params near 1.0).

Custom functions can be registered by adding to
`rl_games.common.pbt.mutation.MUTATION_FUNCS`.
