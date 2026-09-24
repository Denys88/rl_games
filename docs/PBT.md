# Population Based Training (PBT)

rl_games ships the PBT observer lineage of [DexPBT](https://arxiv.org/abs/2305.12127)
(previously maintained downstream in
[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and Isaac Lab's
`isaaclab_rl`), now available to any rl_games backend from `rl_games.common.pbt`.

## How it works

Each member of the population is an independent training process with a unique
`policy_idx`. Every `interval_steps` environment frames, each process:

1. saves a scored checkpoint into a shared workspace directory,
2. loads the latest checkpoint of every member and ranks the population,
3. if the replacement rule selects it, re-execs itself from a better member's checkpoint
   (or its own) with the whitelisted hyperparameters mutated.

Replacement rules (`replace_rule`):

- `dexpbt` (default for new configs): members in the worst `replace_fraction_worst`
  (rounded up) restart from a random member of the best `replace_fraction_best`, if the
  candidate leads by more
  than `replace_threshold_frac_std` population stds (worst 20% trimmed) and by more than
  `replace_threshold_frac_absolute` of its objective; otherwise they restart from themselves
  with mutated parameters. A member whose best objective in the iteration reaches the top
  group is kept, and nothing is replaced until more than half the population has reported.
  The new parameters mutate the source's values with probability `leader_params_prob`,
  otherwise the member's own.
- `threshold`: underperformers (score < min(mean − `threshold_std`·std, mean − `threshold_abs`))
  restart from a random leader (the mirror image above the mean). Configs that set
  `threshold_std`/`threshold_abs` without `replace_rule` keep this rule (with a notice).

Grace windows: `initial_delay` frames from the start of training and `start_after` frames
after each restart pass before a member can be replaced. Right after a restart the score
still mostly measures the source policy, not the mutation.

## Objective

The objective is read from env infos at a dotted address (`objective`) — **required
when PBT is enabled**, since info layouts differ per backend: e.g.
`episode.Episode_Reward/success` for Isaac Lab-style nested infos, or `scores` for
flat info dicts. It is averaged over the most recent finished episodes
(`objective_window`, default: the rank's number of envs) and pooled across ranks. A member
takes part in a PBT iteration only once every rank's window is full; until then the
iteration is retried on every step:

- a per-env tensor is read at the envs that finished on that step;
- a scalar is taken as the mean over the envs that finished on that step (Isaac Lab
  episode logs) and weighted by their count.

Prefer a true task metric over raw reward when reward shaping is non-stationary. With a
curriculum, rank curriculum progress first (e.g. progress + 0.01 × success, as DexPBT's
tolerance curriculum did): members at different curriculum levels face different tasks.

## Usage

Attach the observer when constructing the runner. Train scripts that rewrite `sys.argv`
(Hydra, argparse `parse_known_args`) should pass the original argv:

```python
from rl_games.common.pbt import PbtAlgoObserver, MultiObserver

launch_argv = list(sys.argv)          # before anything rewrites sys.argv
...
pbt = PbtAlgoObserver(params, args_cli, launch_argv=launch_argv)
runner = Runner(algo_observer=MultiObserver([my_other_observer, pbt]))
```

with a `pbt` section in the params:

```yaml
pbt:
  enabled: True
  policy_idx: 0          # unique per process, 0..num_policies-1
  num_policies: 8
  directory: ./pbt_run
  interval_steps: 20000000
  initial_delay: 200000000
  start_after: 100000000
  replace_fraction_worst: 0.3
  objective: episode.success
  mutation_rate: 0.15
  change_range: [1.1, 1.5]
  mutation:
    agent.params.config.kl_threshold: mutate_float
    agent.params.config.entropy_coef: mutate_float
    agent.params.config.mini_epochs: mutate_mini_epochs
    agent.params.config.e_clip: mutate_eps_clip
    agent.params.config.gamma: mutate_discount
```

Every mutation key must resolve to a value. Parameters outside the agent params (e.g. env
reward weights) are passed as `extra_params={"env.rewards.success.weight": 10.0}`, keyed by
the override name the train script accepts.

## Restart mechanism and launchers

On replacement the source checkpoint is copied into the member's workspace
(`restart.pth`, with the member's own frame count and a `pbt_history` entry) and the
process re-execs itself:

- with `launch_argv`, the original command runs verbatim with only the checkpoint argument
  (`checkpoint_arg`, default `--checkpoint`) and the mutated `key=value` overrides replaced;
- without it, the command is rebuilt from `sys.argv` plus the task/seed/num_envs/headless,
  wandb (`--track`), rendering and distributed flags of `args_cli`.

With `reseed_on_restart` (default), the seed argument (`seed_arg`, default `--seed`) is
replaced by a seed derived from (seed, `policy_idx`, restart count); a seed of -1 stays -1.
Set a fixed experiment name per member (e.g. `agent.params.config.full_experiment_name=pbt_p00`)
so restarts keep writing into the same run directory.

The restarted process receives the transfer metadata through `rl_games.common.pbt.restart_info()`:
`policy_idx`, `source_policy`, `restart_count`, `iteration`, `frame`, `checkpoint` and
`mutated` (the parameters that differ from the source's). rl_games uses it to keep mutated
`learning_rate` / `entropy_coef` over the values restored from the checkpoint; environments can
read it in `set_env_state` to tell a population transfer from an ordinary resume (e.g. to
inherit or keep curriculum state, or to re-seed).

By default the restart uses `sys.executable` (plain Python). Isaac Sim workflows that
must go through a wrapper set it explicitly:

```yaml
pbt:
  launcher: /path/to/_isaac_sim/python.sh
```

`rl_games/runner.py` does not currently accept `key=value` overrides, so PBT with the
plain runner requires a thin Hydra-style entry point; native runner support is planned
alongside the config-management work.

Workspace files are written atomically; a member more than 20 iterations behind the newest
one no longer blocks checkpoint cleanup. The best member keeps a copy of its checkpoint in
`<workspace>/best/` (`best.yaml` names the current copy). After `max_consecutive_errors`
failed checkpoint save/load iterations in a row the member raises instead of silently
training without PBT.

## Mutation functions

- `mutate_float` — multiply or divide by a random factor in `change_range`.
- `mutate_float_min_1` — `mutate_float`, floored at 1.0.
- `mutate_eps_clip` — `mutate_float`, clamped to [0.01, 0.3].
- `mutate_mini_epochs` — ±1, clamped to [1, 8].
- `mutate_discount` — mutate `(1 - x)` conservatively (for gamma-like params near 1.0).

Use `mutate_mini_epochs` for integer parameters. With `lr_schedule: adaptive`,
`learning_rate` is only the starting point of the KL-driven schedule; mutate
`kl_threshold` instead.

Custom functions can be registered by adding to
`rl_games.common.pbt.mutation.MUTATION_FUNCS`.
