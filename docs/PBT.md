# Population Based Training (PBT)

rl_games ships the PBT observer of [DexPBT](https://arxiv.org/abs/2305.12127) (the lineage
maintained in [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and Isaac Lab's
`isaaclab_rl`) in `rl_games.common.pbt`. Each member of a population is its own training process,
or torchrun job, with a `PbtAlgoObserver`. Members share a workspace directory and rank themselves
against each other at fixed frame intervals. A member that falls behind re-executes itself from a
better member's checkpoint, or its own, with mutated hyperparameters.

Supported agents: `a2c_continuous` and `a2c_discrete` (PPO). SAC is not supported: the observer
raises `NotImplementedError` when SAC's `train()` starts.

## Quick start

[`examples/pbt/`](../examples/pbt/) is a runnable CartPole population and the template for a train
script ([its README](../examples/pbt/README.md) has the details):

```bash
bash examples/pbt/run_population.sh --members 4 [--device cuda] [--train-dir DIR] [-- member args]
```

It runs on CPU in under a minute. Member i logs to `<train-dir>/logs/member<i>.log`, and
`tensorboard --logdir <train-dir>` shows each member's `pbt/*` scalars next to `rewards/step`.
Each population needs its own `--train-dir`, by default a new timestamped one
([Reusing a workspace](#reusing-a-workspace)). The script passes `--seed` to every member:
members train with `member_seed(seed, idx)`, and each restart gets a seed derived from it. The
example ranks members by their episode return (`episode.r`, from gymnasium's
`RecordEpisodeStatistics`) and turns `torch_compile` off unless
`agent.params.config.torch_compile=true` is given. A member of several GPUs is one torchrun job:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    examples/pbt/train_member.py --policy-idx 0 --num-policies 4 --seed 42 --distributed
```

## Writing a train script

A restart re-executes the member's launch command with `--checkpoint=<member dir>/restart.pth` and
every mutable parameter as a `<key>=<value>` override, e.g. `agent.params.config.gamma=0.99`. A
train script must:

1. accept those overrides and the checkpoint argument (`checkpoint_arg`); `rl_games/runner.py`
   doesn't. Isaac Lab's Hydra train script takes them as they are;
   `examples/pbt/train_member.py` uses argparse's `parse_intermixed_args`. Values are written with
   Python's `str()`, and PyYAML reads `1e-05` as a string: parse a value as int, then float, then
   YAML;
2. pass its launch command as `launch_argv=list(sys.argv)`, captured before anything rewrites
   `sys.argv` (`python -m` launches: [The restart command](#the-restart-command));
3. call `validate()` before building the environment, so a config error costs no scene build;
4. seed each member with `member_seed(seed, policy_idx)` and pass the seed on the command line
   ([Seeds](#seeds));
5. give each member a fixed `full_experiment_name` (e.g. `pbt_p00`) on every launch. Without one
   (rank 0 warns), each restart writes into a new timestamped run directory, and members started
   in the same second share one;
6. launch all members with the same `train_dir` and `pbt.workspace`: the workspace is
   `<train_dir>/<workspace>`;
7. for a multi-GPU member, pass an `args_cli` with `distributed=True`
   ([Multi-GPU members](#multi-gpu-members));
8. register custom mutation functions at import ([Mutation functions](#mutation-functions)).

```python
import sys

LAUNCH_ARGV = list(sys.argv)          # before anything can rewrite sys.argv

from rl_games.common.pbt import MultiObserver, PbtAlgoObserver, member_seed, validate
from rl_games.torch_runner import Runner

agent_cfg = ...                       # {"params": rl_games params, "pbt": {...}}, overrides applied
validate(agent_cfg)                   # before the env is built
params, idx = agent_cfg["params"], agent_cfg["pbt"]["policy_idx"]
seed = args_cli.seed if args_cli.seed is not None else params["seed"]   # the seed the run uses
params["seed"] = member_seed(seed, idx)
params["config"]["full_experiment_name"] = f"pbt_p{idx:02d}"
pbt = PbtAlgoObserver(agent_cfg, args_cli, launch_argv=LAUNCH_ARGV)
runner = Runner(algo_observer=MultiObserver([my_other_observer, pbt]))
runner.load({"params": params})
runner.run({"train": True, "play": False, "checkpoint": args_cli.checkpoint, "sigma": None})
```

The mutable parameters are keys of the flattened `{"agent": agent_cfg}`, e.g.
`agent.params.config.gamma`. Values outside the agent config (e.g. env reward weights) are passed
as `extra_params={"env.rewards.success.weight": 10.0}`, keyed by the override name the train script
accepts. A `pbt` section ([Config reference](#config-reference) has every key):

```yaml
pbt:
  enabled: true                # read by the train script: attach the observer
  policy_idx: 0                # unique per member, 0..num_policies-1
  num_policies: 8
  workspace: pbt_workspace     # <train_dir>/pbt_workspace, shared by the members
  objective: episode.success   # dotted address into the env infos, required
  interval_steps: 20000000     # frames between PBT iterations
  initial_delay: 200000000
  start_after: 100000000
  replace_rule: dexpbt
  replace_fraction_worst: 0.3
  mutation_rate: 0.15
  change_range: [1.1, 1.5]
  mutation:
    agent.params.config.kl_threshold: mutate_float
    agent.params.config.entropy_coef: mutate_float
    agent.params.config.mini_epochs: mutate_mini_epochs
    agent.params.config.e_clip: mutate_eps_clip
    agent.params.config.gamma: mutate_discount
```

Under `lr_schedule: adaptive`, set `min_lr` and `max_lr` explicitly
([CONFIG_PARAMS.md](CONFIG_PARAMS.md)): `max_lr` also caps `learning_rate` mutations.

## How a population runs

Frames are the learner's `frame`: agent steps summed over the member's ranks
(`horizon_length * num_actors * num_agents` per epoch and rank). `interval_steps`, `start_after`,
`initial_delay` and the workspace's `frame` count them. The observer runs once per epoch, after
the rollout and before the update. A PBT iteration is due when `frame // interval_steps` passes the
last iteration; a process's first one comes at the first boundary after its start frame. It runs
once every rank has a full objective window and a new finite objective ([Objective](#objective)),
and is retried at every epoch until then, so a member's iterations may skip boundaries.

At an iteration, rank 0:

1. saves a scored checkpoint (`<iteration>.pth`, `<iteration>.yaml`) into its member directory;
2. loads every member's newest checkpoint at or below this iteration, deletes its own old files
   ([Workspace](#workspace)) and prints the population table;
3. records a new population best in `best/` when it holds it;
4. applies the replacement rule. A member that restarts writes `restart.pth` (the source's
   checkpoint with the member's counters) and `restart.yaml`, and re-executes at the next epoch's
   PBT step; what it trains in between is discarded.

A member is not replaced in its grace windows or near the end of its run. It skips its own
decision, and its workspace yaml says `replaceable: false`, which keeps it out of its peers' worst
group:

- for `start_after` frames after each process start (a fresh launch, a resume or a restart) and
  after a self-mutation applied in process;
- until its frame reaches `initial_delay`;
- when fewer than `max(start_after, interval_steps)` frames are left after this epoch before the
  run ends: `(max_epochs - epoch) * curr_frames` (times `world_size` under `multi_gpu`), or
  `max_frames - frame -` this epoch's frames, whichever is nearer; no limit when both are -1. A
  decision in the last epoch could never run, and a restart near the end could not leave its grace
  window. The log says `... frames left after this epoch before max_epochs ends the run ... no
  replacement this close to the end`.

## Replacement rules

### dexpbt

The default. Members are ranked by objective; members that have not reported rank last. Equal
objectives are ordered by a shuffle seeded with (`pbt.workspace`, iteration): every member uses the
same order at an iteration, and it changes at the next. Each member decides about itself:

1. It is a candidate when it is among the last `ceil(replace_fraction_worst * num_policies)` members
   of the ranking that can be replaced now (status `OK` below).
2. Nothing is replaced until more than `max(2, num_policies // 2)` members have reported. With 2
   members dexpbt never replaces (rank 0 warns at init).
3. It is kept when its best pooled objective during this iteration reaches the lowest objective of
   the top group, the best `ceil(replace_fraction_best * num_policies)` reported members.
4. It draws a donor from the top group. If the donor leads by more than
   `replace_threshold_frac_std` population stds (the worst 20% left out of the std) and by more than
   `replace_threshold_frac_absolute * |donor objective|`, it restarts from the donor's checkpoint
   (a transfer); otherwise from its own, with mutated values (a self-mutation).
5. The new values mutate the donor's with probability `leader_params_prob`, else the member's own.

The absolute margin scales with the donor's objective, so an offset changes decisions: with an
objective of 11 + success and success gaps of a few hundredths, the default 0.05 asks for a lead
of about 0.55, and every replacement becomes a self-mutation. Rank by a quantity whose size matches
the gaps that matter, or lower `replace_threshold_frac_absolute` (0 leaves the std margin alone).

The checkpoint summary's `Status` column shows how the deciding member sees each member:

- `—`: has not reported;
- `not replaceable`: its newest yaml says `replaceable: false` (it decided a restart and the
  restarted process has not saved yet, or it is in a grace window or near its end);
- `behind`: its newest yaml is 2 or more iterations behind the deciding member's and not below the
  decider's own objective at that iteration (a member behind that is below it stays `OK`);
- `OK`: can be replaced.

Members that are `not replaceable` or `behind` still rank, count for the reporting gate and can be
donors. A decision line adds `(the worst replaceable member; [..] rank lower)` when such members
rank below the deciding one.

### threshold

Members below `min(mean - threshold_std * std, mean - threshold_abs)` of the reported objectives
restart from a random leader, a member above
`max(mean + threshold_std * std, mean + threshold_abs)`, always mutating the leader's values;
without leaders they self-mutate. There is no reporting gate or worst group. Configs that set
`threshold_std` or `threshold_abs` without `replace_rule` use this rule.

### Both rules

- A self-mutation that changes no value prints `no-op self-mutation: keep training` and changes
  nothing: no `restart.pth`, no new grace window. Its chance is about `(1 - mutation_rate)^k` for k
  keys (13% for 7 keys at 0.25), more when values sit at a clamp. A transfer always restarts, also
  with unchanged values.
- The tie order also picks the population-best owner (`best/`) and `pbt/rank`, so tied members get
  distinct ranks.
- PBT's draws (the donor, the `leader_params_prob` coin, the mutation coins and factors) come from a
  generator private to the member ([Seeds](#seeds)).

### Differences from IsaacGymEnvs' DexPBT

With every member reported and replaceable and no ties, the rule picks the same members and donors
as IsaacGymEnvs' for the same objectives. Otherwise:

- IsaacGymEnvs takes the worst group from the whole ranking: a member that never reported (ranked
  at -1e9), or one restarting, lagging or in its grace window, holds a worst slot and blocks every
  replacement. Its top group can include members that never reported.
- It breaks ties by index, restarts on no-op self-mutations, decides in a run's last epochs, and
  mutates every member's values at launch (`initial_pbt_check`). rl_games members start with their
  configured values.
- Its objective is a different estimator ([Objective](#objective)).

## Objective

`objective` is a dotted address into the env infos, required: e.g. `episode.Episode_Reward/success`
(Isaac Lab-style nested infos), `scores` (flat dicts) or `episode.r` (gymnasium's
`RecordEpisodeStatistics`). It is read at each env step that finishes episodes:

- a per-env tensor or array is read at the finished envs. Done indices arrive per agent
  (`env * num_agents`), as rl_games passes them to observers, and are read at
  `index // num_agents`. A NaN or inf entry drops only its own episode;
- a scalar is the mean over the envs that finished on that step (Isaac Lab episode logs), weighted
  by their count. A NaN or inf scalar drops all of them.

Dropped episodes count as non-finite; episodes that end on a step where the address can't be read
count as unreadable. `pbt_utils.episode_stats(value, done_indices, num_agents)` returns
`(sum, count, nonfinite)` for one step (`episode_sum_count` the first two).

A member's objective is the mean of the latest `objective_window` finished episodes of each rank (0:
the rank's env count), pooled across ranks. That size is DexPBT's, but the estimator differs:
DexPBT averages each env's latest episode, weighting envs equally, and the window weights episodes.
Envs with short (failed) episodes count more, and after a (re)start, when every env's first
episodes end together at the time limit, the mean can swing for several episode lengths: cover
that with `start_after`.

An iteration runs only when every rank's window is full and every rank has counted a finite episode
since the previous iteration. Until then it is retried at every epoch without advancing the
iteration counter: the member does not save, compare or get replaced, and an old mean is never
published again. Rank 0 warns once per boundary, with the last lookup error of the interval:

- `waits for objective '<key>': N finished episodes counted, W per rank needed (X non-finite and
  Y unreadable dropped)` while the windows fill;
- `waits for a new objective '<key>' [on k of n ranks]: V finite, X non-finite and Y unreadable
  episodes since iteration I` after that.

Every rank raises `RuntimeError` after `max_consecutive_errors` boundaries in a row at which some
rank had failed: since the last iteration (or the process start) each of its envs ended an
episode, at least a window of episodes ended, and none had a finite objective. That is a diverged
policy, a wrong address, or infos that no longer carry the objective. Waiting for episodes to end
never counts: after a (re)start every env resets, so with few early terminations episodes end in
bursts at the time limit. With several ranks the decision uses the values pooled in the one
all-reduce per epoch, so all ranks wait, advance and raise at the same epoch.

A state whose networks hold NaN or inf is never saved into the workspace or copied for a restart.
Such a save fails (it counts toward `max_consecutive_errors`), and the member's last finite
checkpoint stays its newest.

Prefer a true task metric over raw reward when reward shaping is non-stationary. With a
curriculum, rank curriculum progress first (e.g. progress + 0.01 * success, as DexPBT's tolerance
curriculum did): members at different levels face different tasks.

## Restarts

### The restart command

The restart runs the launch command (`launch_argv`), or without it a command rebuilt from
`sys.argv` (below), with:

- the script path made absolute when it exists relative to the launch directory;
- every `checkpoint_arg` replaced by `<checkpoint_arg>=<member dir>/restart.pth`;
- every mutable parameter's `key=value` removed, and all of them appended with their new values,
  changed or not;
- with `reseed_on_restart`, the value of `seed_arg` replaced by a seed derived from it, the member
  and the restart count, when the command has it. The next restart derives from that seed; -1
  stays -1;
- inside a Hydra job that runs in its run dir, `hydra.run.dir` pinned (below).

Without `launch_argv` the command is `sys.argv` plus the task, seed, num_envs, headless, wandb
(`--track` with its entity, project and name), rendering and `--distributed` flags of `args_cli`.
`--seed` is always passed, -1 when `args_cli.seed` is None. A script that rewrote `sys.argv`
(Isaac Lab's train.py keeps only the Hydra overrides) loses every other launch flag, e.g.
`--max_iterations`, `--device`, `--agent`. Rank 0 prints a NOTICE at init when `sys.argv` is no
longer the tail of `sys.orig_argv` (the script rewrote it, or it runs as `python -m`).

**Launch directory.** The restart runs from the launch directory: Hydra's original cwd inside a
Hydra job, else the cwd when the observer was built. With Hydra 1.1 job semantics (`version_base`
omitted or `'1.1'`, or `hydra.job.chdir=True`) the job runs inside its run dir. The restart then
drops any `hydra.run.dir` override and appends `hydra.run.dir=<that run dir>`, single-quoted when
the path needs it: the child reuses the run dir instead of nesting a new one, and a relative
`train_dir` keeps its workspace. A run dir path containing `${` can't be passed. With
`version_base` 1.2 or later (Isaac Lab uses `'1.3'`) none of this is needed; prefer it, or
`hydra.job.chdir=False`.

**`python -m` entry points.** Pass `launch_argv=['-m', 'pkg.module', *sys.argv[1:]]`: the restart
then runs `python -m pkg.module`, or `torch.distributed.run ... -m pkg.module` for a multi-GPU
member. With `launch_argv=list(sys.argv)` the module's file would run as a plain script, and
relative imports fail. Interpreter options (`-u`, `-O`, `-X`, `-W`) are not kept; use environment
variables such as `PYTHONUNBUFFERED`.

**Launcher.** An empty `pbt.launcher` runs `sys.executable`. Otherwise it is resolved at init the
way `os.execv` will see it, plus PATH: an existing file relative to the launch directory becomes
its absolute path and must be executable (a cwd-relative `python.sh` works); else a command on PATH
(`launcher: python3`); else `ValueError`. Isaac Sim workflows use their wrapper:

```yaml
pbt:
  launcher: /path/to/_isaac_sim/python.sh
```

**Environment.** The child inherits the environment, plus `RL_GAMES_PBT_RESTART` (the payload of
`restart_info()`) and the wandb variables below. PATH, PYTHONPATH, LD_LIBRARY_PATH and
OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS are deduplicated, keeping first occurrences and the first empty
entry (the current directory).

**Logs.** Before the re-exec the member prints `Policy N: Restarting self with args [...] in
<launch dir>` and the environment's size: variable count, bytes, the five largest variables' names
and lengths, and SC_ARG_MAX, never values. stdout and stderr are flushed before `execv` and before
the other ranks exit, so block-buffered logs keep their last lines.

### What a restart keeps

| | After a restart |
|---|---|
| weights, optimizer, normalizers, central value net, fp16 loss scale | the source checkpoint's: the donor's, or the member's own for a self-mutation |
| env state (`set_env_state`) | the source checkpoint's, saved by its rank 0; every rank gets it |
| frame and epoch | the member's frame and last completed epoch at the decision; the first epoch is that epoch + 1 |
| `learning_rate`, `entropy_coef` | the restart command's values, by the restore rule below (adaptive: the source's adapted LR moved by this restart's mutation) |
| other config values and `weight_decay` | the restart command's |
| sigma of a fixed-sigma policy | the checkpoint's: the command's `--sigma` is not applied (fresh launches and plain resumes apply it) |
| best-reward watermark | with a fixed `full_experiment_name` the member's own, so `nn/<name>.pth` is only replaced by a policy better than the member's best; else the source's, in a new run directory |
| `pbt_history` | the source's plus one entry `{policy_idx, source_policy, frame, objective, source_objective}` |
| seed | derived per restart when the command has `seed_arg` ([Seeds](#seeds)) |
| run directory | the same with a fixed `full_experiment_name`, else a new timestamped one |
| wandb run | the same after a clean finish, else a new one ([wandb](#wandb)) |
| RNN states | reset, as on any resume |

Workspace `.pth` files, `best/` copies and `restart.pth` store the last completed epoch
(`epoch_num - 1`): the PBT save point is after the epoch counter's increment and before that
epoch's update. So `max_epochs` counts every trained epoch across restarts. The learner's own
checkpoints keep `pbt_history` across its saves and carry the key only when it is non-empty.

### Hyperparameters on restore

This rule holds for every A2C restore: PBT restarts, restarts by other PBT observers that pass
mutated values only on the command line (Isaac Lab's `isaaclab_rl`, IsaacGymEnvs ports), and plain
resumes. The config, on a restart the restart command, is the source of truth for hyperparameters.
Checkpoints store their run's configured `learning_rate` and `entropy_coef` as `config_hparams`.
The checkpoint contributes only runtime state, and only while the config value equals the stored
one:

| `lr_schedule` | learning rate | `entropy_coef` |
|---|---|---|
| `adaptive` | the checkpoint's adapted LR; if `learning_rate` changed, times config/stored, clamped to `[min_lr, max_lr]` | the config's |
| `linear` | the checkpoint's scheduled LR; if `learning_rate` changed, the config's until the schedule's next step | with `schedule_entropy` the checkpoint's scheduled value, the config's if `entropy_coef` changed; else the config's |
| identity / None | the config's | the config's |

- Under `adaptive`, `max_lr` caps `learning_rate` mutations. The clamp applies only when
  `learning_rate` changed, so an unchanged resume continues exactly.
- `weight_decay` always comes from the config, in the actor's and the central value net's
  optimizers; the critic's LR is its config value.
- A checkpoint without `config_hparams` (older versions) keeps its saved LR under `adaptive` and
  `linear`, and its saved entropy under a linear entropy schedule.
- The rule reads the configured `lr_schedule`, so every rank of a multi-GPU member restores the same
  values (only rank 0 runs the scheduler).

`restart.pth` stores the restart's pre-mutation values of the mutable `learning_rate` and
`entropy_coef` (the base the new values were mutated from: the member's own or the donor's) as
`config_hparams`. Under `adaptive` the restarted member
therefore continues from the donor's adapted LR times this restart's mutation factor (1 when
`learning_rate` was not mutated), clamped. It is not reset to a nominal value, nor scaled by the
ratio of two members' configured LRs. Mutating `learning_rate` under `adaptive` only perturbs the
inherited walk; `kl_threshold` changes the schedule.

### restart_info() and restart.yaml

`rl_games.common.pbt.restart_info()` returns None for a fresh launch or a plain resume (also a
relaunch from `restart.pth` by hand), else a dict:

| Key | |
|---|---|
| `schema` | 1 |
| `kind` | `"transfer"` (restarted from another member's checkpoint) or `"self_mutation"` (from its own, with mutated values) |
| `policy_idx`, `source_policy` | the member, and the member whose checkpoint it restarted from |
| `restart_count` | this process's: 1 at the first restart after the member's last launch (fresh or a resume) |
| `iteration`, `frame` | of the decision |
| `checkpoint` | the member's `restart.pth`, absolute |
| `mutated` | the params that differ from the source's |
| `experiment_name` | the run name the previous process wrote into |

Integrations branch on `kind`, not on `is not None`: a self-mutation keeps the member's weights and
env state, and later versions may add kinds. Environments can read it in `set_env_state`, e.g. to
inherit or keep curriculum state. A payload from an older rl_games has no `schema` or
`experiment_name`; `kind` is then set from `source_policy == policy_idx`. The payload is parsed
once per process, and the variable is then removed from the environment. rl_games marks the
restart on every rank (`config['pbt_restart']`, so the runner keeps the checkpoint's sigma) and
continues its restart count. A restarted process notices on rank 0 when its experiment name differs
from the payload's: a train script that timestamps the name (Isaac Lab's train.py) splits the
member's summaries and `nn/` checkpoints across run directories.

`<workspace>/<idx:03d>/restart.yaml`, next to `restart.pth`, records the last restart decision.
Rank 0 writes it at the decision, after `restart.pth` (temp file and rename; only the member
writes it). It holds the payload's keys plus `epoch` (the last completed one, as in
`restart.pth`), `params` (every new value), `seed` (what the command passes as `seed_arg`: -1 when
random per launch, null when the command keeps its own seed) and `time` (`time.time()`).
`checkpoint` is the writer's absolute path; on another mount use `<member dir>/restart.pth`. No
record is written when nothing restarts: a no-op or in-process self-mutation, the end-of-run guard,
a failed `restart.pth`. Nothing in rl_games reads it yet, and workspace loaders and cleanup ignore
it. A process that is not a PBT restart deletes it at its first epoch when it records a decision
above that process's iteration.

### wandb

The restart acts on the open run (`wandb.run`), not on `--track`; without an open run there is
nothing to do. It sets `WANDB_RUN_ID` to the run's id, `WANDB_RESUME=allow` and
`WANDB_INIT_TIMEOUT=300` where unset, and waits up to `pbt.wandb_finish_timeout` seconds (default
120) for `run.finish()`:

- after a clean finish the restarted process resumes the same run through the inherited wandb
  service;
- if `finish()` times out or raises, the member restarts anyway. `WANDB_SERVICE` is removed from
  the child's environment (the child would hang in `wandb.init` or get "run ID ... is in use"),
  `WANDB_RESUME` is forced to `allow`, and the child logs to a new run
  `<run id>-r<restart count>`; an earlier `-r<n>` suffix is replaced. The old wandb-core may go on
  uploading the old run: one stale service per timed-out restart, until the member exits.

The train script still calls `wandb.init` itself in the child.

## Multi-GPU members

- A member is one torchrun job on one node: `LOCAL_WORLD_SIZE` must equal `WORLD_SIZE`
  (`ValueError` at init). Its `args_cli` needs `distributed=True`, also with `launch_argv`.
  Otherwise the restart flag is not broadcast: the observer warns at init when `WORLD_SIZE > 1`, and
  the member dies at its first restart.
- Rank 0 saves, loads and decides. The restart flag is broadcast and the objective pooled once per
  epoch; on a restart every rank restores `restart.pth`.
- A restart re-executes the member from rank 0 as `python -m torch.distributed.run --standalone
  --nnodes=1 --nproc_per_node=WORLD_SIZE <script args>`, and the other ranks exit. torchrun binds a
  free port itself. Flags of the original torchrun command (`--master_port`, `--rdzv-*`,
  `--log-dir`, `--redirects`, `--tee`, `--max-restarts`, ...) are not passed on, and
  `args_cli.master_port` is ignored. The new ranks inherit rank 0's environment and CPU affinity:
  give a member's ranks one `CUDA_VISIBLE_DEVICES` that lists all its GPUs, not one GPU per rank
  (`srun --gpus-per-task`, torchrun's `--virtual-local-rank`).
- Each restart nests a launcher level: the old rank-0 process becomes a torchrun agent that stays
  until the member exits, with about 0.3 GB of host memory.
- Keep the outer torchrun at `--max-restarts=0` (its default). With more, a crash in a later
  generation makes the outer agent rerun the original command: a duplicate member, started from its
  launch arguments.
- Stop a member by signalling its whole process tree: after a restart, the outer agent's SIGKILL
  escalation no longer reaches the inner workers.

## Seeds

- `rl_games.common.pbt.member_seed(seed, policy_idx, stride=1_000_000)` returns
  `seed + policy_idx * stride`; -1 and None come back unchanged. Members launched with one seed
  start as copies (weights, env and sampling streams), so the train script applies it before
  seeding; rl_games doesn't, and there is no config key. Apply it to the seed the run will use,
  `params['seed'] = member_seed(seed, policy_idx)`: a None seed leaves the Runner's
  `int(time.time())`, and members started in the same second start as copies.
- With `multi_gpu`, rl_games adds the global rank to `params.seed` itself. Scripts that add it as
  well (Isaac Lab) get learner seed + 2 * rank and env seed + rank, so within one member rank r's
  learner seed equals rank 2r's env seed. Across members the stride must exceed
  2 * (GPUs per member - 1).
- With `reseed_on_restart` and `seed_arg` on the command line, each restart passes a seed derived
  from the command's seed, the member and the restart count. Without it torch, numpy and the env
  replay the launch seed's streams after every restart; rank 0 prints a NOTICE at init when
  `launch_argv` has no `seed_arg` and the config seed is set (not None or -1). For a Hydra-style
  override, set `pbt.seed_arg` to its name (e.g. `seed`).
- PBT's draws come from `random.Random('pbt-draws:{seed}:{policy_idx}:{restart_count}:{start
  frame}')`, with `seed` = `params.seed` (OS entropy for None or -1), swapped into the global
  `random` module around each iteration. The donor choice, the `leader_params_prob` coin, the
  mutation coins and factors, custom mutation functions and subclasses that replace
  `_pbt_iteration` draw from it, and the env and learner streams are untouched. Draws differ across
  members, restarts and resumes, and repeat for the same (seed, member, restart count, start frame).

## Workspace

```
<train_dir>/<workspace>/
  000/                        a directory per member, written only by that member
    000012.yaml, 000012.pth     a scored checkpoint per PBT iteration
    restart.pth, restart.yaml   the last restart decision
  best/
    p003_000010.yaml            a member's record of a population best
    best_p003_000010.pth        its checkpoint copy
    best.yaml                   a view of the best record
```

**Cleanup.** Each member deletes its own files at or below the slowest member's iteration minus
20; a member that never reported counts as iteration 0. It always keeps its newest yaml at or
below the slowest member's iteration, so a live member any number of iterations behind still finds
every peer at or below its own iteration. Above 50 iterations it thins its older files as
IsaacGymEnvs does, never the newest 10 or the oldest: disk stays at about 50 yaml/pth pairs per
member, also when a member stops or dies. Each `.yaml` is deleted before its `.pth`.

**Files.** Workspace files are plain YAML (`yaml.safe_dump`, `yaml.safe_load`): a value plain YAML
can't hold (a numpy or torch scalar) makes the save raise, a failed iteration. A reader skips a
yaml, and takes the member's next older one, when it is not plain YAML, lacks a key, or has a key
of the wrong type (`iteration` an int; `true_objective` and `frame` finite numbers; `params` a dict
of finite numbers; `checkpoint` a string). A file deleted between the listing and the read also
falls back to the next older one, without a delay.

**Shared filesystems.** A transfer uses the `.pth` next to the listed yaml, so hosts need not mount
the workspace at the same path, and a workspace can be moved or copied; the stored absolute path
is only a fallback.

- One writer per file: a member's directory, its best records and copies, and its `restart.pth`
  are written only by that member. The exception is `best.yaml`, a view that whichever member
  records a new best replaces atomically.
- Writes go to a temp name, then an atomic rename. Mounts without atomic rename (object-store FUSE
  such as s3fs or gcsfuse) are not supported.
- Readers list a directory first; a listed file that is gone means "try the next one", never
  "no best".
- Temp files older than 1 h are swept, which assumes the hosts' clocks agree within 1 h.
- Peer checkpoints are unpickled: the workspace must be writable only by the population's user.

**Population best.** The best member records itself in `best/` when its objective beats the current
best: a checkpoint copy `best_p<idx>_<it>.pth`, then the record `p<idx>_<it>.yaml`, then the view
`best.yaml`. Each member keeps its two best records and copies.
`rl_games.common.pbt.population_best(workspace_dir)` returns the authoritative best record, or
None: a dict with `policy_idx`, `iteration`, `frame`, `true_objective`, `params`, `checkpoint` (the
copy, on this host's mount) and `source_checkpoint`. `best.yaml` has the same keys, but its
`checkpoint` names the writer's mount path, and on a shared filesystem it can lag briefly. A
`best.yaml` without records (older versions) counts as a record.

### Reusing a workspace

Workspace files carry no run identity: a new run in the same workspace would rank, and copy, the
earlier run's files. So a fresh launch (no checkpoint, not a PBT restart, frame 0) into a member
directory that holds numbered yamls raises `RuntimeError` at its first epoch, naming the
directory. Use a new `pbt.workspace` or `train_dir`, delete the old workspace, or set
`reuse_workspace: true`: each member then deletes its own numbered files when it starts, and
`best/` keeps the earlier records, so a new best is recorded only once it beats them. Isaac Lab's
`directory: .` puts every run of a config into `logs/rl_games/<config>/pbt_workspace`, so reruns
hit this guard.

Plain resumes (frame > 0) and PBT restarts pass; a member resumed from an older checkpoint deletes
its own files above its iteration. Bring a dead member of a running population back by resuming it
from its latest workspace `.pth` or its `restart.pth`, not from scratch.

## TensorBoard and logs

Rank 0 writes the `pbt/*` scalars at every iteration:

| Scalar | |
|---|---|
| `pbt/objective` | the pooled window mean |
| `pbt/00_best_objective` | the best reported objective in the population |
| `pbt/rank` | this member's rank, 1 the best (ties in the iteration's order) |
| `pbt/restart_count` | the process's restart count |
| `pbt/<flattened param>` | the current value of each numeric mutable param |

They sit at the x of the epoch's `rewards/step` point, `(frame + curr_frames * world_size) //
num_agents` (`world_size` only under `multi_gpu`): env steps after the epoch. At a restart decision
or an in-process self-mutation the new values follow at the same x, so the plot steps to them. With
the legacy `params.config.population_based_training` switch, `IntervalSummaryWriter` rate-limits
each tag by time and can drop those new values.

Log lines of rank 0:

| Line | Meaning |
|---|---|
| the checkpoint summary table | each member's newest checkpoint at or below this iteration, with its `Status` |
| `WARNING members [..] have not reported` | no checkpoint at or below this iteration: `num_policies` larger than the population, or a member that crashed before its first save |
| `PBT iteration N waits for ...` | the objective gate ([Objective](#objective)) |
| `not among the worst [..]`, `only k of n members reported`, `best objective this iteration ... reaches the top [..]`, `not an underperformer` | no replacement |
| `replace from j: lead ...`, `underperformer` | a transfer |
| `lead of j (...) too small: mutate own parameters`, `no leaders: mutate own parameters` | a self-mutation |
| `within the grace window`, `... no replacement this close to the end` | no decision |
| `exception ... in PBT checkpoint save/load (k/n)` | a failed iteration, counted toward `max_consecutive_errors` |

## Config reference

| Key | Default | Type, range | Meaning |
|---|---|---|---|
| `enabled` | `false` | bool | Read by train scripts to decide whether to attach the observer. The observer runs PBT whenever it is attached, and rank 0 warns when this is false. |
| `policy_idx` | `0` | int in [0, num_policies) | This member's index, unique in the population. |
| `num_policies` | `8` | int >= 1 | Members in the population; dexpbt needs 3 or more. |
| `directory` | `""` | str | Not read by the observer. Isaac Lab-style train scripts put their log root under it. |
| `workspace` | `pbt_workspace` | str | The workspace is `<train_dir>/<workspace>`, the same for every member. |
| `objective` | `""` | str, required | Dotted address of the objective in the env infos ([Objective](#objective)). |
| `objective_window` | `0` | int >= 0 | Finished episodes per rank averaged into the objective; 0: the rank's env count. |
| `interval_steps` | `100000` | int > 0 | Frames between PBT iterations. |
| `start_after` | `0` | int >= 0 | Frames after each process start, or an in-process self-mutation, before the member can be replaced; also part of the end-of-run margin. |
| `initial_delay` | `0` | int >= 0 | Frames from the start of training before the member can be replaced. |
| `replace_rule` | `""` | `""`, `dexpbt`, `threshold` | Empty: `threshold` when `threshold_std` or `threshold_abs` is set, else `dexpbt`. |
| `replace_fraction_worst` | `0.125` | float in (0, 1] | dexpbt: the worst group is the last `ceil(fraction * num_policies)` replaceable members. |
| `replace_fraction_best` | `0.3` | float in (0, 1] | dexpbt: donors come from the best `ceil(fraction * num_policies)` reported members. |
| `replace_threshold_frac_std` | `0.5` | float >= 0 | dexpbt: a transfer needs a lead of more than this many population stds (worst 20% left out). |
| `replace_threshold_frac_absolute` | `0.05` | float >= 0 | dexpbt: and a lead of more than this fraction of the donor's absolute objective. |
| `leader_params_prob` | `0.5` | float in [0, 1] | dexpbt: probability of mutating the donor's values rather than the member's own. |
| `threshold_std` | `null` | float or null | threshold: the std margin; null means 0.10. |
| `threshold_abs` | `null` | float or null | threshold: the absolute margin; null means 0.05. |
| `mutation_rate` | `0.25` | float in [0, 1] | Probability that a restart mutates each key. |
| `change_range` | `[1.1, 2.0]` | two finite numbers > 0, either order | Factor range of `mutate_float`, `mutate_float_min_1` and `mutate_eps_clip`. |
| `mutation` | `{}` | non-empty map | Flattened param name to mutation function ([Mutation functions](#mutation-functions)). |
| `launcher` | `""` | path or command | Executable a restart runs; empty: `sys.executable` ([The restart command](#the-restart-command)). |
| `checkpoint_arg` | `--checkpoint` | str | The train script's checkpoint argument; a restart passes `<name>=<restart.pth>`. |
| `reseed_on_restart` | `true` | bool | Replace the value of `seed_arg` on each restart's command with a derived seed ([Seeds](#seeds)). |
| `seed_arg` | `--seed` | str | The train script's seed argument (`seed` for a Hydra override). |
| `max_consecutive_errors` | `5` | int >= 1 | Raise after this many failed checkpoint save/load iterations in a row, or boundaries in a row with a failed rank ([Objective](#objective)). |
| `wandb_finish_timeout` | `120.0` | float, finite, >= 0 | Seconds a restart waits for wandb's `run.finish()` ([wandb](#wandb)). |
| `reuse_workspace` | `false` | bool | Let a fresh launch start over a member directory that holds an earlier run's files ([Reusing a workspace](#reusing-a-workspace)). |

Integer keys accept ints, whole floats (Hydra and OmegaConf turn `2e7` into a float) and numeric
strings (PyYAML and Isaac Lab's `yaml.full_load` turn `2e7` and `1e-3` into strings; `20_000_000`
works too). Float keys accept numbers and numeric strings. For both, bool, None (except the two
thresholds), NaN and non-numeric strings raise `ValueError` naming `pbt.<key>`, as do fractions
for integer keys, at init and on every rank.

Unknown keys raise `ValueError`, listing every unknown key with the key it likely means.
IsaacGymEnvs' `change_min` and `change_max` point to `change_range`; `dbg_mode` is not supported.

`rl_games.common.pbt.validate(params, extra_params=None, launcher_cwd=None)` returns
`(PbtCfg, mutable params)`. It touches no env and no tensors, so a train script calls it before
building the env; the observer calls it on every rank. It checks:

- the config, the objective and a non-empty `mutation`;
- that every mutation key resolves in the flattened `{'agent': params}` or in `extra_params`;
- the function names;
- the mutation values, which come back as plain Python numbers in config order: numpy scalars,
  0-d tensors and numeric strings are converted; bool, None, containers, NaN, inf and non-numeric
  strings raise naming the key;
- the integer keys ([Mutation functions](#mutation-functions));
- the launcher.

It prints a WARNING when `kl_threshold` is mutated without `lr_schedule: adaptive` (it has no
effect), and a note when `learning_rate` is mutated under `adaptive`: a mutation only moves the LR
the KL-driven schedule restarts from.

At init rank 0 also prints the resolved rule and its margins when `replace_rule` is unset; a
warning for keys of the inactive rule (`threshold_*` under dexpbt; `replace_fraction_*`,
`replace_threshold_frac_*` and `leader_params_prob` under threshold); the dexpbt warning for
`num_policies <= 2`; the warning for `enabled: false`; and a note for
`params.config.population_based_training` ([Porting](#porting-from-isaacgymenvs-and-isaac-lab)).

## Mutation functions

| Function | New value |
|---|---|
| `mutate_float` | multiplied or divided (even odds) by a factor drawn uniformly from `change_range` |
| `mutate_float_min_1` | `mutate_float`, floored at 1.0 |
| `mutate_eps_clip` | `mutate_float`, clamped to [0.01, 0.3] |
| `mutate_mini_epochs` | +1 or -1, clamped to [1, 8]; the value must be an integer |
| `mutate_discount` | 1 - x mutated by a factor in [1.1, 1.2], ignoring `change_range`: for gamma-like values near 1 |

A restart mutates each key with probability `mutation_rate`. A value of 0 never changes under
`mutate_float`, nor 1.0 under `mutate_discount`.

Integer settings are recognised by the last component of their key, so
`central_value_config.<key>` is covered too:

- `mini_epochs` and `mini_epochs_num` need `mutate_mini_epochs` or a custom function;
- `horizon_length`, `minibatch_size`, `minibatch_size_per_env`, `seq_length` and `num_actors` need
  a custom function. No built-in keeps them valid: `mutate_mini_epochs` clamps to [1, 8], the
  others return floats.

Integer values of float settings (e.g. `critic_coef: 4` with `mutate_float`) are fine.

Custom functions are added to `rl_games.common.pbt.mutation.MUTATION_FUNCS` before the observer or
`validate()` runs, in every process, e.g. when the train script is imported (restarts run it
again). They are called as `fn(value, change_min=..., change_max=...)` and should draw from
`random`, which holds the member's private stream then. They may return numpy or torch scalars:
the workspace, the restart command and its payload get plain Python numbers.

## Extending the observer

- To apply a self-mutation in the running process instead of restarting, override
  `_apply_self_mutation(self, changed: dict) -> bool`; don't copy `_pbt_iteration`, whose copy
  misses the replaceable marks, `restart.pth`'s base values, watermark and completed epoch, the
  no-op check, the end-of-run guard and the tie order. The hook is called on rank 0 for a
  self-mutation with changes, before anything is written for a restart; `changed` is
  `{flattened key: new value}`. Return True when the values were applied: the observer takes them
  as `pbt_params`, logs `pbt/<param>` at the iteration's x and the table
  `Mutated params (applied in process)`, appends an entry with `in_process: True` to the learner's
  `pbt_history`, restarts the grace window (its saves say `replaceable: false`) and keeps training.
  The base returns False and restarts. The subclass must apply the values on every rank (broadcast
  from rank 0, e.g. after `super().after_steps()`), and should call
  `self._reset_objective_window()` on every rank, so the next iteration ranks the member by
  episodes played with the new values.
- Integrations that write their own `pbt/*` scalars from `_pbt_iteration` use
  `self._summary_step(frame)` for the x.

## Limitations

- PPO agents only (`a2c_continuous`, `a2c_discrete`).
- Every restart re-executes the member: the env is built again, and the train script must accept
  the restart's command line.
- A member runs on one node.
- Members are not supervised. A member that stopped keeps its last checkpoint in its peers'
  ranking, as in IsaacGymEnvs: it can be a donor, and while it ranks lowest it can hold a worst
  slot. A crashed member stays down until it is resumed
  ([Reusing a workspace](#reusing-a-workspace)).
- The workspace needs a filesystem with atomic rename, writable only by the population's user.

## Porting from IsaacGymEnvs and Isaac Lab

- Mutation keys are flattened from the agent config: IsaacGymEnvs' `train.params.config.<key>`
  becomes `agent.params.config.<key>`, and the train script must accept that override name. Env
  values (`task.env.*`) go through `extra_params`, keyed by the train script's override name.
- `change_min` and `change_max` become `change_range: [min, max]`. IsaacGymEnvs' defaults differ:
  a change range of [1.1, 1.5] (rl_games: [1.1, 2.0]), `mutation_rate` 0.15, `interval_steps` 1e7,
  `start_after` 1e7, `initial_delay` 2e7. `dbg_mode` is not supported.
- Set `objective`: IsaacGymEnvs read `infos['true_objective']`, or the mean reward without it.
- Members start with their configured values (no `initial_pbt_check`); give them distinct seeds
  (`member_seed`).
- The restart passes neither `+full_experiment_name` nor `++pbt_restart=True`. Give each member a
  fixed `full_experiment_name` on every launch; rl_games marks a restart itself
  ([restart_info()](#restart_info-and-restartyaml)).
- Isaac Lab's `PbtCfg` keys are a subset of rl_games': its `pbt` sections construct as they are.
  With `threshold_std` or `threshold_abs` and no `replace_rule` they keep the threshold rule; set
  `replace_rule: dexpbt` for DexPBT's. Isaac Lab's train.py builds the observer without
  `launch_argv`, so its restarts drop launch flags ([The restart command](#the-restart-command)):
  pass `launch_argv`.
- `params.config.population_based_training` is an older, separate switch. It does not enable this
  PBT (attaching the observer does), needs `params.config.pbt_idx` (KeyError without it), adds
  `_pbt_<pbt_idx>` to the run name unless `full_experiment_name` is set, and routes summaries
  through `IntervalSummaryWriter`, which drops points. Rank 0 prints a note when it is set; remove
  it.

## Upgrading a running population

- Checkpoints gain `config_hparams`, and `pbt_history` after transfers; workspace yamls gain
  `replaceable`. Older readers ignore them, and yamls without `replaceable` count as replaceable.
- Resumes take `weight_decay`, and under identity the LR, from the config.
- A `restart.pth` written by an older parent has no base values: under `adaptive` the child scales
  the donor's walk by the ratio of its config LR to the donor's stored one, for that restart.
- `pbt` sections with unknown keys or invalid values raise at init.
- A section written for master #359 without `replace_rule`, `threshold_std` and `threshold_abs`
  runs the dexpbt rule (rank 0 prints the resolved rule); set `replace_rule: threshold` to keep
  the band rule.
- The loader of master #359 runs `int()` on every `*.yaml` name and fails on `restart.yaml`: don't
  mix #359 members with members of this version in one workspace.

## Testing and release qualification

`python -m pytest tests/test_pbt_e2e.py` runs in CI (about 30 s). A 3-member CPU population of the
example with real `os.execv` restarts takes about 15 s of it. The test checks the restart round
trip, the values in effect after each restart, epoch and frame continuity, distinct mutation
draws, that no secret from the environment reaches a log, and each restart's `kind` and
`restart.yaml`.

Multi-GPU members can't run in CI. Before a release, on a node with 2 GPUs:

```bash
bash examples/pbt/run_nccl_qualification.sh                    # fp32 weights, TF32 matmuls
bash examples/pbt/run_nccl_qualification.sh --precision fp16    # fp16 with loss scaling
bash examples/pbt/run_nccl_qualification.sh --lstm              # LSTM policies
```

Options: `--members N` (3 or more), `--nproc`, `--lr-schedule`, `--max-epochs`, `--out`,
`--python`, `--dry-run`. Each run is 3 members of 2 ranks (rank r on GPU r) on the dexpbt rule,
mutation_rate 0.5, the adaptive LR schedule, `torch_compile` and the normalizers on; the last
member alternates transfers and self-mutations. `check_qualification.py` prints PASS, or FAIL with
the failures: the ranks of every process start must agree on state hashes, counters, restart info
and values in effect; every restart must have restored `restart.pth` (the model, the fp16 loss
scale); every rank must end normally; some member must restart at least 3 times. The members'
`--report` lines (a `PBT_MEMBER` JSON line per rank at the first env step) belong to the example,
not to the library API: a debugging aid for DDP state sync.

Results: record each run's commit, node and outcome here. None yet. Still to run: the three
commands above, and one with 4 ranks per member (`--nproc 4`) on a 4-GPU node.
