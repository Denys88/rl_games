# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import pbt_utils
from rl_games.common.pbt.mutation import clamp_to_bounds, mutate
from rl_games.common.pbt.pbt_cfg import (  # noqa: F401 (PbtCfg: public here since #359)
    PbtCfg, load_objective_fn, rule_notices, settings_hash, validate)

_UNINITIALIZED_VALUE = pbt_utils.UNINITIALIZED_VALUE

# prefix of the algo-config keys among the flattened mutable params
_CONFIG_PREFIX = "agent.params.config."

# rank 0's decision at a PBT iteration, broadcast to every rank (`PbtAlgoObserver._sync_decision`);
# 0 keeps training
_RESTART, _MUTATE_IN_PROCESS, _TRANSFER_IN_PROCESS = 1, 2, 3

# what an exec'd restart inherits for wandb (`_finish_wandb`), recorded under restart_mode exit
_WANDB_RESTART_VARS = ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_INIT_TIMEOUT")


class _PooledWindow(NamedTuple):
    """The objective windows of all ranks and their episode counts since the last PBT iteration."""

    total: float       # objective sum over the windows
    count: int         # episodes in the windows
    known: bool        # every rank's window is full
    stale_ranks: int   # ranks without a finite objective since the last iteration
    failed_ranks: int  # stale ranks whose every env ended an episode since then (and a window of them)
    valid: int         # episodes since then with a finite objective
    nonfinite: int     # ... with a NaN or infinite one
    unread: int        # ... on steps whose objective could not be read
    ranks: int         # world size (1 without distributed)


def _num_finished(done_indices) -> int:
    """Episodes a step finished: one per done index, one sample without indices."""
    if done_indices is None:
        return 1
    try:
        return int(done_indices.numel()) if torch.is_tensor(done_indices) else int(np.size(done_indices))
    except (TypeError, ValueError):
        return 0


def build_restart_args(cli_args, new_params, restart_from_checkpoint,
                       wandb_args=None, rendering_args=None, checkpoint_arg="--checkpoint", *, drop_args=None):
    """Rebuild the training command line for a PBT restart from `sys.argv`-style args.

    Drops hydra-style ``key=value`` overrides that are being replaced by ``new_params``, any
    existing checkpoint argument and the launch-only `drop_args` ({name: nargs}, see
    `pbt_utils.remove_cli_arg`; exact name match), keeps everything else verbatim, then appends
    the new checkpoint, the wandb and rendering flags `cli_args` doesn't have (its own stay as
    given: the command doesn't grow at every restart) and the mutated params.
    """
    out = [cli_args[0]] + pbt_utils.remove_cli_arg(list(cli_args[1:]), checkpoint_arg)
    for name, nargs in (drop_args or {}).items():
        out = [out[0]] + pbt_utils.remove_cli_arg(out[1:], name, nargs)
    out = [out[0]] + [arg for arg in out[1:] if not ("=" in arg and arg.split("=", 1)[0] in new_params)]
    out.append(f"{checkpoint_arg}={restart_from_checkpoint}")
    for flags in (wandb_args, rendering_args):
        if flags is not None:
            out.extend(pbt_utils.missing_flags(flags.get_args_list(), cli_args[1:]))
    for param, value in new_params.items():
        out.append(f"{param}={value}")
    return out


def _wandb_run():
    wandb = sys.modules.get("wandb")
    return getattr(wandb, "run", None) if wandb is not None else None


def _flush_stdio():
    """Flush Python's stdout/stderr: os.execv and os._exit discard buffered output, which loses
    the last log lines when output goes to a file or a pipe."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except (OSError, ValueError):
            pass


@contextlib.contextmanager
def _uncompiled(algo):
    """The learner's actor-critic and central value models as the modules torch.compile wraps,
    for a load of checkpoint keys without '_orig_mod.' (`torch_ext.load_checkpoint` strips it), as
    a restarted process restores before the runner compiles. The compiled models share their
    parameters, so they compute with what was loaded."""
    holders = [algo] + ([algo.central_value_net] if algo.has_central_value else [])
    models = [holder.model for holder in holders]
    for holder, model in zip(holders, models):
        holder.model = getattr(model, "_orig_mod", model)
    try:
        yield
    finally:
        for holder, model in zip(holders, models):
            holder.model = model


class PbtAlgoObserver(AlgoObserver):
    """rl_games observer that implements Population-Based Training for a single policy process.

    Each learner in the population runs as its own process with a unique `policy_idx`. On a
    fixed cadence (`interval_steps`) every learner saves a scored checkpoint into a shared
    workspace, compares itself against the population, and — if the replacement rule selects
    it — re-execs itself from a better member's checkpoint (or its own) with mutated
    hyperparameters. With `pbt.self_mutation: in_process` a mutation of its own values is set
    on the running learner instead (`_apply_self_mutation`), and with `pbt.transfer: in_process`
    a better member's checkpoint loads into it (`_load_transfer`).

    Restarts need a train script that accepts `key=value` hyperparameter overrides (Hydra
    style, as in Isaac Lab / IsaacGymEnvs train scripts) and a checkpoint argument. Pass the
    unmodified launch command as `launch_argv` when the script rewrites `sys.argv` (e.g. for
    Hydra); the restart then re-runs it with the checkpoint and the mutated overrides
    replaced, the seed replaced when the command passes `seed_arg` and `reseed_on_restart` is
    set, the script path made absolute, and `hydra.run.dir` pinned to the run dir inside a
    Hydra job that changed into it. The restart runs from the launch directory (Hydra's
    original cwd inside a Hydra job), see `pbt_utils.launch_dirs`. With `pbt.restart_mode: exit`
    the member ends instead, for its supervisor (`python -m rl_games.common.pbt.supervise`) to
    run its command again (`_exit_for_restart`).

    The restarted process receives the decision as `pbt_utils.restart_info()`; the member's
    `restart.yaml`, written at the decision next to `restart.pth`, records it with every new
    param value, the seed and the epoch (`_restart_record`). Rank 0 appends each iteration's
    outcome to the member's `decisions.jsonl` (`_append_decision`), and the learner's checkpoints
    carry the member's PBT state (`_pbt_state`), from which `pbt_utils.member_resume` relaunches
    a member that died.

    Rank 0 keeps the member's owner record (`owner.yaml`, `_claim_member`, `_heartbeat`), from
    which peers tell a live member from a dead one (`pbt.max_member_lag`), and writes an exit
    record at a normal end (`after_train`). With `pbt.stop_with_first` the population ends with
    its first member to finish (`_watch_population_end`).
    """

    def __init__(self, params, args_cli=None, launch_argv=None, extra_params=None, code_fingerprint=None,
                 objective_fn=None):
        """Validate the PBT setup, print the rule and the mutation table, and allocate the restart flag.

        Every rank runs the same checks (`validate()`, single-node members) before any
        collective, so an invalid setup fails at init on every rank instead of at the first
        restart on rank 0.

        Args:
            params (dict): Full agent/task params (Hydra style), containing a "pbt" section.
            args_cli: Parsed CLI args used to reconstruct the restart command when `launch_argv`
                is not given. Any argparse namespace works; missing attributes are treated as
                disabled/absent.
            launch_argv (list[str] | None): The original script argv (script path first),
                captured before argparse/Hydra rewrite `sys.argv`. For a `python -m` launch:
                `['-m', 'pkg.module', *sys.argv[1:]]`.
            extra_params (dict | None): Additional mutable values outside the agent params,
                keyed by their `key=value` override name (e.g. `env.rewards.success.weight`).
            code_fingerprint (str | None): An identifier of the integration's code (e.g. a hash of
                its sources). Workspace yamls record it, and a member that reads another one from
                a peer leaves that peer out of its ranking (`_foreign_members`).
            objective_fn (callable | None): The objective function as a function object, in
                place of `pbt.objective_fn` (see `PbtCfg.objective_fn`): fn(infos, done_indices,
                num_agents, algo), called at every env step.

        Raises:
            ValueError: invalid PBT config (see `validate()`), or a distributed member whose
                ranks span several nodes.
        """
        super().__init__()
        self.printer = pbt_utils.PbtTablePrinter()
        self.launch_argv = list(launch_argv) if launch_argv is not None else None
        # restarts run from the launch directory, pinned to the Hydra run dir when Hydra changed into it
        self.launch_cwd, self.hydra_run_dir = pbt_utils.launch_dirs()

        self.rendering_args = pbt_utils.RenderingArgs(args_cli)
        # the launch argv already carries the wandb flags
        self.wandb_args = pbt_utils.WandbArgs(args_cli if self.launch_argv is None else None)
        self.env_args = pbt_utils.EnvArgs(args_cli)
        self.distributed_args = pbt_utils.DistributedArgs(args_cli)
        # a relative launcher is resolved where the restart execs from
        self.cfg, self.pbt_params = validate(params, extra_params, launcher_cwd=self.launch_cwd,
                                             objective_fn=objective_fn)
        # None: process_infos looks up pbt.objective
        self._objective_fn = objective_fn if objective_fn is not None else load_objective_fn(self.cfg.objective_fn)
        self.dir = self.cfg.directory  # kept for compatibility; the workspace is <train_dir>/<workspace>
        if self.distributed_args.distributed and "LOCAL_WORLD_SIZE" in os.environ:
            local_world_size, world_size = int(os.environ["LOCAL_WORLD_SIZE"]), int(os.environ.get("WORLD_SIZE", "1"))
            if local_world_size != world_size:
                raise ValueError(
                    f"PBT members must run on a single node: LOCAL_WORLD_SIZE={local_world_size} differs from "
                    f"WORLD_SIZE={world_size}. A restart re-execs the whole member from rank 0 as "
                    f"torch.distributed.run --standalone --nnodes=1 --nproc_per_node=WORLD_SIZE, i.e. on rank 0's "
                    f"node")
        self.pbt_it = -1  # dummy value, stands for "not initialized"
        self.initial_frame = 0
        self.score = _UNINITIALIZED_VALUE
        self.best_in_iteration = None
        self._window = deque()
        self._window_sum = 0.0
        self._window_count = 0
        self._window_size = max(1, self.cfg.objective_window)
        self._consecutive_errors = 0
        self._last_objective_error = None
        # this rank's finished episodes since the last PBT iteration: with a finite objective (in
        # the window), with a NaN/inf one, and on steps whose objective could not be read
        self._interval_valid = self._interval_nonfinite = self._interval_unread = 0
        # the envs that ended an episode since the last iteration while the rank had no finite
        # objective (sized in after_init): with every env marked, the rank has failed (diverged)
        self._failed_envs = np.zeros(1, dtype=bool)
        self._wait_iteration = None  # the boundary the next iteration waits at
        self._failed_boundaries = 0  # consecutive boundaries waited at with a failed rank
        self.code_fingerprint = None if code_fingerprint is None else str(code_fingerprint)
        self._identity_warnings = set()  # (peer, field, value) already warned about
        # this launch of the member, in its owner record; liveness: the wall seconds per iteration
        # interval and the last boundary timed (rank 0)
        self.launch_id = uuid.uuid4().hex
        self._iteration_seconds = deque(maxlen=20)
        self._boundary = None
        self._stop_fn = None  # the learner's stop_fn under pbt.stop_with_first
        self._ignored_marker = None  # the time of a finished.yaml of another run, warned about

        if self.distributed_args.rank == 0:
            for line in rule_notices(self.cfg, params["pbt"]):
                print(f"PbtAlgoObserver: {line}")
            if not self.cfg.enabled:
                print("PbtAlgoObserver: WARNING pbt.enabled is False, but the observer runs PBT whenever it is "
                      "attached; train scripts read the flag to decide whether to attach it")
            algo_config = params.get("params", {}).get("config", {})
            if algo_config.get("population_based_training"):
                print("PbtAlgoObserver: note: params.config.population_based_training is an older, separate switch: "
                      "it does not enable this PBT (attaching the observer does); it adds '_pbt_<pbt_idx>' to the "
                      "experiment name unless full_experiment_name is set, and rate-limits summaries through "
                      "IntervalSummaryWriter"
                      + ("" if "pbt_idx" in algo_config else
                         "; params.config.pbt_idx is missing, so the agent will raise KeyError('pbt_idx')"))
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not self.distributed_args.distributed:
            print("PbtAlgoObserver: WARNING WORLD_SIZE > 1 but args_cli.distributed is not set: rank 0's PBT "
                  "decisions are not broadcast and ranks would diverge on a restart; set args_cli.distributed=True")
        if self.cfg.force_outcome:
            # on every rank: no log of a forced member may pass for a PBT run
            print(f"PbtAlgoObserver: WARNING pbt.force_outcome is {self.cfg.force_outcome!r}: every iteration outside "
                  f"the grace windows restarts this member from the source it names, whatever the replacement rule "
                  f"says. For qualifying an integration, not for training")
        if self.launch_argv is None and self.distributed_args.rank == 0 and pbt_utils.argv_rewritten():
            print("PbtAlgoObserver: NOTICE launch_argv is not set and sys.argv differs from the launch command "
                  "(sys.orig_argv): restarts rebuild the command from sys.argv and args_cli and drop the other "
                  "launch flags; pass launch_argv=list(sys.argv) captured before sys.argv is rewritten, or "
                  "['-m', '<module>', *sys.argv[1:]] for a `python -m` launch")
        # PBT's draws come from a generator private to this member and process start (`_draw_rng`)
        self._seed = params.get("params", {}).get("seed")
        self._rng = None
        seed_arg = self.cfg.seed_arg
        # a seed of None or -1 is drawn anew at every launch: nothing replays then
        if (self.launch_argv is not None and self.distributed_args.rank == 0 and self.cfg.reseed_on_restart
                and seed_arg and pbt_utils.get_cli_arg(self.launch_argv, seed_arg) is None
                and not pbt_utils._random_per_launch(self._seed)):
            print(f"PbtAlgoObserver: NOTICE launch_argv has no {seed_arg} argument: every restart seeds torch, "
                  f"numpy and the environment with the launch seed again, so their random streams replay after "
                  f"each restart (PBT's own draws come from a private generator); pass {seed_arg} on the command "
                  f"line, or set pbt.seed_arg to the name it uses (e.g. seed for a Hydra override), and each "
                  f"restart gets a seed derived from it")

        self.printer.print_params_table(self.pbt_params, header="List of params to mutate")

        info = pbt_utils.restart_info()
        self.restart = info if info is not None and info.get("policy_idx") == self.cfg.policy_idx else None
        self.restart_count = int(self.restart["restart_count"]) if self.restart else 0
        if self.cfg.initial_mutation:
            self._initial_mutation(params)

        if self.distributed_args.distributed and torch.cuda.is_available():
            # NCCL collectives require CUDA tensors
            self.device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
        else:
            self.device = params.get("params", {}).get("config", {}).get("device", "cpu")
        # rank 0's decision at a PBT iteration, which every rank acts on in the same step
        # (`_sync_decision`): a restart (`_pbt_iteration` sets restart_flag[0] = 1, as subclasses
        # that replace it do), an in-process self-mutation (`_apply_self_mutation`) or transfer
        self.restart_flag = torch.tensor([0])
        self._mutate_in_process = self._transfer_in_process = False
        # every rank: (changed values, payload) of an in-process transfer, loaded at the next before_steps
        self._pending_transfer = None
        self._before_steps_called = False

    def _initial_mutation(self, params):
        """`pbt.initial_mutation`: mutate a fresh member's `agent.params.config` values once, in
        `params` (the train script hands them to runner.load afterwards) and in `pbt_params`.

        Every rank decides and draws alike, before any process group exists. Fresh: no restart
        payload for the member and no numbered checkpoint in its workspace directory, which every
        rank lists before rank 0 saves the first; rank 0 says when checkpoints there (a relaunch's,
        or an earlier run's under `reuse_workspace`) skip the mutation. The generator is seeded
        with the launch command's `seed_arg` value and the member, not with params['seed'], to
        which train scripts add the rank. A draw that changes no value is drawn again, at most 20
        times. Env keys and extra_params keep their values: the env they apply to is built from
        its own config.
        """
        cfg = self.cfg
        if self.restart is not None:
            return
        rank0 = self.distributed_args.rank == 0
        config = params.get("params", {}).get("config", {})
        member_dir = os.path.join(config.get("train_dir", "runs"), cfg.workspace, f"{cfg.policy_idx:03d}")
        saved = pbt_utils.saved_iterations(member_dir) if os.path.isdir(member_dir) else []
        if saved:
            if rank0:
                print(f"PbtAlgoObserver: no initial mutation: {member_dir} holds checkpoints (iterations "
                      f"{saved[0]}..{saved[-1]}), so this launch counts as a relaunch of the member and keeps its "
                      f"values; delete the member directories to start a population over with initial mutations")
            return
        current = {key: value for key, value in self.pbt_params.items() if key.startswith(_CONFIG_PREFIX)}
        kept = [key for key in self.pbt_params if key not in current]
        if kept and rank0:
            print(f"PbtAlgoObserver: the initial mutation leaves {kept} as configured: only {_CONFIG_PREFIX}* "
                  f"values apply before the learner is built")
        if not current:
            return
        command = self.launch_argv if self.launch_argv is not None else sys.argv
        seed = pbt_utils.get_cli_arg(command, cfg.seed_arg) if cfg.seed_arg else None
        with pbt_utils.swap_random_state(random.Random(f"pbt-init:{seed}:{cfg.policy_idx}")):
            for _ in range(20):
                new = clamp_to_bounds(mutate(current, cfg.mutation, cfg.mutation_rate, cfg.change_range),
                                      cfg.mutation_bounds)
                if new != current:
                    break
        changed = {key: value for key, value in new.items() if value != current[key]}
        if not changed:
            if rank0:
                print(f"PbtAlgoObserver: the initial mutation changed no value in 20 draws (mutation_rate "
                      f"{cfg.mutation_rate:g}): the member starts from the configured values")
            return
        for key, value in changed.items():
            *path, leaf = key[len(_CONFIG_PREFIX):].split(".")
            node = config
            for part in path:
                node = node[part]
            node[leaf] = value
        new_params = dict(self.pbt_params, **changed)
        if rank0:
            self.printer.print_mutation_diff(self.pbt_params, new_params, header="Initial mutation")
        self.pbt_params = new_params

    def before_init(self, base_name, config, experiment_name):
        """On a PBT restart of this member, set `config['pbt_restart']`.

        The checkpoint holds a trained policy (the member's or a donor's), so the runner keeps
        its sigma instead of applying the launch command's `--sigma`. Every rank inherits the
        restart payload, so all ranks agree. Mutated values need no marking: the restore takes
        a hyperparameter from the config whenever it differs from the checkpoint's stored one.
        Rank 0 warns when the config has no `full_experiment_name`.
        """
        if self.distributed_args.rank == 0 and not config.get("full_experiment_name"):
            print("PbtAlgoObserver: WARNING the config has no full_experiment_name: every restart writes into a new "
                  "timestamped run directory, and members started in the same second share one; set "
                  "agent.params.config.full_experiment_name per member")
        if self.restart is not None:
            config["pbt_restart"] = True

    def after_init(self, algo):
        """Keep the algo and give the learner this member's PBT state (`_pbt_state`) on every rank;
        rank 0 creates this policy's workspace folder and claims the member (`_claim_member`). In a
        PBT restart rank 0 notices when the learner's `experiment_name` differs from the one the
        member's previous process wrote into (the payload's): a new run directory.

        Args:
            algo: rl_games algorithm object (provides writer, train_dir, frame counter, etc.).

        Raises:
            NotImplementedError: for SAC, which never calls `after_steps` (PBT would do nothing).
            RuntimeError: on rank 0, when another process may still run this member.
        """
        from rl_games.algos_torch.sac_agent import SACAgent  # lazy: only this check needs the SAC module

        if isinstance(algo, SACAgent):
            raise NotImplementedError(
                "PBT supports the PPO agents (a2c_continuous, a2c_discrete); SAC never calls the observer's "
                "after_steps, so PBT would silently do nothing")
        self.algo = algo
        self.population = self._population(algo)
        algo.pbt_state = self._pbt_state()
        num_envs = max(1, int(getattr(algo, "num_actors", 1) or 1))
        if self.cfg.objective_window <= 0:
            self._window_size = num_envs
        # a failed rank needs a failed episode on every env: one env in a NaN reset loop while the
        # others are mid-episode is not a diverged policy
        self._failed_envs = np.zeros(num_envs, dtype=bool)
        if self.distributed_args.rank != 0:
            return

        previous = (getattr(self, "restart", None) or {}).get("experiment_name")
        current = getattr(algo, "experiment_name", None)
        if previous and current and previous != current:
            print(f"Policy {self.cfg.policy_idx}: NOTICE this restarted process writes into run directory "
                  f"'{current}', the member's previous process into '{previous}': each launch gets a new experiment "
                  f"name (a timestamp), so the member's summaries and nn/ checkpoints are split across run "
                  f"directories; give each member a fixed full_experiment_name on every launch")

        self.root_dir = algo.train_dir
        self.ws_dir = os.path.join(self.root_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        os.makedirs(self.curr_policy_dir, exist_ok=True)
        self._claim_member()

    def _claim_member(self):
        """Rank 0 at init, before any collective (torchrun stops the other ranks when it raises):
        write the member's owner record (owner.yaml, `pbt_utils.owner_record`), which names this
        process and launch (`launch_id`) and is renewed at every iteration boundary (`_heartbeat`).

        A plain launch refuses to start while the process the record names may still run the
        member (`pbt_utils.owner_alive`; not once the member's exit record says that launch
        ended), unless `pbt.force_owner`: two processes of one member would write the same files.
        A process with this member's restart payload (a PBT restart, a `member_resume` relaunch)
        takes the member over. The member's median iteration time continues from the record.
        """
        record = pbt_utils.read_record(self.curr_policy_dir, pbt_utils.OWNER_RECORD) or {}
        exit_record = pbt_utils.read_record(self.curr_policy_dir, pbt_utils.EXIT_RECORD)
        takeover = getattr(self, "restart", None) is not None or self.cfg.force_owner
        if record and not takeover and pbt_utils.owner_alive(record, self.cfg.max_member_lag, exit_record):
            heartbeat = record.get("heartbeat")
            age = f"{time.time() - heartbeat:.0f} s ago" if isinstance(heartbeat, (int, float)) else "unknown"
            raise RuntimeError(
                f"Policy {self.cfg.policy_idx}: {pbt_utils.OWNER_RECORD} in {self.curr_policy_dir} says pid "
                f"{record.get('pid')} on {record.get('host')} runs this member (heartbeat {age}), and it may still "
                f"run: two processes of one member would write the same files. Stop it first, or set "
                f"pbt.force_owner: true when it no longer runs this member; a PBT restart or a member_resume "
                f"relaunch (RL_GAMES_PBT_RESTART) takes the member over")
        seconds = record.get("iteration_seconds")
        if isinstance(seconds, (int, float)) and seconds > 0:
            self._iteration_seconds.append(float(seconds))
        self._claimed = time.time()
        self._write_owner()

    def _write_owner(self):
        median = statistics.median(self._iteration_seconds) if self._iteration_seconds else None
        pbt_utils.write_record(self.curr_policy_dir, pbt_utils.OWNER_RECORD, pbt_utils.owner_record(
            self.launch_id, self.restart_count, self._claimed, median))

    def _heartbeat(self, iteration):
        """Rank 0 at each iteration boundary, also one that waits for its objective: time the
        interval since the last boundary (the member's iteration wall time) and renew the owner
        record. Returns why this process must stop instead, else None: the record names another
        launch, which took the member over (the caller raises after the step's collectives). A
        record that can't be read or written only warns: peers may count the member stale."""
        now = time.time()
        if self._boundary is None or iteration > self._boundary[0]:
            if self._boundary is not None:
                self._iteration_seconds.append((now - self._boundary[1]) / (iteration - self._boundary[0]))
            self._boundary = (iteration, now)
        try:
            record = pbt_utils.read_record(self.curr_policy_dir, pbt_utils.OWNER_RECORD)
            if record is not None and record.get("launch_id") != self.launch_id:
                return (f"Policy {self.cfg.policy_idx}: another process took this member over: "
                        f"{pbt_utils.OWNER_RECORD} in {self.curr_policy_dir} names launch {record.get('launch_id')} "
                        f"(pid {record.get('pid')} on {record.get('host')}), this process is launch {self.launch_id}. "
                        f"It stops, so that one process writes the member's files")
            self._write_owner()
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not renew {pbt_utils.OWNER_RECORD}: {exc!r}")
        return None

    def _member_lag(self):
        """Seconds without a sign of life after which a peer is stale: max_member_lag × this
        member's median iteration wall time; None while unknown, and with max_member_lag 0."""
        if not (self.cfg.max_member_lag and self._iteration_seconds):
            return None
        return self.cfg.max_member_lag * statistics.median(self._iteration_seconds)

    def _population(self, algo):
        """What this member runs, recorded in its workspace yamls and learner checkpoints (K02): the
        rl_games version and git commit, the integration's `code_fingerprint`, the hash of the
        population settings (`settings_hash`), the unit its iterations count (`interval_unit`) and
        the learner's precision (the resolved mixed precision and the torch_compile setting). See
        `_foreign_members`."""
        mixed_precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(getattr(algo, "amp_dtype", None), "off")
        torch_compile = (getattr(algo, "config", None) or {}).get("torch_compile", True)  # the runner's default
        return {"rl_games_version": pbt_utils.rl_games_version(), "git_commit": pbt_utils.git_commit(),
                "code_fingerprint": self.code_fingerprint, "settings": settings_hash(self.cfg),
                "interval_unit": self.cfg.interval_unit,
                "precision": {"mixed_precision": mixed_precision, "torch_compile": torch_compile}}

    def _pbt_state(self, params=None, restart_count=None):
        """The learner's PBT state (`A2CBase.pbt_state`, which its checkpoints carry as 'pbt'): this
        member, its params (default `pbt_params`), its restart count (default this process's) and
        what it runs (`population`). Workspace yamls record the count and `population` from it."""
        return {"policy_idx": self.cfg.policy_idx, "params": dict(self.pbt_params if params is None else params),
                "restart_count": self.restart_count if restart_count is None else restart_count,
                "population": dict(self.population)}

    def _set_pbt_params(self, new_params):
        """Take `new_params` as this member's params, in the learner's PBT state too (its next
        checkpoints carry them): for values applied in the running process."""
        self.pbt_params = new_params
        self.algo.pbt_state = self._pbt_state()

    def _take_learner_state(self):
        """First step of the process, every rank: the restore replaced the learner's PBT state with
        the checkpoint's (or none). The restart count continues from a state of this member when it
        is higher than this process's (a resume by hand: no payload), so the member's next restarts
        derive new seeds. The learner then takes this process's state again."""
        restored = getattr(self.algo, "pbt_state", None)
        if restored and restored.get("policy_idx") == self.cfg.policy_idx:
            self.restart_count = max(self.restart_count, int(restored.get("restart_count", 0)))
        self.algo.pbt_state = self._pbt_state()

    def process_infos(self, infos, done_indices):
        """Accumulate the objective of finished episodes into a window of recent episodes.

        The objective is `pbt.objective` looked up in `infos`, or the objective function's value
        (`pbt.objective_fn`), whose exceptions propagate. `self.score` is this rank's mean over
        the window. An episode whose objective is NaN or infinite is left out on its own (see
        `pbt_utils.episode_stats`). For the iteration gate the rank counts its finished episodes
        since the last PBT iteration: with a finite objective, with a non-finite one, and on steps
        without one (it could not be read, or the function returned None). Until it has a finite
        one, it also marks the envs that ended an episode.
        """
        algo = getattr(self, "algo", None)
        num_agents = int(getattr(algo, "num_agents", 1) or 1)
        stats = error = None
        if self._objective_fn is not None:
            # the user's code: an exception is a bug, not an objective missing on this step
            value = self._objective_fn(infos, done_indices, num_agents, algo)
            if value is None:
                error = f"{self._objective_name()} returned None"
            else:
                stats = pbt_utils.episode_stats(value, done_indices, num_agents)
        else:
            try:
                value = infos
                for part in self.cfg.objective.split("."):
                    value = value[part]
                stats = pbt_utils.episode_stats(value, done_indices, num_agents)
            except (KeyError, TypeError, IndexError, ValueError, RuntimeError) as exc:
                error = repr(exc)  # the address may not resolve on every step (or every backend)
        if stats is None:
            # keep the previous score, and report the last failure if no objective arrives by an
            # iteration boundary
            self._last_objective_error = error
            self._interval_unread += _num_finished(done_indices)
            self._mark_failed_envs(done_indices, num_agents)
            return
        total, count, nonfinite = stats
        if count and not math.isfinite(total):
            nonfinite, count = nonfinite + count, 0  # finite values whose sum overflows
        self._interval_nonfinite += nonfinite
        if count == 0:
            if nonfinite:
                self._mark_failed_envs(done_indices, num_agents)
            return
        self._interval_valid += count
        self._window.append((total, count))
        self._window_sum += total
        self._window_count += count
        while len(self._window) > 1 and self._window_count - self._window[0][1] >= self._window_size:
            old_total, old_count = self._window.popleft()
            self._window_sum -= old_total
            self._window_count -= old_count
        self.score = self._window_sum / self._window_count

    def _objective_name(self):
        """The objective in messages: `pbt.objective`, or the objective function."""
        if self._objective_fn is None:
            return self.cfg.objective
        return self.cfg.objective_fn or getattr(self._objective_fn, "__qualname__", repr(self._objective_fn))

    def _mark_failed_envs(self, done_indices, num_agents):
        """Mark the envs that ended an episode on a step without a finite objective. Only needed
        while the rank has had no finite objective since the last iteration (one makes it fresh
        until the next iteration, whatever the mask holds), so the done indices are copied to
        the host (one sync) on such steps only."""
        if self._interval_valid or self._failed_envs.all():
            return
        if done_indices is None:
            self._failed_envs[:] = True  # no env indices: the sample stands for the rank
            return
        try:
            env_idx = torch.as_tensor(done_indices).reshape(-1).cpu().long().numpy() // max(1, num_agents)
        except (TypeError, ValueError, RuntimeError):
            return
        self._failed_envs[env_idx[(env_idx >= 0) & (env_idx < self._failed_envs.size)]] = True

    def _distributed(self):
        return self.distributed_args.distributed and dist.is_available() and dist.is_initialized()

    def _pooled_window(self, distributed):
        """The episode windows of all ranks and their episode counts since the last PBT iteration,
        pooled in the one all-reduce every rank makes per step (see `_PooledWindow`).

        A rank is stale without a finite objective since the last iteration, and failed when it
        is stale although every one of its envs ended an episode since then, and at least a
        window of episodes ended (all non-finite or unreadable: a diverged policy, or infos
        without the objective; not episodes yet to end, nor a few envs failing on their own).
        """
        fresh = self._interval_valid > 0
        failed = (not fresh and self._interval_nonfinite + self._interval_unread >= self._window_size
                  and bool(self._failed_envs.all()))
        local = [self._window_sum, self._window_count, self._window_count >= self._window_size, fresh, failed,
                 self._interval_valid, self._interval_nonfinite, self._interval_unread]
        ranks = 1
        if distributed:
            sums = torch.tensor([float(v) for v in local], dtype=torch.float64, device=self.device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            local, ranks = sums.tolist(), dist.get_world_size()
        total, count, full_ranks, fresh_ranks, failed_ranks, valid, nonfinite, unread = local
        return _PooledWindow(total=float(total), count=int(count), known=int(full_ranks) == ranks,
                             stale_ranks=ranks - int(fresh_ranks), failed_ranks=int(failed_ranks), valid=int(valid),
                             nonfinite=int(nonfinite), unread=int(unread), ranks=ranks)

    def after_steps(self):
        """Main PBT tick executed every train step.

        Flow:
            1) The first step after a (re)start only sets the PBT cadence from the restored
               frame (or epoch) and gives the learner this process's PBT state again
               (`_take_learner_state`); rank 0 checks the member directory
               (`_check_member_dir`), and a restarted process logs the restart's cost
               (`_record_restart`).
            2) All ranks: pool the objective windows and episode counts across ranks and
               track the best pooled objective of the iteration.
            3) On the PBT cadence (`interval_steps` frames or epochs, see `_progress`), once
               every rank has a full window and a finite objective since the last iteration,
               rank 0 renews the member's owner record (`_heartbeat`), saves a checkpoint, loads
               the population, logs, and applies the replacement rule (`_pbt_iteration`), with
               the global `random` module swapped to the member's private generator
               (`_draw_rng`). Until then the iteration is retried on every step without
               advancing `pbt_it` (see `_wait_for_objective`). When the iteration decided a
               restart (the flag is set) or an in-process transfer, rank 0 writes the member's
               restart.yaml (`_save_restart_record`), unless the population has finished
               (`pbt.stop_with_first`); then it appends the iteration's outcome to
               decisions.jsonl (`_append_decision`).
            4) All ranks, at the iteration: act on rank 0's decision (`_sync_decision`): the
               member restarts before this epoch's update, applies an in-process
               self-mutation, or loads an in-process transfer at the start of the next epoch
               (`before_steps`). Then rank 0 raises if another launch took the member over.
        With `pbt.stop_with_first` rank 0 first watches for the population's end
        (`_watch_population_end`). With `pbt.transfer: in_process` the first step raises on every
        rank when the learner's before_steps doesn't reach the observer.
        """
        distributed = self._distributed()
        frame = self.algo.frame
        progress = self._progress(frame)
        if self.cfg.stop_with_first and self.distributed_args.rank == 0:
            self._watch_population_end(frame)
        if self.pbt_it == -1:
            if self.cfg.transfer == "in_process" and not self._before_steps_called:
                # every rank alike: the same observer chain
                raise RuntimeError(
                    f"Policy {self.cfg.policy_idx}: pbt.transfer is in_process, but the learner's before_steps "
                    f"does not reach this observer, which loads a transfer there: an observer chain must forward "
                    f"before_steps (rl_games' MultiObserver does), or set pbt.transfer: restart")
            # first step after (re)start: the checkpoint restore has already set the frame and epoch
            self.pbt_it = progress // self.cfg.interval_steps
            self.initial_frame = progress
            self._take_learner_state()
            if self.distributed_args.rank == 0:
                self._check_member_dir(frame)
                if getattr(self, "restart", None) is not None:
                    self._record_restart(frame)
            return

        pooled = self._pooled_window(distributed)
        if pooled.known:
            objective = pooled.total / pooled.count
            if self.best_in_iteration is None or objective > self.best_in_iteration:
                self.best_in_iteration = objective

        iteration = progress // self.cfg.interval_steps
        if iteration <= self.pbt_it:
            return
        if not pooled.known or pooled.stale_ranks:
            # decided on pooled values: every rank waits (or raises) at the same step
            self._wait_for_objective(iteration, pooled)
            return

        self.pbt_it = iteration
        # the next iteration needs a new objective on every rank; its warnings report its own lookup errors
        self._interval_valid = self._interval_nonfinite = self._interval_unread = 0
        self._failed_envs[:] = False
        self._wait_iteration, self._failed_boundaries = None, 0
        self._last_objective_error = None
        best_in_iteration, self.best_in_iteration = self.best_in_iteration, None
        previous = dict(self.pbt_params)  # every rank's values before rank 0's decision
        taken_over = self._heartbeat(iteration) if self.distributed_args.rank == 0 else None
        if self.distributed_args.rank == 0 and taken_over is None:
            objective = pooled.total / pooled.count
            # PBT's draws (donor, leader-params coin, mutations) come from this member's private stream,
            # also in a subclass that replaces _pbt_iteration and calls `random` itself
            with pbt_utils.swap_random_state(self._draw_rng()):
                decision = self._pbt_iteration(objective, best_in_iteration, frame)
            if int(self.restart_flag[0]) == 1 and self.cfg.stop_with_first and self._population_finished():
                # a subclass's _pbt_iteration (the base decides none once the population has finished)
                self.restart_flag[0] = 0
                print(f"Policy {self.cfg.policy_idx}: the population has finished ({pbt_utils.FINISHED_MARKER}): "
                      f"the restart is dropped")
                decision = {"outcome": "ending", f"{self.cfg.interval_unit}_left": 0,
                            "limit": pbt_utils.FINISHED_MARKER}
            # here rather than in _pbt_iteration: a subclass that replaces it sets the same attributes
            if int(self.restart_flag[0]) == 1 or self._transfer_in_process:
                self.restart_time = time.time()
                self._save_restart_record()
            if decision is not None:
                self._append_decision(self.pbt_it, frame, objective, decision)
        self._sync_decision(distributed, previous)
        if taken_over is not None:
            raise RuntimeError(taken_over)

    def _sync_decision(self, distributed, previous):
        """Every rank, right after rank 0's PBT iteration: act on its decision. Rank 0 broadcasts
        [action, n, values...] (float64, 2 + one slot per mutable param), the iteration's one
        broadcast. A restart (`restart_flag`) happens now, before this epoch's update: the other
        ranks exit, rank 0 re-execs. An in-process self-mutation (`_apply_self_mutation`) or
        transfer carries the n new values in sorted key order (rank 0's new params may come in a
        yaml's order; a value a donor doesn't have is the member's own); every rank applies those
        that differ from `previous`, its values before the decision, which every rank shares: a
        self-mutation now (`_apply_in_process`), a transfer at the start of the next epoch
        (`before_steps`), with its restart payload, which rank 0 broadcasts next.
        """
        keys = sorted(previous)
        message = torch.zeros(2 + len(keys), dtype=torch.float64, device=self.device)
        rank0 = self.distributed_args.rank == 0
        if rank0:
            if int(self.restart_flag[0]) == 1:
                message[0] = _RESTART
            elif self._mutate_in_process or self._transfer_in_process:
                new = self.new_params if self._transfer_in_process else self.pbt_params
                message[0] = _TRANSFER_IN_PROCESS if self._transfer_in_process else _MUTATE_IN_PROCESS
                message[1] = len(keys)
                message[2:] = torch.tensor([float(new.get(key, previous[key])) for key in keys], dtype=torch.float64)
            self._mutate_in_process = self._transfer_in_process = False
        if distributed:
            dist.broadcast(message, src=0)
        action, count = int(message[0]), int(message[1])
        if action == _RESTART:
            if not rank0:
                _flush_stdio()
                os._exit(0)
            self._restart_with_new_params(self.new_params, self.restart_from_checkpoint)
        elif action in (_MUTATE_IN_PROCESS, _TRANSFER_IN_PROCESS):
            values = dict(zip(keys, message[2:2 + count].tolist()))
            changed = {key: value for key, value in values.items() if value != previous[key]}
            if action == _MUTATE_IN_PROCESS:
                self._apply_in_process(changed)
                return
            # the payload a restarted process would parse (JSON types)
            payload = [json.loads(json.dumps(self._restart_payload(self.new_params, self.restart_from_checkpoint),
                                             default=pbt_utils.json_default)) if rank0 else None]
            if distributed:
                dist.broadcast_object_list(payload, src=0, device=self.device)
            self._pending_transfer = changed, payload[0]

    def before_steps(self):
        """Every rank, at the start of each epoch, before its rollout: load the in-process transfer
        rank 0 decided at the last iteration (`_load_transfer`). Not at the decision: that epoch's
        update would then train the donor's weights on this member's rollout (and its adaptive LR
        would fall on the KL between the two policies)."""
        self._before_steps_called = True
        if getattr(self, "_pending_transfer", None) is not None:
            changed, payload = self._pending_transfer
            self._pending_transfer = None
            self._load_transfer(changed, payload)

    def _load_transfer(self, changed, payload):
        """Every rank, at the start of the epoch after an in-process transfer was decided: load the
        member's restart.pth into the running learner as the restarted process would restore it,
        and start the member over from it.

        The changed values are set first (`algo.set_param`), so the restore compares the new
        values with the pre-mutation ones restart.pth stores (see `_pbt_iteration`). Every rank
        loads the whole state itself (weights, normalizers, optimizer, scaler, central value net):
        ranks given the weights alone would step apart. Compiled models load through the modules
        they wrap (`_uncompiled`). The member keeps its epoch, frame and best reward (it writes
        into its own run directory), and `restart_info()` returns the transfer's payload from
        then on, already in the env's `set_env_state`. What belonged to the old policy starts
        over: the envs, its episodes, the RNN states, the objective and grace windows.
        """
        algo = self.algo
        for key in sorted(changed):
            algo.set_param(key[len(_CONFIG_PREFIX):], changed[key])
        # rank 0 keeps the values its decision drew (see _apply_in_process)
        new_params = dict(self.pbt_params, **(self.new_params if self.distributed_args.rank == 0 else changed))
        best = algo.last_mean_rewards
        pbt_utils._set_restart_info(payload)
        state = torch_ext.load_checkpoint(payload["checkpoint"])
        with _uncompiled(algo):
            algo.set_full_state_weights(state, set_epoch=False)
        algo.last_mean_rewards = best
        algo.reset_envs()
        envs = algo.num_agents * algo.num_actors
        algo.init_current_rewards(envs, (envs, algo.value_size))
        algo.clear_stats(clean_rewards=False)
        if algo.has_central_value and algo.central_value_net.is_rnn:
            critic = algo.central_value_net
            critic.rnn_states = [s.to(critic.ppo_device) for s in critic.model.get_default_rnn_state()]
        self.restart_count += 1
        self._set_pbt_params(new_params)
        self._reset_objective_window()
        self.initial_frame = self._progress(algo.frame)
        if self.distributed_args.rank == 0:
            print(f"Policy {self.cfg.policy_idx}: loaded member {payload['source_policy']}'s checkpoint in process "
                  f"(restart {self.restart_count}); the grace window restarts at {self.initial_frame} "
                  f"{self.cfg.interval_unit}")

    def _apply_in_process(self, changed):
        """Every rank: set the changed `agent.params.config` values of an in-process self-mutation
        on the learner, in key order, with `algo.set_param` (rank-symmetric: every rank changes
        the same values), take them as this member's params, and start the objective window
        over: the next iteration ranks the member on episodes played with them. Rank 0 keeps the
        values its `_pbt_iteration` took: the broadcast carries floats, and rank 0's yamls and
        restart commands need an integer param such as mini_epochs as an int."""
        for key in sorted(changed):
            self.algo.set_param(key[len(_CONFIG_PREFIX):], changed[key])
        if self.distributed_args.rank != 0:
            self._set_pbt_params(dict(self.pbt_params, **changed))
        self._reset_objective_window()

    def _draw_rng(self):
        """The generator of PBT's draws, created at the first iteration from the config seed, the
        member, the restart count and the frame (or epoch, `_progress`) this process started at
        (`pbt_utils.draw_rng`)."""
        if getattr(self, "_rng", None) is None:
            self._rng = pbt_utils.draw_rng(getattr(self, "_seed", None), self.cfg.policy_idx,
                                           getattr(self, "restart_count", 0), getattr(self, "initial_frame", 0))
        return self._rng

    def _wait_for_objective(self, iteration, pooled):
        """The iteration due at boundary `iteration` waits: a rank's window is not full yet, or a
        rank has no finite objective since the last iteration (its episodes are still running,
        or their objective is NaN or missing). `after_steps` retries it at every step. Nothing
        is saved meanwhile, so a healthy member's workspace iterations skip the boundaries it
        waits through: with synchronized time limits (every env resets at a (re)start) episodes
        end in bursts, and an interval shorter than an episode often passes without one ending.

        Rank 0 renews the owner record (`_heartbeat`; it raises when another launch took the
        member over), warns, and appends a `stale` line to decisions.jsonl, once per boundary.
        Every rank raises after `max_consecutive_errors` boundaries in a row at which a rank had
        failed (no finite objective since the last iteration or the process start, although every
        env of the rank ended an episode, and at least a window of them): a diverged member, also
        one that is NaN from its first step (a restart or resume from a diverged checkpoint), would
        otherwise train on without ever saving or being replaced. Boundaries that only wait for
        episodes to end don't count. Pooled values: every rank decides alike.
        """
        if iteration == self._wait_iteration:
            return
        self._wait_iteration = iteration
        cfg, name = self.cfg, self._objective_name()
        self._failed_boundaries = self._failed_boundaries + 1 if pooled.failed_ranks else 0
        counts = (f"{pooled.valid} finite, {pooled.nonfinite} non-finite and {pooled.unread} unreadable episodes "
                  f"since iteration {self.pbt_it}")
        if self._failed_boundaries >= cfg.max_consecutive_errors:
            raise RuntimeError(
                f"PBT objective '{name}': {self._failed_boundaries} consecutive iteration boundaries at "
                f"which {pooled.failed_ranks} of {pooled.ranks} ranks had no finite objective although every env "
                f"had ended an episode (max_consecutive_errors): a diverged policy, or infos that no longer carry "
                f"the objective; {counts}; last lookup error: {self._last_objective_error}")
        if self.distributed_args.rank != 0:
            return
        taken_over = self._heartbeat(iteration)
        if taken_over is not None:
            raise RuntimeError(taken_over)
        self._append_decision(iteration, self.algo.frame, None, {
            "outcome": "stale", "episodes": pooled.count, "valid": pooled.valid, "nonfinite": pooled.nonfinite,
            "unread": pooled.unread, "stale_ranks": pooled.stale_ranks})
        if pooled.known:
            ranks = f" on {pooled.stale_ranks} of {pooled.ranks} ranks" if pooled.ranks > 1 else ""
            what = f"a new objective '{name}'{ranks}: {counts}"
        else:
            what = (f"objective '{name}': {pooled.count} finished episodes counted, {self._window_size} "
                    f"per rank needed ({pooled.nonfinite} non-finite and {pooled.unread} unreadable dropped)")
        stops = (f"; the member stops at {cfg.max_consecutive_errors} boundaries in a row at which a rank had no "
                 f"finite objective although every env had ended an episode ({self._failed_boundaries} so far)"
                 if self._failed_boundaries else "")
        print(f"Policy {cfg.policy_idx}: WARNING PBT iteration {iteration} waits for {what}; "
              f"last lookup error: {self._last_objective_error}{stops}")

    def _check_member_dir(self, frame):
        """Rank 0, first step of the process: make the member directory this run's own.

        Workspace files are named by iteration only, so peers read whatever this directory holds
        as this member's current state, and may copy its weights. A fresh launch (not a PBT
        restart, frame 0 after the restore) over numbered checkpoints of an earlier run raises
        unless `pbt.reuse_workspace` is set; the step runs no collective, and torchrun then stops
        the other ranks. Otherwise this member's files above its current iteration are deleted: a
        relaunch from an older checkpoint, or reused leftovers. So is a restart.yaml of a decision
        above the current iteration, unless this process is a PBT restart (the record is then its
        own). A fresh launch under `reuse_workspace` also removes the workspace's finished.yaml
        (`pbt.stop_with_first`) when it was written before this launch: the earlier run's end.
        """
        directory = getattr(self, "curr_policy_dir", None)
        if directory is None or not os.path.isdir(directory):
            return
        saved = pbt_utils.saved_iterations(directory)
        restart = getattr(self, "restart", None)
        fresh = restart is None and frame == 0
        reuse = getattr(self.cfg, "reuse_workspace", False)
        marker = pbt_utils.read_record(self.ws_dir, pbt_utils.FINISHED_MARKER) if fresh and reuse else None
        written = (marker or {}).get("time")
        if marker is not None and not (isinstance(written, (int, float)) and written >= self._claimed):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(self.ws_dir, pbt_utils.FINISHED_MARKER))
            print(f"Policy {self.cfg.policy_idx}: removed the {pbt_utils.FINISHED_MARKER} an earlier run left in "
                  f"{self.ws_dir}: this launch starts the workspace over (reuse_workspace)")
        if fresh and saved and not reuse:
            raise RuntimeError(
                f"Policy {self.cfg.policy_idx}: {directory} holds PBT checkpoints of an earlier run (iterations "
                f"{saved[0]}..{saved[-1]}), and this is a fresh launch (no checkpoint, not a PBT restart): peers "
                f"would rank them, and copy their weights, as this member's current state. Start the population "
                f"in a new workspace (pbt.workspace or train_dir), delete the old one, or set "
                f"pbt.reuse_workspace: true to start over it (each member then deletes its own numbered files)")
        removed = pbt_utils.remove_iterations_above(directory, self.pbt_it)
        if removed:
            print(f"Policy {self.cfg.policy_idx}: removed iterations {removed[0]}..{removed[-1]} ({len(removed)}) "
                  f"above iteration {self.pbt_it} from {directory}: an earlier run left them, and peers would read "
                  f"them as this member's current state")
        decided = pbt_utils.remove_restart_record_above(directory, self.pbt_it) if restart is None else None
        if decided is not None:
            print(f"Policy {self.cfg.policy_idx}: removed {pbt_utils.RESTART_RECORD} of a restart decided at "
                  f"iteration {decided}, above iteration {self.pbt_it}, from {directory}: this process does not "
                  f"continue that run")

    def _record_restart(self, frame):
        """Rank 0, first step of a process started with a restart payload: log what the restart cost,
        `pbt/restart_seconds` (since the decision) and `pbt/frames_discarded` (what the member had
        played past this process's restored `frame` when it re-exec'd, which this process plays
        again), and append a `restarted` line to decisions.jsonl. A resume has no decision: its
        line has neither number; a retried restart (`member_resume`) has no frames, as what its
        dead process played is unknown."""
        info = self.restart
        decided, reached = info.get("time"), info.get("frame_reached")
        seconds = None if decided is None else time.time() - float(decided)
        frames = None if reached is None else int(reached) - int(frame)
        writer = getattr(self.algo, "writer", None)
        for tag, value in (("pbt/restart_seconds", seconds), ("pbt/frames_discarded", frames)):
            if writer is not None and value is not None:
                writer.add_scalar(tag, value, self._summary_step(frame))
        self._append_decision(self.pbt_it, frame, None, {
            "outcome": "restarted", "kind": info.get("kind"), "source": info.get("source_policy"),
            "attempt": info.get("attempt"), "seconds": seconds, "frames_discarded": frames})

    def _append_decision(self, iteration, frame, objective, decision):
        """Rank 0: append an outcome to the member's decisions.jsonl (`pbt_utils.append_decision`):
        {"schema", "time", "iteration", "frame", "objective", "outcome", ...} plus the outcome's
        numbers. Outcomes: those of the rule ("not_worst", "too_few_reported", "kept" by the
        interval's best objective, "transfer", "self_mutation", with the worst and top groups, the
        candidate, its lead and the thresholds; a transfer loaded in process says `in_process:
        true`), "grace", "ending", "no_op", "in_process", "error",
        "stale" (an iteration waiting for its objective), "restarted" (`_record_restart`) and
        "forced" (`pbt.force_outcome` names a donor that has not reported). Under
        `pbt.force_outcome` every line carries `forced`. A line that can't be written only warns."""
        line = dict({"schema": pbt_utils.SCHEMA, "time": time.time(), "iteration": int(iteration),
                     "frame": int(frame), "objective": objective}, **self._forced_stamp(), **decision)
        try:
            pbt_utils.append_decision(self.curr_policy_dir, line)
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not append to {pbt_utils.DECISIONS}: {exc!r}")

    def _pbt_iteration(self, score, best_in_iteration, frame):
        """One PBT iteration on rank 0: save this member's checkpoint, load the population's, log,
        record a new population best, and apply the replacement rule (or take the source
        `pbt.force_outcome` names).

        No decision inside the grace windows (`start_after` since the process started or a
        self-mutation was applied, `initial_delay` since the start of training, in frames or
        epochs: `_progress`), nor when the run ends before a restart could pay off (`_to_end`) or
        the population has finished (`pbt.stop_with_first`); the member's yaml says
        `replaceable: false` then, so peers keep it out of their worst group. The new values are
        clamped into `mutation_bounds`. A self-mutation that changes no value keeps training; one
        with changes goes to `_apply_self_mutation` first, and restarts unless that applied it.
        Under `pbt.transfer: in_process` a transfer whose changed values the running learner
        takes (`_not_settable`) loads at the start of the next epoch instead of restarting
        (`before_steps`), from the same restart.pth and restart.yaml. A restart (or such a
        transfer) marks this iteration's yaml not replaceable once restart.pth is written: peers
        read it until the member saves again. Peers that run other code or another
        population setup (`_foreign_members`), and stale ones (`pbt.max_member_lag`,
        `pbt_utils.member_status`), are left out.

        Returns the iteration's outcome, which `after_steps` appends to decisions.jsonl (and records
        a restart in restart.yaml): {"outcome": ...} with the rule's numbers, see `_append_decision`.
        """
        cfg, unit = self.cfg, self.cfg.interval_unit
        progress = self._progress(frame)
        left = (self.pbt_it + 1) * cfg.interval_steps - progress
        print(f"Policy {cfg.policy_idx}, {unit}_left {left}, PBT it {self.pbt_it}, objective {score:.6g}")
        in_grace = progress - self.initial_frame < cfg.start_after or progress < cfg.initial_delay
        to_end = self._to_end(frame)
        if cfg.stop_with_first and self._population_finished() is not None:
            to_end = (0, pbt_utils.FINISHED_MARKER)
        ending = to_end is not None and to_end[0] < max(cfg.start_after, cfg.interval_steps)
        try:
            pbt_utils.save_pbt_checkpoint(self.curr_policy_dir, score, self.pbt_it, self.algo, self.pbt_params,
                                          replaceable=not (in_grace or ending), checkpoint_arg=cfg.checkpoint_arg,
                                          extra=self._forced_stamp(), restart_drop_args=cfg.restart_drop_args)
            ckpts = pbt_utils.load_pbt_ckpts(self.ws_dir, cfg.policy_idx, cfg.num_policies, self.pbt_it)
            pbt_utils.cleanup(ckpts, self.curr_policy_dir)
            # what this member's yamls said where members 2+ iterations behind are (see member_status)
            own_objectives = pbt_utils.load_reader_objectives(self.curr_policy_dir, ckpts, self.pbt_it)
            activity = {p: pbt_utils.member_activity(os.path.join(self.ws_dir, f"{p:03d}")) for p in ckpts}
        except Exception as exc:
            self._consecutive_errors += 1
            print(f"Policy {cfg.policy_idx}: exception {exc!r} in PBT checkpoint save/load "
                  f"({self._consecutive_errors}/{cfg.max_consecutive_errors})")
            if self._consecutive_errors >= cfg.max_consecutive_errors:
                raise RuntimeError(
                    f"PBT checkpoint save/load failed {self._consecutive_errors} times in a row") from exc
            return {"outcome": "error", "error": repr(exc)}
        self._consecutive_errors = 0

        lag = self._member_lag()
        status = pbt_utils.member_status(ckpts, self.pbt_it, reader_objectives=own_objectives, activity=activity,
                                         lag=lag, foreign=self._foreign_members(ckpts))
        sumry = {i: None if c is None else {k: v for k, v in c.items() if k != "params"} for i, c in ckpts.items()}
        self.printer.print_ckpt_summary(sumry, status=status)
        unreported = sorted(p for p, s in status.items() if s == pbt_utils.NOT_REPORTED)
        if unreported:
            print(f"Policy {cfg.policy_idx}: WARNING members {unreported} have not reported (no checkpoint at or "
                  f"below iteration {self.pbt_it} in {self.ws_dir}), so they are not ranked: are all "
                  f"{cfg.num_policies} members (num_policies) running?")
        stale = sorted(p for p, s in status.items() if s == pbt_utils.STALE)
        if stale:
            print(f"Policy {cfg.policy_idx}: WARNING members {stale} have shown no sign of life (a heartbeat in "
                  f"{pbt_utils.OWNER_RECORD}, a workspace yaml) for {lag:.0f} s, max_member_lag × this member's "
                  f"median iteration time: they are left out of the ranking and the donors")

        left_out = (pbt_utils.FOREIGN, pbt_utils.STALE)
        objectives = {p: (c["true_objective"] if c and status[p] not in left_out else None) for p, c in ckpts.items()}
        initialized = {p: o for p, o in objectives.items() if o is not None and o > _UNINITIALIZED_VALUE}
        # equal objectives rank in an order every member shares at this iteration, new at the next
        order = pbt_utils.tie_break_order(cfg.workspace, self.pbt_it, cfg.num_policies)
        self._write_summaries(score, initialized, frame, order=order)
        if not initialized:
            print("No policies initialized; skipping PBT iteration.")
            return {"outcome": "too_few_reported", "reported": 0}

        best_policy = pbt_utils.rank_members(initialized, order)[0]
        if best_policy == cfg.policy_idx:
            try:
                pbt_utils.maybe_save_best(self.ws_dir, cfg.policy_idx, initialized[best_policy], self.pbt_it, frame,
                                          ckpts[best_policy]["checkpoint"], self.pbt_params, extra=self._forced_stamp())
            except Exception as exc:
                print(f"Policy {cfg.policy_idx}: exception {exc!r} while saving the population-best checkpoint")

        if in_grace:
            print(f"Policy {cfg.policy_idx}: within the grace window (at {progress} {unit}, restarted at "
                  f"{self.initial_frame}, start_after {cfg.start_after}, initial_delay {cfg.initial_delay})")
            return {"outcome": "grace"}
        if ending:
            print(f"Policy {cfg.policy_idx}: {to_end[0]} {unit} left after this epoch before {to_end[1]} ends the "
                  f"run, fewer than max(start_after, interval_steps) = {max(cfg.start_after, cfg.interval_steps)}: "
                  f"no replacement this close to the end (a restarted member could not leave its grace window "
                  f"or reach another iteration)")
            return {"outcome": "ending", f"{unit}_left": to_end[0], "limit": to_end[1]}

        details = {}  # the rule's outcome and numbers
        # with pbt.force_outcome either rule returns the source it names (pbt_utils.select_forced)
        if cfg.replace_rule == "dexpbt":
            eligible = {p for p, s in status.items() if s == pbt_utils.REPLACEABLE}
            source, reason = pbt_utils.select_dexpbt(objectives, cfg.policy_idx, score, best_in_iteration, cfg,
                                                     eligible=eligible, order=order, details=details)
        else:
            source, reason = pbt_utils.select_threshold(objectives, cfg.policy_idx, cfg, details=details)
        print(f"Policy {cfg.policy_idx}: {reason}")
        if source is None:
            return details

        source_params = ckpts[source]["params"]
        use_source_params = cfg.replace_rule == "threshold" or random.random() < cfg.leader_params_prob
        base_params = source_params if use_source_params else self.pbt_params
        new_params = clamp_to_bounds(mutate(base_params, cfg.mutation, cfg.mutation_rate, cfg.change_range),
                                     cfg.mutation_bounds)
        # the decision in the learner's pbt_history (its checkpoints' PBT provenance)
        history_entry = {"policy_idx": cfg.policy_idx, "source_policy": source, "frame": int(frame),
                         "objective": float(score), "source_objective": float(objectives[source])}
        if source == cfg.policy_idx:
            changed = {k: v for k, v in new_params.items() if self.pbt_params.get(k) != v}
            if not changed:
                # a restart would cost a start-up and a new grace window for the same values
                print(f"Policy {cfg.policy_idx}: no-op self-mutation: keep training (no value changed)")
                return dict(details, outcome="no_op")
            if self._apply_self_mutation(changed):
                # at the x of this iteration's current values, as a restart logs them
                self._write_params(self.algo.writer, new_params, self._summary_step(frame))
                self.algo.writer.flush()
                self.printer.print_mutation_diff(self.pbt_params, new_params,
                                                 header="Mutated params (applied in process)")
                self._set_pbt_params(new_params)
                # the learner's next checkpoints record it, as restart.pth records a restart (rank 0 saves them)
                history = getattr(self.algo, "pbt_history", None)
                if isinstance(history, list):
                    history.append(dict(history_entry, in_process=True))
                # the grace window starts over; the saves in it say replaceable: false
                self.initial_frame = progress
                print(f"Policy {cfg.policy_idx}: self-mutation applied in process, no restart; the grace window "
                      f"restarts at {self.initial_frame} {unit}")
                return dict(details, outcome="in_process", mutated=changed)
        try:
            # the values this restart mutated from: the child's restore then moves an inherited
            # adaptive LR by this restart's mutation factor only (1 when nothing was mutated)
            config_hparams = {key: float(base_params[_CONFIG_PREFIX + key])
                              for key in ("learning_rate", "entropy_coef") if _CONFIG_PREFIX + key in base_params}
            # a fixed run name keeps nn/<name>.pth across restarts: keep the member's watermark
            fixed_name = (getattr(self.algo, "config", None) or {}).get("full_experiment_name")
            restart_checkpoint = pbt_utils.prepare_restart_checkpoint(
                ckpts[source]["checkpoint"], self.curr_policy_dir, frame,
                # this runs in after_steps, after update_epoch and before the epoch's update and
                # frame increment: the member's frame belongs to the last completed epoch
                epoch=int(self.algo.epoch_num) - 1,
                history_entry=history_entry,
                config_hparams=config_hparams or None,
                last_mean_rewards=float(self.algo.last_mean_rewards) if fixed_name else None,
                # a resume from restart.pth reads this member, not the source
                pbt_state=self._pbt_state(new_params, self.restart_count + 1))
        except Exception as exc:
            # the source checkpoint may have been cleaned up meanwhile: keep training as is
            print(f"Policy {cfg.policy_idx}: exception {exc!r} while preparing the restart checkpoint; "
                  f"continuing with the current weights")
            return dict(details, outcome="error", error=repr(exc))

        # at the x of this iteration's current values: the plot steps to the new ones (as IsaacGymEnvs)
        self._write_params(self.algo.writer, new_params, self._summary_step(frame))
        self.algo.writer.flush()

        self.new_params = new_params
        self.restart_from_checkpoint = restart_checkpoint
        self.restart_source = source
        self.restart_frame = int(frame)
        # values the restored checkpoint would otherwise impose on the new process
        self.restart_mutated = {k: v for k, v in new_params.items() if source_params.get(k) != v}
        if source != cfg.policy_idx and cfg.transfer == "in_process":
            fixed = self._not_settable({k: v for k, v in new_params.items() if self.pbt_params.get(k) != v})
            if fixed:
                print(f"Policy {cfg.policy_idx}: {fixed} can't change in the running process: the transfer restarts "
                      f"the member")
            self._transfer_in_process = not fixed
        if not self._transfer_in_process:
            self.restart_flag[0] = 1
        self.printer.print_mutation_diff(base_params, new_params)
        # peers read this iteration's yaml as the member's state until the member saves again
        try:
            pbt_utils.mark_not_replaceable(self.curr_policy_dir)
        except Exception as exc:
            print(f"Policy {cfg.policy_idx}: WARNING could not mark the newest workspace yaml not replaceable: "
                  f"{exc!r}")
        decision = dict(details, outcome=pbt_utils.restart_kind(cfg.policy_idx, source), source=source,
                        mutated=self.restart_mutated)
        return dict(decision, in_process=True) if self._transfer_in_process else decision

    def _foreign_members(self, ckpts):
        """The peers whose newest yaml (in `ckpts`) says they run other code (`code_fingerprint`) or
        another population setup (`settings`) than this member: left out of the ranking, the
        donors and the gate's count. A different rl_games version, git commit or precision only
        warns. Each difference warns once per peer and value; a field missing on either side
        (older yamls) is unknown (`pbt_utils.identity_mismatches`)."""
        foreign = set()
        for policy, ckpt in ckpts.items():
            if policy == self.cfg.policy_idx or not ckpt:
                continue
            for key, (theirs, mine) in pbt_utils.identity_mismatches(ckpt, self.population).items():
                left_out = key in pbt_utils.FOREIGN_KEYS
                if left_out:
                    foreign.add(policy)
                if (policy, key, repr(theirs)) not in self._identity_warnings:
                    self._identity_warnings.add((policy, key, repr(theirs)))
                    left_out_note = ": it is left out of the ranking and the donors" if left_out else ""
                    print(f"Policy {self.cfg.policy_idx}: WARNING member {policy} runs with {key} {theirs!r}, "
                          f"this member with {mine!r}{left_out_note}")
        return foreign

    def _apply_self_mutation(self, changed):
        """Hook: apply a self-mutation in the running process instead of restarting; return True
        when it is applied.

        Called on rank 0 inside `_pbt_iteration` for a self-mutation (the member keeps its own
        weights) that changes some values, before anything is written for a restart. `changed`
        maps each changed flattened param to its new value. On True the observer takes the new
        values as `pbt_params` (`_set_pbt_params`), logs them, appends the entry a restart puts
        into restart.pth to the learner's `pbt_history` with `in_process: True` (on rank 0, which
        saves the learner's checkpoints), restarts the member's grace window (`initial_frame`; its
        saves say `replaceable: false` until `start_after` frames or epochs have passed), records
        the outcome `in_process` in decisions.jsonl and keeps training.

        The base applies it under `pbt.self_mutation: in_process` when every changed key is an
        `agent.params.config` value the learner's `set_param` takes (`A2CBase.SET_PARAMS`):
        `after_steps` then broadcasts the new values, and every rank sets them before this
        epoch's update (`_apply_in_process`). Env and extra_params values, which apply when the
        env is built, and central_value_config values restart the member, as does the default
        'restart'. A subclass that applies the values itself returns True without calling this
        method; the base then broadcasts none, and the subclass must apply them on every rank
        (rank 0 broadcasts them, e.g. after `super().after_steps()`) and should call
        `_reset_objective_window()` on every rank: the next iteration then ranks the member by
        episodes played with the new values.
        """
        if self.cfg.self_mutation != "in_process":
            return False
        fixed = self._not_settable(changed)
        if fixed:
            print(f"Policy {self.cfg.policy_idx}: {fixed} can't change in the running process: the self-mutation "
                  f"restarts the member")
            return False
        self._mutate_in_process = True
        return True

    def _not_settable(self, changed):
        """The keys of `changed` the running learner can't take: all but this member's
        `agent.params.config` values its `set_param` sets (`A2CBase.SET_PARAMS`). Env and
        extra_params values apply when the env is built, central_value_config values when the
        critic is; a donor's value this member doesn't mutate has no slot in the decision
        broadcast."""
        settable = getattr(self.algo, "SET_PARAMS", ())
        return sorted(key for key in changed if key not in self.pbt_params or not key.startswith(_CONFIG_PREFIX)
                      or key[len(_CONFIG_PREFIX):] not in settable)

    def _forced_stamp(self):
        """`{"forced": <pbt.force_outcome>}` for the member's workspace yamls, best records,
        restart.yaml and decisions.jsonl lines when it is set, else {}: a forced member's records
        must not pass for a population's."""
        return {"forced": self.cfg.force_outcome} if self.cfg.force_outcome else {}

    def _reset_objective_window(self):
        """Start this rank's objective over: drop its window of episodes, its score and the
        iteration's best objective. The next iteration waits until every rank's window is full
        again (see `after_steps`). For a subclass that applies a self-mutation in process
        (`_apply_self_mutation`), on every rank."""
        self._window.clear()
        self._window_sum, self._window_count = 0.0, 0
        self.score = _UNINITIALIZED_VALUE
        self.best_in_iteration = None

    def _epoch_frames(self):
        """The frames the learner adds to `algo.frame` after this epoch's `after_steps`:
        `curr_frames`, on every rank with multi_gpu."""
        algo = self.algo
        curr_frames = int(getattr(algo, "curr_frames", 0) or 0)
        if getattr(algo, "multi_gpu", False):
            curr_frames *= int(getattr(algo, "world_size", 1) or 1)
        return curr_frames

    def _progress(self, frame):
        """The member's progress at this `after_steps` in `pbt.interval_unit`, which the cadence,
        the grace windows (`initial_frame` holds the start in it) and the end-of-run margin count:
        `frame`, the frames of the completed epochs, or the completed epochs, `epoch_num - 1`
        (after_steps runs in epoch `epoch_num`, before its update). A restart keeps both:
        restart.pth holds the last completed epoch and its frame."""
        return int(self.algo.epoch_num) - 1 if self.cfg.interval_unit == "epochs" else int(frame)

    def _to_end(self, frame):
        """`_frames_to_end` in `pbt.interval_unit`: with epochs, the frames left in epochs, rounded
        up (the epoch that passes max_frames is played in full)."""
        to_end = self._frames_to_end(frame)
        if to_end is None or self.cfg.interval_unit != "epochs":
            return to_end
        return -(-to_end[0] // self._epoch_frames()), to_end[1]

    def _frames_to_end(self, frame):
        """(frames, limit): the frames this member trains after the current epoch before `limit`
        (max_epochs or max_frames, the nearer one) ends the run; None without either limit.

        `after_steps` runs before the epoch's frames are added to `algo.frame`, and the run ends
        after the epoch that reaches a limit: in the last epoch this is <= 0.
        """
        algo = self.algo
        epoch_frames = self._epoch_frames()
        left = []
        max_epochs = getattr(algo, "max_epochs", -1)
        if max_epochs is not None and max_epochs != -1:
            left.append(((int(max_epochs) - int(algo.epoch_num)) * epoch_frames, "max_epochs"))
        max_frames = getattr(algo, "max_frames", -1)
        if max_frames is not None and max_frames != -1:
            left.append((int(max_frames) - int(frame) - epoch_frames, "max_frames"))
        return min(left) if left else None

    def after_train(self, reason):
        """Rank 0, when training ends normally (`reason`, one of a2c_common.EXIT_REASONS): write the
        member's exit record (exit.yaml: schema, reason, frame, epoch, time, launch_id). Peers then
        see the member as finished: ranked and a donor, never replaced. Not written once another
        launch has taken the member over (its owner record names that launch). An observer chain
        must forward the hook (`MultiObserver` here does), or peers drop a member that ended
        normally as stale once its heartbeat is old."""
        if self.distributed_args.rank != 0:
            return
        try:
            owner = pbt_utils.read_record(self.curr_policy_dir, pbt_utils.OWNER_RECORD)
            if owner is not None and owner.get("launch_id") != self.launch_id:
                return
            pbt_utils.write_record(self.curr_policy_dir, pbt_utils.EXIT_RECORD, {
                "schema": pbt_utils.SCHEMA, "reason": reason, "frame": int(self.algo.frame),
                "epoch": int(self.algo.epoch_num), "time": time.time(), "launch_id": self.launch_id})
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not write {pbt_utils.EXIT_RECORD}: {exc!r}")

    def _watch_population_end(self, frame):
        """`pbt.stop_with_first`, rank 0 at every step. The learner's stop_fn, which rank 0 calls
        after each epoch's update (training then saves the custom-stop checkpoint and ends on every
        rank), also returns True once the workspace holds this population's finished.yaml
        (`_population_finished`); a stop_fn set before train(), or since, is called first. In this
        member's last epoch by max_epochs or max_frames, when train() calls no stop_fn, it writes
        the marker unless a member of the population has; a write that fails only warns."""
        algo = self.algo
        if self._stop_fn is None or algo.stop_fn is not self._stop_fn:
            user_stop_fn = algo.stop_fn

            def stop_fn(learner):
                if user_stop_fn is not None and user_stop_fn(learner):
                    return True
                return self._population_finished() is not None

            self._stop_fn = algo.stop_fn = stop_fn
        to_end = self._frames_to_end(frame)
        if to_end is None or to_end[0] > 0 or self._population_finished() is not None:
            return
        try:
            pbt_utils.write_record(self.ws_dir, pbt_utils.FINISHED_MARKER, dict(
                {"schema": pbt_utils.SCHEMA, "policy_idx": self.cfg.policy_idx, "reason": to_end[1],
                 "frame": int(frame) + self._epoch_frames(), "epoch": int(algo.epoch_num)},
                **self._run_targets(), time=time.time()))
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not write {pbt_utils.FINISHED_MARKER}: {exc!r}")
            return
        print(f"Policy {self.cfg.policy_idx}: {to_end[1]} ends the run with this epoch, the first of the population: "
              f"wrote {pbt_utils.FINISHED_MARKER}, the other members stop at the end of their epochs")

    def _run_targets(self):
        """What makes a finished.yaml this population's: max_epochs, max_frames and the settings hash."""
        algo = self.algo
        return {"max_epochs": int(getattr(algo, "max_epochs", -1)), "max_frames": int(getattr(algo, "max_frames", -1)),
                "settings": self.population["settings"]}

    def _population_finished(self):
        """The workspace's finished.yaml (`pbt.stop_with_first`) when a member of this population wrote
        it (the same `_run_targets`), else None. A marker of another run is ignored, with one warning.
        A filesystem error only warns (None): the learner's stop_fn asks at every epoch."""
        try:
            marker = pbt_utils.read_record(self.ws_dir, pbt_utils.FINISHED_MARKER)
        except OSError as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not read {pbt_utils.FINISHED_MARKER}: {exc!r}")
            return None
        if marker is None:
            return None
        own = self._run_targets()
        differs = [key for key, value in own.items() if marker.get(key) != value]
        if not differs:
            return marker
        if marker.get("time") != self._ignored_marker:
            self._ignored_marker = marker.get("time")
            theirs, mine = ({key: values[key] for key in differs} for values in (marker, own))
            print(f"Policy {self.cfg.policy_idx}: WARNING ignoring {pbt_utils.FINISHED_MARKER} of another run in "
                  f"{self.ws_dir}: it has {theirs}, this member {mine}")
        return None

    def _summary_step(self, frame=None):
        """x of the pbt/* scalars: the x of this epoch's `rewards/step` point (env steps).

        `after_steps` runs before the learner adds the epoch's frames (`curr_frames`, on every
        rank with multi_gpu) to `algo.frame`, which counts agent steps. `frame` defaults to
        `algo.frame`; the cadence, the grace windows and the workspace keep counting it.
        """
        algo = self.algo
        frame = int(algo.frame if frame is None else frame)
        return (frame + self._epoch_frames()) // max(1, int(getattr(algo, "num_agents", 1) or 1))

    def _write_summaries(self, score, initialized, frame, *, order=None):
        """The iteration's pbt/* scalars at `_summary_step(frame)`: objective, population best,
        rank, restart count and the current value of every numeric mutable param. Members with
        equal objectives rank in `order` (`pbt_utils.tie_break_order`); without it they share the
        best of their ranks."""
        writer = getattr(self.algo, "writer", None)
        if writer is None:
            return
        step = self._summary_step(frame)
        if score > _UNINITIALIZED_VALUE:
            writer.add_scalar("pbt/objective", score, step)
        if initialized:
            writer.add_scalar("pbt/00_best_objective", max(initialized.values()), step)
            if self.cfg.policy_idx in initialized:
                if order is None:
                    rank = 1 + sorted(initialized.values(), reverse=True).index(initialized[self.cfg.policy_idx])
                else:
                    rank = 1 + pbt_utils.rank_members(initialized, order).index(self.cfg.policy_idx)
                writer.add_scalar("pbt/rank", rank, step)
        writer.add_scalar("pbt/restart_count", self.restart_count, step)
        self._write_params(writer, self.pbt_params, step)
        writer.flush()

    @staticmethod
    def _write_params(writer, params, step):
        """`pbt/<param>` for every numeric value in `params` (numpy and torch scalars as numbers)."""
        for param, value in params.items():
            value = pbt_utils.plain_number(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"pbt/{param}", value, step)

    def _get_launcher(self):
        """The executable used to re-exec training: `pbt.launcher` from config, else sys.executable."""
        return self.cfg.launcher or sys.executable

    def _restart_script_args(self, new_params, restart_from_checkpoint):
        """The restart's script arguments (script path first) before the seed is replaced: the
        launch argv, or on the legacy path `sys.argv` and the `args_cli` flags it doesn't have yet,
        with the checkpoint argument and the new params, without `restart_drop_args`."""
        cfg = self.cfg
        # an observer built without __init__ (DexLab's tests) has a cfg without the newer keys
        drop_args = getattr(cfg, "restart_drop_args", None)
        if self.launch_argv is not None:
            return pbt_utils.build_launch_restart_args(
                self.launch_argv, new_params, restart_from_checkpoint, cfg.checkpoint_arg, drop_args=drop_args)
        modified_args = build_restart_args(sys.argv, new_params, restart_from_checkpoint, self.wandb_args,
                                           self.rendering_args, cfg.checkpoint_arg, drop_args=drop_args)
        env_args = pbt_utils.missing_flags(self.env_args.get_args_list(), sys.argv[1:])
        script_args = [modified_args[0]] + env_args + modified_args[1:]
        if self.distributed_args.distributed:
            script_args += pbt_utils.missing_flags(["--distributed"], sys.argv[1:])
        return script_args

    def _restart_seed(self, script_args):
        """The seed the restart command passes as `seed_arg` in place of the one in `script_args`
        (`_restart_script_args`): derived from it, the member and the restart count
        (`pbt_utils.derive_seed`; -1 stays -1). On the legacy path (no launch_argv) a command
        without one gets a seed derived from the config seed (params['seed']) instead, never that
        seed itself: train scripts offset it. None when the command keeps its seed:
        `reseed_on_restart` off, no seed to derive from (a config seed of None or -1 neither), or a
        value that is not an integer."""
        cfg = self.cfg
        if not (cfg.reseed_on_restart and cfg.seed_arg):
            return None
        base = pbt_utils.get_cli_arg(script_args, cfg.seed_arg)
        config_seed = getattr(self, "_seed", None)
        if base is None and self.launch_argv is None and not pbt_utils._random_per_launch(config_seed):
            base = config_seed
        if base is None:
            return None
        try:
            return pbt_utils.derive_seed(int(base), cfg.policy_idx, self.restart_count + 1)
        except (TypeError, ValueError):
            return None

    def _restart_command(self, new_params, restart_from_checkpoint):
        """Full re-exec command: launcher, optional torch.distributed.run prefix, script args.

        The seed argument is replaced by `_restart_seed`'s, the script path is made absolute (see
        `pbt_utils.absolute_script_path`), and inside a Hydra job that changed into its run dir
        the command pins `hydra.run.dir` to it.
        """
        cfg = self.cfg
        new_params = {param: pbt_utils.plain_number(value) for param, value in new_params.items()}
        script_args = self._restart_script_args(new_params, restart_from_checkpoint)
        seed = self._restart_seed(script_args)
        if seed is not None:
            script_args = [script_args[0]] + pbt_utils.remove_cli_arg(script_args[1:], cfg.seed_arg)
            script_args.append(f"{cfg.seed_arg}={seed}")

        script_args = pbt_utils.absolute_script_path(script_args, self._launch_cwd())
        hydra_run_dir = getattr(self, "hydra_run_dir", None)
        if hydra_run_dir is not None:
            script_args = pbt_utils.pin_hydra_run_dir(script_args, hydra_run_dir)

        command = [self._get_launcher()]
        if self.distributed_args.distributed:
            command.extend(self.distributed_args.get_args_list())
        return command + script_args

    def _launch_cwd(self):
        """The directory restarts run from: `launch_cwd` from init, else (an observer built
        without `__init__`, as some downstream tests do) the current directory."""
        return getattr(self, "launch_cwd", None) or os.getcwd()

    def _finish_wandb(self, run, command):
        """Finish the open wandb run before the re-exec, on the calling (main) thread: with
        console="redirect" finish() restores a signal handler, which only the main thread can do.

        After a clean finish the child resumes the run by id through the inherited wandb service.
        A finish that raises, or that has not returned after `wandb_finish_timeout` seconds
        (wandb unreachable), must not stop the restart: the child then logs to a new run
        (`_new_wandb_run`). After a timeout a daemon watchdog thread does that and execs `command`
        itself while finish() still blocks, and ends the process if that exec fails. Returns when
        the calling thread is to exec `command`, never once the watchdog has taken over.
        """
        run_id = run.id
        # setdefault only affects the restarted child process
        os.environ.setdefault("WANDB_RUN_ID", run_id)  # continue with the same run id
        os.environ.setdefault("WANDB_RESUME", "allow")  # allow wandb to resume
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")  # give wandb init more time to be fault tolerant
        timeout = getattr(self.cfg, "wandb_finish_timeout", PbtCfg.wandb_finish_timeout)
        finished = threading.Event()
        exec_claim = threading.Lock()  # taken, never released, by the one thread that execs

        def watchdog():
            if finished.wait(timeout) or not exec_claim.acquire(blocking=False):
                return
            try:
                self._new_wandb_run(run_id, f"did not finish within {timeout:g} s")
                self._exec_restart(command)
            except BaseException:
                try:
                    traceback.print_exc()
                except Exception:
                    pass
            # execv returned, so it failed, and finish() may never return: end the member
            _flush_stdio()
            os._exit(1)

        thread = threading.Thread(target=watchdog, name="pbt-wandb-watchdog", daemon=True)
        thread.start()
        failure = None
        try:
            run.finish()
        except Exception as exc:
            failure = f"finish raised {exc!r}"
        finally:
            finished.set()  # the watchdog stands down, also when finish() is interrupted
        if not exec_claim.acquire(blocking=False):
            thread.join()  # the watchdog is replacing (or ending) the process
            raise RuntimeError("PBT restart: the wandb watchdog thread neither restarted nor ended the process")
        if failure is not None:
            self._new_wandb_run(run_id, failure)

    def _new_wandb_run(self, run_id, failure):
        """After a wandb finish that failed or timed out: the child gets its own wandb service and
        a new run `<run id>-r<restart count>`, as the old service may still be writing this run."""
        # replace the suffix of an earlier failed restart instead of appending to it
        new_id = re.sub(r"-r\d+$", "", run_id) + f"-r{self.restart_count + 1}"
        # an inherited service makes the child's wandb.init hang, or fail with "run ID is in use"
        os.environ.pop("WANDB_SERVICE", None)
        os.environ["WANDB_RUN_ID"] = new_id
        os.environ["WANDB_RESUME"] = "allow"
        print(f"Policy {self.cfg.policy_idx}: WARNING wandb run {run_id} {failure}; the restarted process "
              f"logs to a new run {new_id}", flush=True)

    def _restart_payload(self, new_params, restart_from_checkpoint):
        """The restart payload the child reads as `pbt_utils.restart_info()` (see there), without the
        `frame_reached` the re-exec adds: the decision's attributes set by `_pbt_iteration`
        (`restart_source`, `restart_frame`, `restart_mutated`) and `after_steps` (`restart_time`),
        defaulting to a restart of this member from `restart_from_checkpoint` with `new_params`
        mutated at the current frame and time."""
        source = getattr(self, "restart_source", self.cfg.policy_idx)
        return {
            "schema": pbt_utils.SCHEMA,
            "kind": pbt_utils.restart_kind(self.cfg.policy_idx, source),
            "policy_idx": self.cfg.policy_idx,
            "source_policy": source,
            "restart_count": self.restart_count + 1,
            "iteration": self.pbt_it,
            "frame": getattr(self, "restart_frame", int(self.algo.frame)),
            "checkpoint": restart_from_checkpoint,
            "mutated": getattr(self, "restart_mutated", new_params),
            "experiment_name": getattr(self.algo, "experiment_name", None),
            "time": getattr(self, "restart_time", None) or time.time(),
        }

    def _restart_record(self):
        """The member's restart.yaml for the decision `_pbt_iteration` just took: the restart
        payload, the last completed `epoch` (restart.pth's: the decision runs in after_steps,
        before the epoch's update), every new param value (`params`), the `seed` the restart
        command passes as `seed_arg` (`_restart_seed`; None when it keeps its own), with which
        `pbt_utils.member_resume` retries the restart, the train script's `checkpoint_arg` and
        `restart_drop_args` (a subclass that saves with its own call records neither in its
        workspace yamls), and `forced` under `pbt.force_outcome`. Plain JSON types, as the child
        reads the payload."""
        new_params = {param: pbt_utils.plain_number(value) for param, value in self.new_params.items()}
        checkpoint = self.restart_from_checkpoint
        record = dict(self._restart_payload(new_params, checkpoint),
                      epoch=int(self.algo.epoch_num) - 1, params=new_params,
                      seed=self._restart_seed(self._restart_script_args(new_params, checkpoint)),
                      seed_arg=self.cfg.seed_arg, checkpoint_arg=self.cfg.checkpoint_arg,
                      restart_drop_args=dict(getattr(self.cfg, "restart_drop_args", None) or {}),
                      **self._forced_stamp())
        return json.loads(json.dumps(record, default=pbt_utils.json_default))

    def _save_restart_record(self):
        """Rank 0, at a restart decision (restart.pth written, the flag set): write the member's
        restart.yaml (`pbt_utils.write_restart_record`): `pbt_utils.member_resume` retries the
        restart from it when the restarted process dies before it saves, and `_check_member_dir`
        removes a record above a relaunch's iteration. A record that can't be written doesn't stop
        the restart; an older one is removed then, as it would describe another restart.pth."""
        try:
            pbt_utils.write_restart_record(self.curr_policy_dir, self._restart_record())
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not write {pbt_utils.RESTART_RECORD}: {exc!r}")
            try:
                os.remove(os.path.join(self.curr_policy_dir, pbt_utils.RESTART_RECORD))
            except OSError:
                pass

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):
        """Re-exec the current process with the restart command and the transfer metadata.

        Notes:
            - The child reads the metadata via `pbt_utils.restart_info()` (env var
              `RL_GAMES_PBT_RESTART`, `_restart_payload`); rl_games uses it to mark the restart
              (`pbt_restart`: the checkpoint's sigma is kept), and environments can use it in
              `set_env_state`.
            - An open wandb run is finished first (see `_finish_wandb`).
            - Distributed members restart under torch.distributed.run with a standalone rendezvous.
            - The command runs from the launch directory (`launch_cwd`).
            - Under `pbt.restart_mode: exit` the process ends in place of the exec
              (`_exit_for_restart`).
        """
        print(f"previous command line args: {self.launch_argv if self.launch_argv is not None else sys.argv}")
        command = self._restart_command(new_params, restart_from_checkpoint)

        payload = dict(self._restart_payload(new_params, restart_from_checkpoint), frame_reached=self._frame_reached())
        os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps(payload, default=pbt_utils.json_default)

        self.algo.writer.flush()
        self.algo.writer.close()

        run = _wandb_run()
        if run is not None:
            self._finish_wandb(run, command)
        self._exec_restart(command)

    def _frame_reached(self):
        """The member's frame at its restart, this epoch's rollout included: the restarted process
        logs what it plays again past restart.pth's frame."""
        return int(self.algo.frame) + self._epoch_frames()

    def _exit_for_restart(self):
        """`pbt.restart_mode: exit` in place of the exec (rank 0): complete the member's
        restart.yaml with what an exec'd restart gets besides its command, the payload's
        frame_reached and the wandb variables `_finish_wandb` set (`environ`: the run to resume),
        and end the process with `pbt_utils.RESTART_EXIT_CODE`. The supervisor that runs the
        member's command (`python -m rl_games.common.pbt.supervise`) runs it again from the
        record, which frame_reached marks as one the member ended for (a transfer loaded in
        process writes it without); a record that can't be written only warns (the supervisor
        then relaunches from what the member's directory holds, `pbt_utils.member_resume`). The
        other ranks have exited with status 0 (`_sync_decision`), as in an exec: torchrun stops
        every rank once one fails, and rank 0 may still be finishing its wandb run."""
        environ = {key: os.environ[key] for key in _WANDB_RESTART_VARS if key in os.environ}
        try:
            pbt_utils.write_restart_record(self.curr_policy_dir, dict(
                self._restart_record(), frame_reached=self._frame_reached(), environ=environ))
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING could not write {pbt_utils.RESTART_RECORD}: {exc!r}")
        print(f"Policy {self.cfg.policy_idx}: restart_mode exit: the process ends with status "
              f"{pbt_utils.RESTART_EXIT_CODE}, for the supervisor to restart the member from "
              f"{pbt_utils.RESTART_RECORD}", flush=True)
        _flush_stdio()
        os._exit(pbt_utils.RESTART_EXIT_CODE)

    def _exec_restart(self, command):
        """Replace this process with `command`, run from the launch directory, or end it under
        `pbt.restart_mode: exit` (`_exit_for_restart`). Called on the main thread, or by the wandb
        watchdog thread while finish() blocks the main thread."""
        # an observer built without __init__ (DexLab's tests) has a cfg without the newer keys
        if getattr(self.cfg, "restart_mode", "exec") == "exit":
            return self._exit_for_restart()
        launch_cwd = self._launch_cwd()
        print("Running command:", command, flush=True)
        print(f"Policy {self.cfg.policy_idx}: Restarting self with args {command[1:]} in {launch_cwd}", flush=True)

        pbt_utils.dump_env_sizes()

        # dedup PATH-like env vars so the exec'd child's environment doesn't grow across restarts
        for var in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"):
            val = os.environ.get(var)
            if val and os.pathsep in val:
                os.environ[var] = pbt_utils.dedup_path_list(val)

        try:
            os.chdir(launch_cwd)
        except OSError as exc:
            print(f"Policy {self.cfg.policy_idx}: WARNING cannot return to the launch directory: {exc!r}")
        _flush_stdio()
        os.execv(command[0], command)


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices):
        self._call_multi("process_infos", infos, done_indices)

    def before_steps(self):
        self._call_multi("before_steps")

    def after_steps(self):
        self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)

    def after_train(self, reason):
        self._call_multi("after_train", reason)
