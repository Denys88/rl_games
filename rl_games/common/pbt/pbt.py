# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import random
import re
import sys
import threading
import traceback
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import pbt_utils
from rl_games.common.pbt.mutation import mutate
from rl_games.common.pbt.pbt_cfg import PbtCfg, rule_notices, validate  # noqa: F401 (PbtCfg: public here since #359)

_UNINITIALIZED_VALUE = pbt_utils.UNINITIALIZED_VALUE

# prefix of the algo-config keys among the flattened mutable params
_CONFIG_PREFIX = "agent.params.config."


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
                       wandb_args=None, rendering_args=None, checkpoint_arg="--checkpoint"):
    """Rebuild the training command line for a PBT restart from `sys.argv`-style args.

    Drops hydra-style ``key=value`` overrides that are being replaced by ``new_params``
    and any existing checkpoint argument (exact name match), keeps everything else
    verbatim, then appends the new checkpoint and the mutated params.
    """
    out = [cli_args[0]] + pbt_utils.remove_cli_arg(list(cli_args[1:]), checkpoint_arg)
    out = [out[0]] + [arg for arg in out[1:] if not ("=" in arg and arg.split("=", 1)[0] in new_params)]
    out.append(f"{checkpoint_arg}={restart_from_checkpoint}")
    if wandb_args is not None:
        out.extend(wandb_args.get_args_list())
    if rendering_args is not None:
        out.extend(rendering_args.get_args_list())
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


class PbtAlgoObserver(AlgoObserver):
    """rl_games observer that implements Population-Based Training for a single policy process.

    Each learner in the population runs as its own process with a unique `policy_idx`. On a
    fixed cadence (`interval_steps`) every learner saves a scored checkpoint into a shared
    workspace, compares itself against the population, and — if the replacement rule selects
    it — re-execs itself from a better member's checkpoint (or its own) with mutated
    hyperparameters.

    Restarts need a train script that accepts `key=value` hyperparameter overrides (Hydra
    style, as in Isaac Lab / IsaacGymEnvs train scripts) and a checkpoint argument. Pass the
    unmodified launch command as `launch_argv` when the script rewrites `sys.argv` (e.g. for
    Hydra); the restart then re-runs it with the checkpoint and the mutated overrides
    replaced, the seed replaced when the command passes `seed_arg` and `reseed_on_restart` is
    set, the script path made absolute, and `hydra.run.dir` pinned to the run dir inside a
    Hydra job that changed into it. The restart runs from the launch directory (Hydra's
    original cwd inside a Hydra job), see `pbt_utils.launch_dirs`.
    """

    def __init__(self, params, args_cli=None, launch_argv=None, extra_params=None):
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
        self.cfg, self.pbt_params = validate(params, extra_params, launcher_cwd=self.launch_cwd)
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
            print("PbtAlgoObserver: WARNING WORLD_SIZE > 1 but args_cli.distributed is not set: the restart flag "
                  "is not broadcast and ranks would diverge on a restart; set args_cli.distributed=True")
        if self.launch_argv is None and self.distributed_args.rank == 0 and pbt_utils.argv_rewritten():
            print("PbtAlgoObserver: NOTICE launch_argv is not set and sys.argv differs from the launch command "
                  "(sys.orig_argv): restarts rebuild the command from sys.argv and args_cli and drop the other "
                  "launch flags; pass launch_argv=list(sys.argv) captured before sys.argv is rewritten, or "
                  "['-m', '<module>', *sys.argv[1:]] for a `python -m` launch")
        seed_arg = self.cfg.seed_arg
        if (self.launch_argv is not None and self.distributed_args.rank == 0 and self.cfg.reseed_on_restart
                and seed_arg and pbt_utils.get_cli_arg(self.launch_argv, seed_arg) is None):
            print(f"PbtAlgoObserver: NOTICE launch_argv has no {seed_arg} argument: every restart seeds torch, "
                  f"numpy and the environment with the launch seed again, so their random streams replay after "
                  f"each restart (PBT's own draws come from a private generator); pass {seed_arg} on the command "
                  f"line, or set pbt.seed_arg to the name it uses (e.g. seed for a Hydra override), and each "
                  f"restart gets a seed derived from it")
        # PBT's draws come from a generator private to this member and process start (`_draw_rng`)
        self._seed = params.get("params", {}).get("seed")
        self._rng = None

        self.printer.print_params_table(self.pbt_params, header="List of params to mutate")

        info = pbt_utils.restart_info()
        self.restart = info if info is not None and info.get("policy_idx") == self.cfg.policy_idx else None
        self.restart_count = int(self.restart["restart_count"]) if self.restart else 0

        if self.distributed_args.distributed and torch.cuda.is_available():
            # NCCL broadcast requires a CUDA tensor
            self.device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
        else:
            self.device = params.get("params", {}).get("config", {}).get("device", "cpu")
        self.restart_flag = torch.tensor([0], device=self.device)

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
        """Keep the algo on every rank; rank 0 creates this policy's workspace folder.

        Args:
            algo: rl_games algorithm object (provides writer, train_dir, frame counter, etc.).

        Raises:
            NotImplementedError: for SAC, which never calls `after_steps` (PBT would do nothing).
        """
        from rl_games.algos_torch.sac_agent import SACAgent  # lazy: only this check needs the SAC module

        if isinstance(algo, SACAgent):
            raise NotImplementedError(
                "PBT supports the PPO agents (a2c_continuous, a2c_discrete); SAC never calls the observer's "
                "after_steps, so PBT would silently do nothing")
        self.algo = algo
        num_envs = max(1, int(getattr(algo, "num_actors", 1) or 1))
        if self.cfg.objective_window <= 0:
            self._window_size = num_envs
        # a failed rank needs a failed episode on every env: one env in a NaN reset loop while the
        # others are mid-episode is not a diverged policy
        self._failed_envs = np.zeros(num_envs, dtype=bool)
        if self.distributed_args.rank != 0:
            return

        self.root_dir = algo.train_dir
        self.ws_dir = os.path.join(self.root_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        os.makedirs(self.curr_policy_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        """Accumulate the objective of finished episodes into a window of recent episodes.

        `self.score` is this rank's mean over the window. An episode whose objective is NaN or
        infinite is left out on its own (see `pbt_utils.episode_stats`). For the iteration gate
        the rank counts its finished episodes since the last PBT iteration: with a finite
        objective, with a non-finite one, and on steps whose objective could not be read. Until
        it has a finite one, it also marks the envs that ended an episode.
        """
        value = infos
        num_agents = int(getattr(getattr(self, "algo", None), "num_agents", 1) or 1)
        try:
            for part in self.cfg.objective.split("."):
                value = value[part]
            total, count, nonfinite = pbt_utils.episode_stats(value, done_indices, num_agents)
        except (KeyError, TypeError, IndexError, ValueError, RuntimeError) as exc:
            # the address may not resolve on every step (or every backend); keep the previous
            # score, and report the last failure if no objective arrives by an iteration boundary
            self._last_objective_error = repr(exc)
            self._interval_unread += _num_finished(done_indices)
            self._mark_failed_envs(done_indices, num_agents)
            return
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
            1) All ranks: sync the restart flag; non-zero ranks exit and rank 0 re-execs
               when it is set. The first step after a (re)start then only sets the PBT
               cadence from the restored frame, and rank 0 checks the member directory
               (`_check_member_dir`).
            2) All ranks: pool the objective windows and episode counts across ranks and
               track the best pooled objective of the iteration.
            3) On the PBT cadence (`interval_steps`), once every rank has a full window and a
               finite objective since the last iteration, rank 0 saves a checkpoint, loads the
               population, logs, and applies the replacement rule (`_pbt_iteration`), with the
               global `random` module swapped to the member's private generator (`_draw_rng`).
               Until then the iteration is retried on every step without advancing `pbt_it`
               (see `_wait_for_objective`).
        """
        distributed = self._distributed()
        if distributed:
            dist.broadcast(self.restart_flag, src=0)

        if self.restart_flag.cpu().item() == 1:
            if self.distributed_args.rank != 0:
                _flush_stdio()
                os._exit(0)
            self._restart_with_new_params(self.new_params, self.restart_from_checkpoint)
            return

        frame = self.algo.frame
        if self.pbt_it == -1:
            # first step after (re)start: the checkpoint restore has already set the frame
            self.pbt_it = frame // self.cfg.interval_steps
            self.initial_frame = frame
            if self.distributed_args.rank == 0:
                self._check_member_dir(frame)
            return

        pooled = self._pooled_window(distributed)
        if pooled.known:
            objective = pooled.total / pooled.count
            if self.best_in_iteration is None or objective > self.best_in_iteration:
                self.best_in_iteration = objective

        iteration = frame // self.cfg.interval_steps
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
        if self.distributed_args.rank != 0:
            return
        # PBT's draws (donor, leader-params coin, mutations) come from this member's private stream,
        # also in a subclass that replaces _pbt_iteration and calls `random` itself
        with pbt_utils.swap_random_state(self._draw_rng()):
            self._pbt_iteration(pooled.total / pooled.count, best_in_iteration, frame)

    def _draw_rng(self):
        """The generator of PBT's draws, created at the first iteration from the config seed, the
        member, the restart count and the frame this process started at (`pbt_utils.draw_rng`)."""
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

        Rank 0 warns once per boundary. Every rank raises after `max_consecutive_errors`
        boundaries in a row at which a rank had failed (no finite objective since the last
        iteration or the process start, although every env of the rank ended an episode, and at
        least a window of them): a diverged member, also one that is NaN from its first step (a
        restart or resume from a diverged checkpoint), would otherwise train on without ever
        saving or being replaced. Boundaries that only wait for episodes to end don't count.
        Pooled values: every rank decides alike.
        """
        if iteration == self._wait_iteration:
            return
        self._wait_iteration = iteration
        cfg = self.cfg
        self._failed_boundaries = self._failed_boundaries + 1 if pooled.failed_ranks else 0
        counts = (f"{pooled.valid} finite, {pooled.nonfinite} non-finite and {pooled.unread} unreadable episodes "
                  f"since iteration {self.pbt_it}")
        if self._failed_boundaries >= cfg.max_consecutive_errors:
            raise RuntimeError(
                f"PBT objective '{cfg.objective}': {self._failed_boundaries} consecutive iteration boundaries at "
                f"which {pooled.failed_ranks} of {pooled.ranks} ranks had no finite objective although every env "
                f"had ended an episode (max_consecutive_errors): a diverged policy, or infos that no longer carry "
                f"the objective; {counts}; last lookup error: {self._last_objective_error}")
        if self.distributed_args.rank != 0:
            return
        if pooled.known:
            ranks = f" on {pooled.stale_ranks} of {pooled.ranks} ranks" if pooled.ranks > 1 else ""
            what = f"a new objective '{cfg.objective}'{ranks}: {counts}"
        else:
            what = (f"objective '{cfg.objective}': {pooled.count} finished episodes counted, {self._window_size} "
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
        unless `pbt.reuse_workspace` is set; this step's only collective (the restart flag
        broadcast) is behind it, and torchrun then stops the other ranks. Otherwise this member's
        files above its current iteration are deleted: a relaunch from an older checkpoint, or
        reused leftovers.
        """
        directory = getattr(self, "curr_policy_dir", None)
        if directory is None or not os.path.isdir(directory):
            return
        saved = pbt_utils.saved_iterations(directory)
        fresh = getattr(self, "restart", None) is None and frame == 0
        if fresh and saved and not getattr(self.cfg, "reuse_workspace", False):
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

    def _pbt_iteration(self, score, best_in_iteration, frame):
        """One PBT iteration on rank 0: save this member's checkpoint, load the population's, log,
        record a new population best, and apply the replacement rule.

        No decision inside the grace windows (`start_after` since the process started or a
        self-mutation was applied, `initial_delay` since the start of training), nor when the
        run ends before a restart could pay off (`_frames_to_end`); the member's yaml says
        `replaceable: false` then, so peers keep it out of their worst group. A self-mutation
        that changes no value keeps training; one with changes goes to `_apply_self_mutation`
        first, and restarts unless that applied it. A restart marks this iteration's yaml not
        replaceable once restart.pth is written: peers read it until the restarted process saves.
        """
        cfg = self.cfg
        frame_left = (self.pbt_it + 1) * cfg.interval_steps - frame
        print(f"Policy {cfg.policy_idx}, frames_left {frame_left}, PBT it {self.pbt_it}, objective {score:.6g}")
        in_grace = frame - self.initial_frame < cfg.start_after or frame < cfg.initial_delay
        to_end = self._frames_to_end(frame)
        ending = to_end is not None and to_end[0] < max(cfg.start_after, cfg.interval_steps)
        try:
            pbt_utils.save_pbt_checkpoint(self.curr_policy_dir, score, self.pbt_it, self.algo, self.pbt_params,
                                          replaceable=not (in_grace or ending))
            ckpts = pbt_utils.load_pbt_ckpts(self.ws_dir, cfg.policy_idx, cfg.num_policies, self.pbt_it)
            pbt_utils.cleanup(ckpts, self.curr_policy_dir)
        except Exception as exc:
            self._consecutive_errors += 1
            print(f"Policy {cfg.policy_idx}: exception {exc!r} in PBT checkpoint save/load "
                  f"({self._consecutive_errors}/{cfg.max_consecutive_errors})")
            if self._consecutive_errors >= cfg.max_consecutive_errors:
                raise RuntimeError(
                    f"PBT checkpoint save/load failed {self._consecutive_errors} times in a row") from exc
            return
        self._consecutive_errors = 0

        status = pbt_utils.member_status(ckpts, self.pbt_it)
        sumry = {i: None if c is None else {k: v for k, v in c.items() if k != "params"} for i, c in ckpts.items()}
        self.printer.print_ckpt_summary(sumry, status=status)
        unreported = sorted(p for p, s in status.items() if s == pbt_utils.NOT_REPORTED)
        if unreported:
            print(f"Policy {cfg.policy_idx}: WARNING members {unreported} have not reported (no checkpoint at or "
                  f"below iteration {self.pbt_it} in {self.ws_dir}), so they are not ranked: are all "
                  f"{cfg.num_policies} members (num_policies) running?")

        objectives = {p: (c["true_objective"] if c else None) for p, c in ckpts.items()}
        initialized = {p: o for p, o in objectives.items() if o is not None and o > _UNINITIALIZED_VALUE}
        # equal objectives rank in an order every member shares at this iteration, new at the next
        order = pbt_utils.tie_break_order(cfg.workspace, self.pbt_it, cfg.num_policies)
        self._write_summaries(score, initialized, frame, order=order)
        if not initialized:
            print("No policies initialized; skipping PBT iteration.")
            return

        best_policy = pbt_utils.rank_members(initialized, order)[0]
        if best_policy == cfg.policy_idx:
            try:
                pbt_utils.maybe_save_best(self.ws_dir, cfg.policy_idx, initialized[best_policy], self.pbt_it, frame,
                                          ckpts[best_policy]["checkpoint"], self.pbt_params)
            except Exception as exc:
                print(f"Policy {cfg.policy_idx}: exception {exc!r} while saving the population-best checkpoint")

        if in_grace:
            print(f"Policy {cfg.policy_idx}: within the grace window (frame {frame}, restarted at "
                  f"{self.initial_frame}, start_after {cfg.start_after}, initial_delay {cfg.initial_delay})")
            return
        if ending:
            print(f"Policy {cfg.policy_idx}: {to_end[0]} frames left after this epoch before {to_end[1]} ends the "
                  f"run, fewer than max(start_after, interval_steps) = {max(cfg.start_after, cfg.interval_steps)}: "
                  f"no replacement this close to the end (a restarted member could not leave its grace window "
                  f"or reach another iteration)")
            return

        if cfg.replace_rule == "dexpbt":
            eligible = {p for p, s in status.items() if s == pbt_utils.REPLACEABLE}
            source, reason = pbt_utils.select_dexpbt(objectives, cfg.policy_idx, score, best_in_iteration, cfg,
                                                     eligible=eligible, order=order)
        else:
            source, reason = pbt_utils.select_threshold(objectives, cfg.policy_idx, cfg)
        print(f"Policy {cfg.policy_idx}: {reason}")
        if source is None:
            return

        source_params = ckpts[source]["params"]
        use_source_params = cfg.replace_rule == "threshold" or random.random() < cfg.leader_params_prob
        base_params = source_params if use_source_params else self.pbt_params
        new_params = mutate(base_params, cfg.mutation, cfg.mutation_rate, cfg.change_range)
        if source == cfg.policy_idx:
            changed = {k: v for k, v in new_params.items() if self.pbt_params.get(k) != v}
            if not changed:
                # a restart would cost a start-up and a new grace window for the same values
                print(f"Policy {cfg.policy_idx}: no-op self-mutation: keep training (no value changed)")
                return
            if self._apply_self_mutation(changed):
                # at the x of this iteration's current values, as a restart logs them
                self._write_params(self.algo.writer, new_params, self._summary_step(frame))
                self.algo.writer.flush()
                self.printer.print_mutation_diff(self.pbt_params, new_params,
                                                 header="Mutated params (applied in process)")
                self.pbt_params = new_params
                # the grace window starts over; the saves in it say replaceable: false
                self.initial_frame = int(frame)
                print(f"Policy {cfg.policy_idx}: self-mutation applied in process, no restart; grace window from "
                      f"frame {self.initial_frame}")
                return
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
                history_entry={"policy_idx": cfg.policy_idx, "source_policy": source, "frame": int(frame),
                               "objective": float(score), "source_objective": float(objectives[source])},
                config_hparams=config_hparams or None,
                last_mean_rewards=float(self.algo.last_mean_rewards) if fixed_name else None)
        except Exception as exc:
            # the source checkpoint may have been cleaned up meanwhile: keep training as is
            print(f"Policy {cfg.policy_idx}: exception {exc!r} while preparing the restart checkpoint; "
                  f"continuing with the current weights")
            return

        # at the x of this iteration's current values: the plot steps to the new ones (as IsaacGymEnvs)
        self._write_params(self.algo.writer, new_params, self._summary_step(frame))
        self.algo.writer.flush()

        self.new_params = new_params
        self.restart_from_checkpoint = restart_checkpoint
        self.restart_source = source
        self.restart_frame = int(frame)
        # values the restored checkpoint would otherwise impose on the new process
        self.restart_mutated = {k: v for k, v in new_params.items() if source_params.get(k) != v}
        self.restart_flag[0] = 1
        self.printer.print_mutation_diff(base_params, new_params)
        # peers read this iteration's yaml as the member's state until the restarted process saves
        try:
            pbt_utils.mark_not_replaceable(self.curr_policy_dir)
        except Exception as exc:
            print(f"Policy {cfg.policy_idx}: WARNING could not mark the newest workspace yaml not replaceable: "
                  f"{exc!r}")

    def _apply_self_mutation(self, changed):
        """Hook: apply a self-mutation in the running process instead of restarting; return True
        when it was applied. The base observer restarts (returns False).

        Called on rank 0 inside `_pbt_iteration` for a self-mutation (the member keeps its own
        weights) that changes some values, before anything is written for a restart. `changed`
        maps each changed flattened param to its new value. On True the observer takes the new
        values as `pbt_params`, logs them, restarts the member's grace window (`initial_frame`;
        its saves say `replaceable: false` until `start_after` frames have passed) and keeps
        training. The subclass must apply the values on every rank (rank 0 broadcasts them, e.g.
        after `super().after_steps()`), and should call `_reset_objective_window()` on every rank:
        the next iteration then ranks the member by episodes played with the new values.
        """
        return False

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

    def _frames_to_end(self, frame):
        """(frames, limit): the frames this member trains after the current epoch before `limit`
        (max_epochs or max_frames, the nearer one) ends the run; None without either limit.

        `after_steps` runs before the epoch's frames are added to `algo.frame`, and the run ends
        after the epoch that reaches a limit: in the last epoch this is <= 0, and no later
        `after_steps` would act on a restart flag.
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

    def _restart_command(self, new_params, restart_from_checkpoint):
        """Full re-exec command: launcher, optional torch.distributed.run prefix, script args.

        The script path is made absolute (see `pbt_utils.absolute_script_path`), and inside a
        Hydra job that changed into its run dir the command pins `hydra.run.dir` to it.
        """
        cfg = self.cfg
        new_params = {param: pbt_utils.plain_number(value) for param, value in new_params.items()}
        if self.launch_argv is not None:
            script_args = pbt_utils.build_launch_restart_args(
                self.launch_argv, new_params, restart_from_checkpoint, cfg.checkpoint_arg)
        else:
            modified_args = build_restart_args(sys.argv, new_params, restart_from_checkpoint,
                                               self.wandb_args, self.rendering_args, cfg.checkpoint_arg)
            script_args = [modified_args[0]] + self.env_args.get_args_list() + modified_args[1:]
            if self.distributed_args.distributed:
                script_args.append("--distributed")

        base = pbt_utils.get_cli_arg(script_args, cfg.seed_arg) if cfg.reseed_on_restart and cfg.seed_arg else None
        if base is not None:
            try:
                seed = pbt_utils.derive_seed(int(base), cfg.policy_idx, self.restart_count + 1)
            except ValueError:
                seed = None
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

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):
        """Re-exec the current process with the restart command and the transfer metadata.

        Notes:
            - The child reads the metadata via `pbt_utils.restart_info()` (env var
              `RL_GAMES_PBT_RESTART`); rl_games uses it to mark the restart (`pbt_restart`:
              the checkpoint's sigma is kept), and environments can use it in `set_env_state`.
            - An open wandb run is finished first (see `_finish_wandb`).
            - Distributed members restart under torch.distributed.run with a standalone rendezvous.
            - The command runs from the launch directory (`launch_cwd`).
        """
        print(f"previous command line args: {self.launch_argv if self.launch_argv is not None else sys.argv}")
        command = self._restart_command(new_params, restart_from_checkpoint)

        os.environ[pbt_utils.RESTART_ENV_VAR] = json.dumps({
            "policy_idx": self.cfg.policy_idx,
            "source_policy": getattr(self, "restart_source", self.cfg.policy_idx),
            "restart_count": self.restart_count + 1,
            "iteration": self.pbt_it,
            "frame": getattr(self, "restart_frame", int(self.algo.frame)),
            "checkpoint": restart_from_checkpoint,
            "mutated": getattr(self, "restart_mutated", new_params),
        }, default=pbt_utils.json_default)

        self.algo.writer.flush()
        self.algo.writer.close()

        run = _wandb_run()
        if run is not None:
            self._finish_wandb(run, command)
        self._exec_restart(command)

    def _exec_restart(self, command):
        """Replace this process with `command`, run from the launch directory. Called on the main
        thread, or by the wandb watchdog thread while finish() blocks the main thread."""
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

    def after_steps(self):
        self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)
