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

import torch
import torch.distributed as dist

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import pbt_utils
from rl_games.common.pbt.mutation import mutate
from rl_games.common.pbt.pbt_cfg import PbtCfg, rule_notices, validate  # noqa: F401 (PbtCfg: public here since #359)

_UNINITIALIZED_VALUE = pbt_utils.UNINITIALIZED_VALUE

# prefix of the algo-config keys among the flattened mutable params
_CONFIG_PREFIX = "agent.params.config."


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
        self._warned_iteration = None

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
        if self.cfg.objective_window <= 0:
            self._window_size = max(1, int(getattr(algo, "num_actors", 1) or 1))
        if self.distributed_args.rank != 0:
            return

        self.root_dir = algo.train_dir
        self.ws_dir = os.path.join(self.root_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        os.makedirs(self.curr_policy_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        """Accumulate the objective of finished episodes into a window of recent episodes.

        `self.score` is this rank's mean over the window.
        """
        value = infos
        num_agents = int(getattr(getattr(self, "algo", None), "num_agents", 1) or 1)
        try:
            for part in self.cfg.objective.split("."):
                value = value[part]
            total, count = pbt_utils.episode_sum_count(value, done_indices, num_agents)
        except (KeyError, TypeError, IndexError, ValueError, RuntimeError) as exc:
            # the address may not resolve on every step (or every backend); keep the previous
            # score, and report the last failure if no objective arrives by an iteration boundary
            self._last_objective_error = repr(exc)
            return
        if count == 0 or not math.isfinite(total):
            return
        self._window.append((total, count))
        self._window_sum += total
        self._window_count += count
        while len(self._window) > 1 and self._window_count - self._window[0][1] >= self._window_size:
            old_total, old_count = self._window.popleft()
            self._window_sum -= old_total
            self._window_count -= old_count
        self.score = self._window_sum / self._window_count

    def _distributed(self):
        return self.distributed_args.distributed and dist.is_available() and dist.is_initialized()

    def _pooled_window(self, distributed):
        """(sum, count, known) of the episode windows of all ranks; known once every rank's is full."""
        full = self._window_count >= self._window_size
        if not distributed:
            return self._window_sum, self._window_count, full
        sums = torch.tensor([self._window_sum, float(self._window_count), float(full)],
                            dtype=torch.float64, device=self.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        total, count, full_ranks = sums.tolist()
        return total, count, int(full_ranks) == dist.get_world_size()

    def after_steps(self):
        """Main PBT tick executed every train step.

        Flow:
            1) All ranks: sync the restart flag; non-zero ranks exit and rank 0 re-execs
               when it is set.
            2) All ranks: pool the objective window across ranks and track the best pooled
               objective of the iteration.
            3) On the PBT cadence (`interval_steps`), once every rank has a full window, rank 0
               saves a checkpoint, loads the population, logs, and applies the replacement
               rule (after the `initial_delay` / `start_after` grace windows). Until the
               windows are full the iteration is retried on every step.
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
            return

        total, count, known = self._pooled_window(distributed)
        if known:
            objective = total / count
            if self.best_in_iteration is None or objective > self.best_in_iteration:
                self.best_in_iteration = objective

        if frame // self.cfg.interval_steps <= self.pbt_it:
            return
        if not known:
            iteration = frame // self.cfg.interval_steps
            if self.distributed_args.rank == 0 and self._warned_iteration != iteration:
                self._warned_iteration = iteration
                print(f"Policy {self.cfg.policy_idx}: WARNING PBT iteration {iteration} waits for objective "
                      f"'{self.cfg.objective}': {int(count)} finished episodes counted, "
                      f"{self._window_size} per rank needed; last lookup error: {self._last_objective_error}")
            return

        self.pbt_it = frame // self.cfg.interval_steps
        best_in_iteration, self.best_in_iteration = self.best_in_iteration, None
        if self.distributed_args.rank != 0:
            return
        self._pbt_iteration(total / count, best_in_iteration, frame)

    def _pbt_iteration(self, score, best_in_iteration, frame):
        cfg = self.cfg
        frame_left = (self.pbt_it + 1) * cfg.interval_steps - frame
        print(f"Policy {cfg.policy_idx}, frames_left {frame_left}, PBT it {self.pbt_it}, objective {score:.6g}")
        try:
            pbt_utils.save_pbt_checkpoint(self.curr_policy_dir, score, self.pbt_it, self.algo, self.pbt_params)
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

        sumry = {i: None if c is None else {k: v for k, v in c.items() if k != "params"} for i, c in ckpts.items()}
        self.printer.print_ckpt_summary(sumry)

        objectives = {p: (c["true_objective"] if c else None) for p, c in ckpts.items()}
        initialized = {p: o for p, o in objectives.items() if o is not None and o > _UNINITIALIZED_VALUE}
        self._write_summaries(score, initialized, frame)
        if not initialized:
            print("No policies initialized; skipping PBT iteration.")
            return

        best_objective, best_policy = max((o, p) for p, o in initialized.items())
        if best_policy == cfg.policy_idx:
            try:
                pbt_utils.maybe_save_best(self.ws_dir, cfg.policy_idx, best_objective, self.pbt_it, frame,
                                          ckpts[best_policy]["checkpoint"], self.pbt_params)
            except Exception as exc:
                print(f"Policy {cfg.policy_idx}: exception {exc!r} while saving the population-best checkpoint")

        if frame - self.initial_frame < cfg.start_after or frame < cfg.initial_delay:
            print(f"Policy {cfg.policy_idx}: within the grace window (frame {frame}, restarted at "
                  f"{self.initial_frame}, start_after {cfg.start_after}, initial_delay {cfg.initial_delay})")
            return

        if cfg.replace_rule == "dexpbt":
            source, reason = pbt_utils.select_dexpbt(objectives, cfg.policy_idx, score, best_in_iteration, cfg)
        else:
            source, reason = pbt_utils.select_threshold(objectives, cfg.policy_idx, cfg)
        print(f"Policy {cfg.policy_idx}: {reason}")
        if source is None:
            return

        source_params = ckpts[source]["params"]
        use_source_params = cfg.replace_rule == "threshold" or random.random() < cfg.leader_params_prob
        base_params = source_params if use_source_params else self.pbt_params
        new_params = mutate(base_params, cfg.mutation, cfg.mutation_rate, cfg.change_range)
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

        for param, value in new_params.items():
            self.algo.writer.add_scalar(f"pbt/{param}", value, frame)
        self.algo.writer.flush()

        self.new_params = new_params
        self.restart_from_checkpoint = restart_checkpoint
        self.restart_source = source
        self.restart_frame = int(frame)
        # values the restored checkpoint would otherwise impose on the new process
        self.restart_mutated = {k: v for k, v in new_params.items() if source_params.get(k) != v}
        self.restart_flag[0] = 1
        self.printer.print_mutation_diff(base_params, new_params)

    def _write_summaries(self, score, initialized, frame):
        writer = getattr(self.algo, "writer", None)
        if writer is None:
            return
        if score > _UNINITIALIZED_VALUE:
            writer.add_scalar("pbt/objective", score, frame)
        if initialized:
            writer.add_scalar("pbt/00_best_objective", max(initialized.values()), frame)
            if self.cfg.policy_idx in initialized:
                rank = 1 + sorted(initialized.values(), reverse=True).index(initialized[self.cfg.policy_idx])
                writer.add_scalar("pbt/rank", rank, frame)
        writer.add_scalar("pbt/restart_count", self.restart_count, frame)
        writer.flush()

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
