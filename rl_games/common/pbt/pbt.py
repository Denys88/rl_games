# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import json
import math
import os
import random
import sys
from collections import deque

import torch
import torch.distributed as dist

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.pbt import pbt_utils
from rl_games.common.pbt.mutation import mutate
from rl_games.common.pbt.pbt_cfg import PbtCfg

_UNINITIALIZED_VALUE = pbt_utils.UNINITIALIZED_VALUE

# prefix of mutable algo-config keys; restarts use it to tell rl_games which config values
# must win over the checkpoint's runtime copies
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
    Hydra); the restart then re-runs it verbatim with only the checkpoint and mutated
    overrides replaced.
    """

    def __init__(self, params, args_cli=None, launch_argv=None, extra_params=None):
        """Initialize observer, print the mutation table, and allocate the restart flag.

        Args:
            params (dict): Full agent/task params (Hydra style), containing a "pbt" section.
            args_cli: Parsed CLI args used to reconstruct the restart command when `launch_argv`
                is not given. Any argparse namespace works; missing attributes are treated as
                disabled/absent.
            launch_argv (list[str] | None): The original script argv (script path first),
                captured before argparse/Hydra rewrite `sys.argv`.
            extra_params (dict | None): Additional mutable values outside the agent params,
                keyed by their `key=value` override name (e.g. `env.rewards.success.weight`).
        """
        super().__init__()
        self.printer = pbt_utils.PbtTablePrinter()
        self.dir = params["pbt"]["directory"]
        self.launch_argv = list(launch_argv) if launch_argv is not None else None

        self.rendering_args = pbt_utils.RenderingArgs(args_cli)
        # the launch argv already carries the wandb flags
        self.wandb_args = pbt_utils.WandbArgs(args_cli if self.launch_argv is None else None)
        self.env_args = pbt_utils.EnvArgs(args_cli)
        self.distributed_args = pbt_utils.DistributedArgs(args_cli)
        self.cfg = PbtCfg.from_dict(params["pbt"])
        if not self.cfg.objective:
            raise ValueError(
                "pbt.objective is required: a dotted address into env infos, "
                "e.g. 'episode.Episode_Reward/success' (Isaac Lab) or 'scores' (flat infos)")
        self.pbt_it = -1  # dummy value, stands for "not initialized"
        self.initial_frame = 0
        self.score = _UNINITIALIZED_VALUE
        self.best_in_iteration = None
        self._window = deque()
        self._window_sum = 0.0
        self._window_count = 0
        self._window_size = max(1, self.cfg.objective_window)
        self._consecutive_errors = 0

        flat = pbt_utils.flatten_dict({"agent": params})
        flat.update(extra_params or {})
        self.pbt_params = pbt_utils.filter_params(flat, self.cfg.mutation)
        missing = sorted(set(self.cfg.mutation) - set(self.pbt_params))
        if missing:
            raise ValueError(f"pbt.mutation keys not found in the params or extra_params: {missing}")

        assert len(self.pbt_params) > 0, "[DANGER]: Dictionary that contains params to mutate is empty"
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
        """On a PBT restart, mark mutated algo-config keys so the checkpoint restore keeps them.

        rl_games restores runtime copies of some config values (learning rate, entropy
        coefficient) from checkpoints; for mutated values the new config must win.
        """
        if self.restart is None:
            return
        keys = [k[len(_CONFIG_PREFIX):] for k in self.restart.get("mutated", {})
                if k.startswith(_CONFIG_PREFIX) and "." not in k[len(_CONFIG_PREFIX):]]
        if keys:
            config["pbt_restart_overrides"] = keys

    def after_init(self, algo):
        """Keep the algo on every rank; rank 0 creates this policy's workspace folder.

        Args:
            algo: rl_games algorithm object (provides writer, train_dir, frame counter, etc.).
        """
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
        try:
            for part in self.cfg.objective.split("."):
                value = value[part]
            total, count = pbt_utils.episode_sum_count(value, done_indices)
        except (KeyError, TypeError, IndexError, ValueError, RuntimeError):
            # the address may not resolve on every step (or every backend);
            # keep the previous score instead of killing training
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
        if not distributed:
            return self._window_sum, self._window_count, self._window_count >= self._window_size
        sums = torch.tensor([self._window_sum, float(self._window_count)], dtype=torch.float64, device=self.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        total, count = sums[0].item(), sums[1].item()
        return total, count, count >= self._window_size * dist.get_world_size()

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

        if frame // self.cfg.interval_steps <= self.pbt_it or not known:
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
            restart_checkpoint = pbt_utils.prepare_restart_checkpoint(
                ckpts[source]["checkpoint"], self.curr_policy_dir, frame,
                {"policy_idx": cfg.policy_idx, "source_policy": source, "frame": int(frame),
                 "objective": float(score), "source_objective": float(objectives[source])})
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
        """Full re-exec command: launcher, optional torch.distributed.run prefix, script args."""
        cfg = self.cfg
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

        command = [self._get_launcher()]
        if self.distributed_args.distributed:
            self.distributed_args.master_port = str(pbt_utils.find_free_port())
            command.extend(self.distributed_args.get_args_list())
        return command + script_args

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):
        """Re-exec the current process with the restart command and the transfer metadata.

        Notes:
            - The child reads the metadata via `pbt_utils.restart_info()` (env var
              `RL_GAMES_PBT_RESTART`); rl_games uses it to keep mutated config values over the
              checkpoint's, and environments can use it in `set_env_state`.
            - On distributed runs, assigns a fresh master port and forwards distributed args.
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
        })

        self.algo.writer.flush()
        self.algo.writer.close()

        wandb_run = _wandb_run()
        if self.wandb_args.enabled or wandb_run is not None:
            import wandb

            # setdefault only affects the restarted child process
            os.environ.setdefault("WANDB_RUN_ID", wandb.run.id)  # continue with the same run id
            os.environ.setdefault("WANDB_RESUME", "allow")  # allow wandb to resume
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")  # give wandb init more time to be fault tolerant
            wandb.run.finish()

        print("Running command:", command, flush=True)
        print(f"Policy {self.cfg.policy_idx}: Restarting self with args {command[1:]}", flush=True)

        pbt_utils.dump_env_sizes()

        # dedup PATH-like env vars so the exec'd child's environment doesn't grow across restarts
        for var in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"):
            val = os.environ.get(var)
            if not val or os.pathsep not in val:
                continue
            seen = set()
            new_parts = []
            for p in val.split(os.pathsep):
                if p and p not in seen:
                    seen.add(p)
                    new_parts.append(p)
            os.environ[var] = os.pathsep.join(new_parts)

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
