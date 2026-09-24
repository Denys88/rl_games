# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import json
import math
import os
import random
import shutil
import socket
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml

from rl_games.algos_torch.torch_ext import safe_filesystem_op, safe_save

# i.e. value for target objective when it is not known
UNINITIALIZED_VALUE = float(-1e9)


class DistributedArgs:
    """Distributed-launch flags reconstructed for a PBT restart.

    All attribute lookups are defaulted so any argparse namespace (or None) works.
    """

    def __init__(self, args_cli):
        self.distributed = getattr(args_cli, "distributed", False)
        self.nproc_per_node = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.nnodes = 1
        self.master_port = getattr(args_cli, "master_port", None)

    def get_args_list(self) -> list[str]:
        args = ["-m", "torch.distributed.run", f"--nnodes={self.nnodes}", f"--nproc_per_node={self.nproc_per_node}"]
        if self.master_port:
            args.append(f"--master_port={self.master_port}")
        return args


class EnvArgs:
    """Environment CLI flags (Isaac Lab style) forwarded across a PBT restart."""

    def __init__(self, args_cli):
        self.task = getattr(args_cli, "task", None)
        seed = getattr(args_cli, "seed", None)
        self.seed = seed if seed is not None else -1
        self.headless = getattr(args_cli, "headless", False)
        self.num_envs = getattr(args_cli, "num_envs", None)

    def get_args_list(self) -> list[str]:
        args = []
        if self.task is not None:
            args.append(f"--task={self.task}")
        args.append(f"--seed={self.seed}")
        if self.num_envs is not None:
            args.append(f"--num_envs={self.num_envs}")
        if self.headless:
            args.append("--headless")
        return args


class RenderingArgs:
    """Rendering/video CLI flags (Isaac Lab style) forwarded across a PBT restart."""

    def __init__(self, args_cli):
        self.camera_enabled = getattr(args_cli, "enable_cameras", False)
        self.video = getattr(args_cli, "video", False)
        self.video_length = getattr(args_cli, "video_length", None)
        self.video_interval = getattr(args_cli, "video_interval", None)

    def get_args_list(self) -> list[str]:
        args = []
        if self.camera_enabled:
            args.append("--enable_cameras")
        if self.video:
            args.extend(["--video", f"--video_length={self.video_length}", f"--video_interval={self.video_interval}"])
        return args


class WandbArgs:
    """Weights & Biases CLI flags forwarded across a PBT restart."""

    def __init__(self, args_cli):
        self.enabled = getattr(args_cli, "track", False)
        self.project_name = getattr(args_cli, "wandb_project_name", None)
        self.name = getattr(args_cli, "wandb_name", None)
        self.entity = getattr(args_cli, "wandb_entity", None)
        # fail fast: a missing entity would otherwise only surface at restart
        # time, killing the process mid-training and shrinking the population
        if self.enabled and not self.entity:
            raise ValueError("wandb entity must be specified when tracking is enabled")

    def get_args_list(self) -> list[str]:
        args = []
        if self.enabled:
            args.append("--track")
            if self.entity:
                args.append(f"--wandb-entity={self.entity}")
            else:
                raise ValueError("entity must be specified if wandb is enabled")
            if self.project_name:
                args.append(f"--wandb-project-name={self.project_name}")
            if self.name:
                args.append(f"--wandb-name={self.name}")
        return args


def dump_env_sizes():
    """Print summary of environment variable usage (count, bytes, top-5 largest, SC_ARG_MAX)."""

    n = len(os.environ)
    # total bytes in "KEY=VAL\0" for all envp entries
    total = sum(len(k) + 1 + len(v) + 1 for k, v in os.environ.items())
    # find the 5 largest values
    biggest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]

    print(f"[ENV MONITOR] vars={n}, total_bytes={total}")
    for k, v in biggest:
        print(f"    {k!r} length={len(v)} → {v[:60]}{'…' if len(v) > 60 else ''}")

    try:
        argmax = os.sysconf("SC_ARG_MAX")
        print(f"[ENV MONITOR] SC_ARG_MAX = {argmax}")
    except (ValueError, AttributeError):
        pass


def flatten_dict(d, prefix="", separator="."):
    """Flatten nested dictionaries into a flat dict with keys joined by `separator`."""

    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def find_free_port(max_tries: int = 20) -> int:
    """Return an OS-assigned free TCP port, with a few retries; fall back to a random high port."""
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", 0))
                return s.getsockname()[1]
            except OSError:
                continue
    return random.randint(20000, 65000)


def filter_params(params, params_to_mutate):
    """Filter `params` to only those in `params_to_mutate`, converting str floats (e.g. '1e-4') to float."""

    def try_float(v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        return v

    return {k: try_float(v) for k, v in params.items() if k in params_to_mutate}


def to_float(value) -> float:
    """Plain Python float from a number, numpy value or tensor (mean over its elements).

    The workspace YAML must hold plain floats: torch tags do not load back with FullLoader.
    """
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    return float(value)


def episode_sum_count(value, done_indices) -> tuple[float, int]:
    """Objective contribution of one step: (sum over finished episodes, number of episodes).

    A per-env tensor/array is read at `done_indices`. A scalar is taken as the mean over the
    envs that finished on this step (Isaac Lab episode logs), so it is weighted by their count;
    without `done_indices` it counts as one sample.
    """
    n_done = None
    if done_indices is not None:
        idx = done_indices.reshape(-1) if torch.is_tensor(done_indices) else np.asarray(done_indices).reshape(-1)
        n_done = int(idx.shape[0])
    if torch.is_tensor(value) and value.numel() > 1:
        if n_done is None:
            return float(value.detach().float().sum().item()), int(value.numel())
        if n_done == 0:
            return 0.0, 0
        picked = value.reshape(-1)[torch.as_tensor(idx, device=value.device)]
        return float(picked.detach().float().sum().item()), n_done
    if isinstance(value, np.ndarray) and value.size > 1:
        if n_done is None:
            return float(value.sum()), int(value.size)
        if n_done == 0:
            return 0.0, 0
        idx_np = idx.cpu().numpy() if torch.is_tensor(idx) else idx
        return float(value.reshape(-1)[idx_np].sum()), n_done
    scalar = to_float(value)
    if n_done is None:
        return scalar, 1
    return scalar * n_done, n_done


def _read_text(path) -> str:
    with open(path) as fobj:
        return fobj.read()


def _atomic_write_yaml(path, data) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fobj:
        yaml.dump(data, fobj)
    os.replace(tmp, path)


def _atomic_torch_save(state, path) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    safe_save(state, tmp)
    os.replace(tmp, path)


_CKPT_KEYS = ("iteration", "true_objective", "frame", "params", "checkpoint")


def save_pbt_checkpoint(workspace_dir, curr_policy_score, curr_iter, algo, params):
    """Save a PBT checkpoint (.pth and .yaml) with policy state, score, and metadata (rank 0 only).

    Both files are written to a temp name and renamed, so readers never see partial files.
    """
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.pth")
        _atomic_torch_save(algo.get_full_state_weights(), checkpoint_file)
        pbt_checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.yaml")

        pbt_checkpoint = {
            "iteration": curr_iter,
            "true_objective": to_float(curr_policy_score),
            "frame": int(algo.frame),
            "params": params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": algo.experiment_name,
        }
        _atomic_write_yaml(pbt_checkpoint_file, pbt_checkpoint)


def load_pbt_ckpts(workspace_dir, cur_policy_id, num_policies, pbt_iteration) -> dict | None:
    """
    Load the latest available PBT checkpoint for each policy (≤ current iteration).
    Returns a dict mapping policy_idx → checkpoint dict or None. (rank 0 only)
    Files missing required keys are skipped.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    checkpoints = dict()
    for policy_idx in range(num_policies):
        checkpoints[policy_idx] = None
        policy_dir = os.path.join(workspace_dir, f"{policy_idx:03d}")

        if not os.path.isdir(policy_dir):
            continue

        pbt_checkpoint_files = sorted(
            [f for f in os.listdir(policy_dir) if f.endswith(".yaml") and f[:-5].isdigit()], reverse=True)
        for pbt_checkpoint_file in pbt_checkpoint_files:
            iteration = int(pbt_checkpoint_file.split(".")[0])
            if iteration > pbt_iteration:
                continue

            path = os.path.join(policy_dir, pbt_checkpoint_file)
            ctime_ts = os.path.getctime(path)
            created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"Policy {cur_policy_id} [{now_str}]: Loading"
                f" policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})"
            )
            text = safe_filesystem_op(_read_text, path)
            try:
                ckpt = yaml.load(text, Loader=yaml.FullLoader)
            except yaml.YAMLError:
                ckpt = None
            if not isinstance(ckpt, dict) or any(k not in ckpt for k in _CKPT_KEYS):
                print(f"Policy {cur_policy_id}: skipping malformed PBT checkpoint {path}")
                continue
            checkpoints[policy_idx] = ckpt
            break

    return checkpoints


def cleanup(checkpoints: dict[int, dict], policy_dir, keep_back: int = 20, max_yaml: int = 50) -> None:
    """
    Cleanup old checkpoints for the current policy directory (rank 0 only).
    - Delete files older than (oldest iteration among live members - keep_back). A member
      more than keep_back iterations behind the newest one counts as stopped and is ignored.
    - Keep at most `max_yaml` latest YAML iterations.
    """
    if int(os.environ.get("RANK", "0")) == 0:
        iterations = [ckpt["iteration"] for ckpt in checkpoints.values() if ckpt]
        if not iterations:
            return
        newest = max(iterations)
        live = [it for it in iterations if it >= newest - keep_back]
        threshold = max(0, min(live) - keep_back)
        root = Path(policy_dir)

        # group files by numeric iteration (only *.yaml / *.pth)
        groups: dict[int, list[Path]] = {}
        for p in root.iterdir():
            if p.suffix in (".yaml", ".pth") and p.stem.isdigit():
                groups.setdefault(int(p.stem), []).append(p)

        # 1) drop anything older than threshold
        for it in [i for i in groups if i <= threshold]:
            for p in groups[it]:
                p.unlink(missing_ok=True)
            groups.pop(it, None)

        # 2) cap total YAML checkpoints: keep newest `max_yaml` iters
        yaml_iters = sorted((i for i, ps in groups.items() if any(p.suffix == ".yaml" for p in ps)), reverse=True)
        for it in yaml_iters[max_yaml:]:
            for p in groups.get(it, []):
                p.unlink(missing_ok=True)
            groups.pop(it, None)


def prepare_restart_checkpoint(source_checkpoint, policy_dir, frame, history_entry) -> str:
    """Copy the source checkpoint into this member's workspace for the restart.

    The copy survives the source's cleanup, keeps this member's frame count (DexPBT) and
    appends `history_entry` to the checkpoint's `pbt_history`.
    """
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    state["frame"] = int(frame)
    state["pbt_history"] = list(state.get("pbt_history", [])) + [history_entry]
    restart_checkpoint = os.path.join(policy_dir, "restart.pth")
    _atomic_torch_save(state, restart_checkpoint)
    return os.path.abspath(restart_checkpoint)


def maybe_save_best(workspace_dir, policy_idx, objective, iteration, frame, checkpoint, params) -> bool:
    """Keep a copy of the population-best checkpoint in <workspace>/best (called by the best member).

    Checkpoint copies have unique names and `best.yaml` names the current one, so concurrent
    writers never leave a checkpoint/metadata pair from different members.
    """
    best_dir = os.path.join(workspace_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    meta_file = os.path.join(best_dir, "best.yaml")
    if os.path.isfile(meta_file):
        try:
            previous = yaml.load(_read_text(meta_file), Loader=yaml.FullLoader) or {}
        except yaml.YAMLError:
            previous = {}
        if to_float(previous.get("true_objective", UNINITIALIZED_VALUE)) >= objective:
            return False
    name = f"best_p{policy_idx:03d}_{iteration:06d}.pth"
    best_checkpoint = os.path.join(best_dir, name)
    tmp = f"{best_checkpoint}.tmp{os.getpid()}"
    shutil.copyfile(checkpoint, tmp)
    os.replace(tmp, best_checkpoint)
    _atomic_write_yaml(meta_file, {
        "policy_idx": policy_idx, "iteration": iteration, "frame": int(frame),
        "true_objective": to_float(objective), "params": params,
        "checkpoint": os.path.abspath(best_checkpoint), "source_checkpoint": os.path.abspath(checkpoint),
    })
    # drop this member's older copies that best.yaml no longer names
    current = os.path.basename(yaml.load(_read_text(meta_file), Loader=yaml.FullLoader)["checkpoint"])
    for f in os.listdir(best_dir):
        if f.startswith(f"best_p{policy_idx:03d}_") and f.endswith(".pth") and f != current:
            try:
                os.remove(os.path.join(best_dir, f))
            except FileNotFoundError:
                pass
    return True


def select_dexpbt(objectives: dict[int, float | None], policy_idx: int, current_objective: float,
                  best_in_iteration: float | None, cfg) -> tuple[int | None, str]:
    """DexPBT replacement rule.

    Returns (source, reason): None keeps training; `policy_idx` restarts from its own
    checkpoint with mutated parameters; any other index restarts from that member.
    """
    n = cfg.num_policies
    ranked = sorted(((UNINITIALIZED_VALUE if objectives.get(p) is None else objectives[p], p) for p in range(n)),
                    reverse=True)
    ranked_policies = [p for _, p in ranked]
    initialized = [o for o, _ in ranked if o > UNINITIALIZED_VALUE]

    replace_worst = math.ceil(cfg.replace_fraction_worst * n)
    replace_best = math.ceil(cfg.replace_fraction_best * n)
    top = [(o, p) for o, p in ranked[:replace_best] if o > UNINITIALIZED_VALUE]
    best_policies = [p for _, p in top]
    worst_policies = ranked_policies[-replace_worst:]

    if policy_idx not in worst_policies:
        return None, f"not among the worst {worst_policies}"
    if len(initialized) <= max(2, n // 2) or not top:
        return None, f"only {len(initialized)} of {n} members reported"
    if best_in_iteration is not None and best_in_iteration >= min(o for o, _ in top):
        return None, f"best objective this iteration {best_in_iteration:.6g} reaches the top {best_policies}"

    candidate = random.choice(best_policies)
    candidate_objective = objectives[candidate]
    delta = candidate_objective - current_objective

    # drop the worst 20% from the spread: members that stopped improving would inflate it
    num_outliers = int(math.floor(0.2 * len(initialized)))
    trimmed = sorted(initialized)[num_outliers:] if len(initialized) > num_outliers else initialized
    std_threshold = cfg.replace_threshold_frac_std * float(np.std(trimmed))
    abs_threshold = cfg.replace_threshold_frac_absolute * abs(candidate_objective)
    if delta > std_threshold and delta > abs_threshold:
        return candidate, f"replace from {candidate}: lead {delta:.6g} > {std_threshold:.6g}, {abs_threshold:.6g}"
    return policy_idx, f"lead of {candidate} ({delta:.6g}) too small: mutate own parameters"


def select_threshold(objectives: dict[int, float | None], policy_idx: int, cfg) -> tuple[int | None, str]:
    """Band rule: underperformers restart from a random leader, or from themselves without leaders."""
    initialized = {p: o for p, o in objectives.items() if o is not None and o > UNINITIALIZED_VALUE}
    values = list(initialized.values())
    mean_obj = float(np.mean(values))
    std_obj = float(np.std(values))
    upper_cut = max(mean_obj + cfg.threshold_std * std_obj, mean_obj + cfg.threshold_abs)
    lower_cut = min(mean_obj - cfg.threshold_std * std_obj, mean_obj - cfg.threshold_abs)
    leaders = [p for p, o in initialized.items() if o > upper_cut]
    underperformers = [p for p, o in initialized.items() if o < lower_cut]
    print(f"mean={mean_obj:.4f}, std={std_obj:.4f}, upper={upper_cut:.4f}, lower={lower_cut:.4f}")
    print(f"Leaders: {leaders} Underperformers: {underperformers}")
    if policy_idx not in underperformers:
        return None, "not an underperformer"
    if not leaders:
        return policy_idx, "no leaders: mutate own parameters"
    return random.choice(leaders), "underperformer"


# --- restart protocol ------------------------------------------------------------------------

RESTART_ENV_VAR = "RL_GAMES_PBT_RESTART"
_restart_info_cache: dict | None = None
_restart_info_read = False


def restart_info() -> dict | None:
    """Transfer metadata when this process is a PBT restart, else None.

    Keys: policy_idx, source_policy, restart_count, iteration, frame, checkpoint and
    `mutated` ({flattened param: new value} for params that differ from the source's).
    Environments can read it in `set_env_state` to tell a population transfer from a
    resume. Parsed once per process; the variable is then removed from the environment.
    """
    global _restart_info_cache, _restart_info_read
    if not _restart_info_read:
        _restart_info_read = True
        raw = os.environ.pop(RESTART_ENV_VAR, None)
        _restart_info_cache = json.loads(raw) if raw else None
    return _restart_info_cache


def _reset_restart_info_cache() -> None:
    """Test helper: forget the parsed restart info."""
    global _restart_info_cache, _restart_info_read
    _restart_info_cache, _restart_info_read = None, False


def _arg_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def remove_cli_arg(args: list[str], name: str) -> list[str]:
    """Drop every occurrence of `name` (`name=value`, or `name value` for dash-prefixed flags)."""
    out, skip_next = [], False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if _arg_name(arg) == name:
            if "=" not in arg and name.startswith("-"):
                skip_next = True
            continue
        out.append(arg)
    return out


def get_cli_arg(args: list[str], name: str) -> str | None:
    """Last value of `name` in `args` (argparse semantics), or None."""
    value = None
    for i, arg in enumerate(args):
        if _arg_name(arg) != name:
            continue
        if "=" in arg:
            value = arg.split("=", 1)[1]
        elif name.startswith("-") and i + 1 < len(args):
            value = args[i + 1]
    return value


def derive_seed(base_seed: int, policy_idx: int, restart_count: int) -> int:
    """Deterministic per-restart seed; -1 (random per launch) stays -1."""
    if base_seed == -1:
        return -1
    return random.Random(f"{base_seed}:{policy_idx}:{restart_count}").randrange(2**31 - 1)


def build_launch_restart_args(launch_argv, new_params, restart_from_checkpoint, checkpoint_arg="--checkpoint"):
    """Restart command from the original launch argv: everything verbatim except the checkpoint
    argument and the mutated `key=value` overrides (exact key match)."""
    out = [launch_argv[0]]
    rest = remove_cli_arg(list(launch_argv[1:]), checkpoint_arg)
    out += [arg for arg in rest if not ("=" in arg and _arg_name(arg) in new_params)]
    out.append(f"{checkpoint_arg}={restart_from_checkpoint}")
    out += [f"{param}={value}" for param, value in new_params.items()]
    return out


def _render_table(headers: list[str], rows: list[list]) -> str:
    """Render a simple fixed-width text table (no external dependencies)."""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    out = [sep, line(headers), sep]
    out.extend(line(r) for r in str_rows)
    out.append(sep)
    return "\n".join(out)


class PbtTablePrinter:
    """Plain-text table rendering for PBT logs."""

    def __init__(self, *, float_digits: int = 6, path_maxlen: int = 52):
        self.float_digits = float_digits
        self.path_maxlen = path_maxlen

    # format helpers
    def fmt(self, v):
        return f"{v:.{self.float_digits}g}" if isinstance(v, float) else v

    def short(self, s: str) -> str:
        s = str(s)
        L = self.path_maxlen
        return s if len(s) <= L else s[: L // 2 - 1] + "…" + s[-L // 2 :]

    # tables
    def print_params_table(self, params: dict, header: str = "Parameters"):
        rows = [[k, self.fmt(params[k])] for k in sorted(params)]
        print(header + ":")
        print(_render_table(["Parameter", "Value"], rows))

    def print_ckpt_summary(self, sumry: dict[int, dict | None]):
        headers = ["Policy", "Status", "Objective", "Iter", "Frame", "Experiment", "Checkpoint", "YAML"]
        rows = []
        for p in sorted(sumry.keys()):
            c = sumry[p]
            if c is None:
                rows.append([p, "—", "", "", "", "", "", ""])
            else:
                rows.append([
                    p,
                    "OK",
                    self.fmt(c.get("true_objective", "")),
                    c.get("iteration", ""),
                    c.get("frame", ""),
                    c.get("experiment_name", ""),
                    self.short(c.get("checkpoint", "")),
                    self.short(c.get("pbt_checkpoint", "")),
                ])
        print(_render_table(headers, rows))

    def print_mutation_diff(self, before: dict, after: dict, *, header: str = "Mutated params (changed only)"):
        rows = [[k, self.fmt(before[k]), self.fmt(after[k])] for k in sorted(before) if before[k] != after[k]]
        print(header + ":")
        print(_render_table(["Parameter", "Old", "New"], rows) if rows else "(no changes)")
