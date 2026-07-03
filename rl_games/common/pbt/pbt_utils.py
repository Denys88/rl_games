# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os
import random
import socket
from collections import OrderedDict
from pathlib import Path

import yaml

from rl_games.algos_torch.torch_ext import safe_filesystem_op, safe_save


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


def save_pbt_checkpoint(workspace_dir, curr_policy_score, curr_iter, algo, params):
    """Save a PBT checkpoint (.pth and .yaml) with policy state, score, and metadata (rank 0 only)."""
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.pth")
        safe_save(algo.get_full_state_weights(), checkpoint_file)
        pbt_checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.yaml")

        pbt_checkpoint = {
            "iteration": curr_iter,
            "true_objective": curr_policy_score,
            "frame": algo.frame,
            "params": params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": algo.experiment_name,
        }

        with open(pbt_checkpoint_file, "w") as fobj:
            yaml.dump(pbt_checkpoint, fobj)


def load_pbt_ckpts(workspace_dir, cur_policy_id, num_policies, pbt_iteration) -> dict | None:
    """
    Load the latest available PBT checkpoint for each policy (≤ current iteration).
    Returns a dict mapping policy_idx → checkpoint dict or None. (rank 0 only)
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    checkpoints = dict()
    for policy_idx in range(num_policies):
        checkpoints[policy_idx] = None
        policy_dir = os.path.join(workspace_dir, f"{policy_idx:03d}")

        if not os.path.isdir(policy_dir):
            continue

        pbt_checkpoint_files = sorted([f for f in os.listdir(policy_dir) if f.endswith(".yaml")], reverse=True)
        for pbt_checkpoint_file in pbt_checkpoint_files:
            iteration = int(pbt_checkpoint_file.split(".")[0])

            ctime_ts = os.path.getctime(os.path.join(policy_dir, pbt_checkpoint_file))
            created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")

            if iteration <= pbt_iteration:
                with open(os.path.join(policy_dir, pbt_checkpoint_file)) as fobj:
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Policy {cur_policy_id} [{now_str}]: Loading"
                        f" policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})"
                    )
                    checkpoints[policy_idx] = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                    break

    return checkpoints


def cleanup(checkpoints: dict[int, dict], policy_dir, keep_back: int = 20, max_yaml: int = 50) -> None:
    """
    Cleanup old checkpoints for the current policy directory (rank 0 only).
    - Delete files older than (oldest iteration - keep_back).
    - Keep at most `max_yaml` latest YAML iterations.
    """
    if int(os.environ.get("RANK", "0")) == 0:
        oldest = min((ckpt["iteration"] if ckpt else 0) for ckpt in checkpoints.values())
        threshold = max(0, oldest - keep_back)
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
