# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import datetime
import json
import math
import os
import random
import re
import shutil
import socket
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from rl_games.algos_torch.torch_ext import safe_filesystem_op, safe_save

# i.e. value for target objective when it is not known
UNINITIALIZED_VALUE = float(-1e9)


class DistributedArgs:
    """Distributed-launch flags reconstructed for a PBT restart.

    The restart relaunches every rank of the member on rank 0's node with a standalone
    rendezvous: torchrun binds a free port itself (a port picked in advance can be taken
    before torchrun binds it). Other flags of the original torchrun command are not kept.
    All attribute lookups are defaulted so any argparse namespace (or None) works.
    """

    def __init__(self, args_cli):
        self.distributed = getattr(args_cli, "distributed", False)
        self.nproc_per_node = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.nnodes = 1

    def get_args_list(self) -> list[str]:
        return ["-m", "torch.distributed.run", "--standalone", f"--nnodes={self.nnodes}",
                f"--nproc_per_node={self.nproc_per_node}"]


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
    """Print the size of the environment a restart passes on: variable count, bytes, the five
    largest variables and SC_ARG_MAX (execv fails with E2BIG past it).

    Names and lengths only, never values: they hold credentials (WANDB_API_KEY, cloud
    session tokens) and member logs are widely shared.
    """

    n = len(os.environ)
    # total bytes in "KEY=VAL\0" for all envp entries
    total = sum(len(k) + 1 + len(v) + 1 for k, v in os.environ.items())
    # find the 5 largest values
    biggest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]

    print(f"[ENV MONITOR] vars={n}, total_bytes={total}")
    for k, v in biggest:
        print(f"    {k!r} length={len(v)}")

    try:
        argmax = os.sysconf("SC_ARG_MAX")
        print(f"[ENV MONITOR] SC_ARG_MAX = {argmax}")
    except (ValueError, AttributeError, OSError):
        pass


def dedup_path_list(value: str) -> str:
    """A PATH-like list without repeated entries, first occurrences kept in order.

    The first empty entry is kept too: it stands for the current directory (what
    `export PYTHONPATH=$PYTHONPATH:/x` leaves when the variable was unset). A list of empty
    entries only (`:`) stays one separator, since an empty value has no entry at all.
    """
    seen, parts = set(), []
    for part in value.split(os.pathsep):
        if part not in seen:
            seen.add(part)
            parts.append(part)
    deduped = os.pathsep.join(parts)
    return deduped if deduped or not value else os.pathsep


def flatten_dict(d, prefix="", separator="."):
    """Flatten nested dictionaries into a flat dict with keys joined by `separator`."""

    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


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


def plain_number(value):
    """A numpy or torch scalar as the Python number it holds; anything else unchanged.

    Custom mutation functions may return numpy scalars (np.clip gives np.int64): the restart
    passes plain numbers on the command line and in the JSON payload.
    """
    if type(value) in (bool, int, float, str) or not callable(getattr(value, "item", None)):
        return value
    try:
        number = value.item()
    except (TypeError, ValueError, RuntimeError):
        return value
    return number if isinstance(number, (bool, int, float)) else value


def json_default(value):
    """`json.dumps` default: numpy and torch scalars as Python numbers, anything else a TypeError."""
    number = plain_number(value)
    if number is value:
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    return number


def episode_stats(value, done_indices, num_agents: int = 1) -> tuple[float, int, int]:
    """Objective of the episodes that finished on one step: (sum, count) over the episodes with
    a finite value, and the number of finished episodes whose value is NaN or infinite.

    `done_indices` are agent indices (env * num_agents), one per finished env, as rl_games
    passes them to observers. A per-env tensor/array is read at those envs, and a non-finite
    entry drops only its own episode: one diverged env must not discard the episodes the
    other envs finished on the same step. A scalar is taken as the mean over the envs that
    finished on this step (Isaac Lab episode logs), so it is weighted by their count, and a
    non-finite one drops all of them. Without `done_indices` every element is one sample.
    """
    n_done = None
    if done_indices is not None:
        idx = done_indices.reshape(-1) if torch.is_tensor(done_indices) else np.asarray(done_indices).reshape(-1)
        n_done = int(idx.shape[0])
    per_env = (torch.is_tensor(value) and value.numel() > 1) or (isinstance(value, np.ndarray) and value.size > 1)
    if per_env:
        if n_done == 0:
            return 0.0, 0, 0
        picked = value.reshape(-1)
        if n_done is not None:
            size = int(picked.shape[0])
            env_idx = torch.as_tensor(idx).cpu().long() // max(1, int(num_agents))
            if int(env_idx.min()) < 0 or int(env_idx.max()) >= size:
                # checked on the host: an out-of-range CUDA index is a device-side assert
                raise IndexError(f"done env index {int(env_idx.max())} out of range for a per-env objective "
                                 f"of size {size}")
            picked = picked[env_idx.to(picked.device)] if torch.is_tensor(picked) else picked[env_idx.numpy()]
        if torch.is_tensor(picked):
            # the picked values to the host (one sync) before widening them: MPS has no float64
            picked = picked.detach().cpu().double().numpy()
        picked = np.asarray(picked, dtype=np.float64)
        finite = np.isfinite(picked)
        count = int(finite.sum())
        return float(picked[finite].sum()), count, int(picked.shape[0]) - count
    scalar = to_float(value)
    episodes = 1 if n_done is None else n_done
    if not math.isfinite(scalar):
        return 0.0, 0, episodes
    return scalar * episodes, episodes, 0


def episode_sum_count(value, done_indices, num_agents: int = 1) -> tuple[float, int]:
    """(sum, count) over the finished episodes with a finite objective, see `episode_stats`."""
    total, count, _ = episode_stats(value, done_indices, num_agents)
    return total, count


def _read_text_and_ctime(path):
    """(text, ctime) of a file, or None when it is gone: members delete their old files while
    others read them. Bytes that are not UTF-8 are replaced (the file is malformed, not worth a
    retry)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fobj:
            return fobj.read(), os.fstat(fobj.fileno()).st_ctime
    except FileNotFoundError:
        return None


def _is_number(value) -> bool:
    """A finite int or float, as plain YAML loads one (a bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or math.isfinite(value)


_TMP_MARK = ".tmp."
_STALE_TMP_SECONDS = 3600


def _tmp_path(path, tag="") -> str:
    """Temp name unique per writer: members may run in containers sharing a filesystem."""
    return f"{path}{_TMP_MARK}{tag}{'.' if tag else ''}{socket.gethostname()}.{os.getpid()}"


def _remove_stale_tmp(directory, max_age=_STALE_TMP_SECONDS) -> None:
    """Drop temp files left by writers that died between write and rename."""
    now = time.time()
    for name in os.listdir(directory):
        if _TMP_MARK in name:
            path = os.path.join(directory, name)
            try:
                if now - os.path.getmtime(path) > max_age:
                    os.remove(path)
            except FileNotFoundError:
                pass


def _atomic_write_text(path, text, tag="") -> None:
    tmp = _tmp_path(path, tag)
    with open(tmp, "w", encoding="utf-8") as fobj:
        fobj.write(text)
    os.replace(tmp, path)


def _atomic_write_yaml(path, data, tag="") -> None:
    """Write `data` as plain YAML (`yaml.safe_dump`), atomically. A value plain YAML can't hold
    (a numpy scalar, a tensor) raises before any file exists, instead of leaving a file that no
    reader loads."""
    _atomic_write_text(path, yaml.safe_dump(data), tag)


def _atomic_torch_save(state, path, tag="") -> None:
    tmp = _tmp_path(path, tag)
    safe_save(state, tmp)
    os.replace(tmp, path)


_CKPT_KEYS = ("iteration", "true_objective", "frame", "params", "checkpoint")
_ITERATION_STEM = re.compile(r"[0-9]+")


def _valid_pbt_checkpoint(ckpt) -> bool:
    """Whether a loaded workspace yaml has every key with a usable type. The ranking and the
    mutation run outside the iteration's error handling: a wrong type there stops the member."""
    if not isinstance(ckpt, dict) or any(k not in ckpt for k in _CKPT_KEYS):
        return False
    iteration, params = ckpt["iteration"], ckpt["params"]
    return (isinstance(iteration, int) and not isinstance(iteration, bool)
            and _is_number(ckpt["true_objective"]) and _is_number(ckpt["frame"])
            and isinstance(ckpt["checkpoint"], str) and isinstance(params, dict)
            and all(isinstance(key, str) and _is_number(value) for key, value in params.items()))


def _numbered_files(policy_dir) -> dict[int, list[Path]]:
    """A member directory's iteration files (`<iteration>.yaml`, `<iteration>.pth`) by iteration."""
    groups: dict[int, list[Path]] = {}
    for p in Path(policy_dir).iterdir():
        if p.suffix in (".yaml", ".pth") and _ITERATION_STEM.fullmatch(p.stem):
            groups.setdefault(int(p.stem), []).append(p)
    return groups


def _unlink_iteration(paths) -> None:
    """Delete one iteration's files, the .yaml first: readers list the yamls, so they never pick
    an iteration whose checkpoint is already gone."""
    for p in sorted(paths, key=lambda p: p.suffix != ".yaml"):
        p.unlink(missing_ok=True)


def saved_iterations(policy_dir) -> list[int]:
    """The iterations a member directory holds a workspace yaml for, ascending."""
    return sorted(it for it, paths in _numbered_files(policy_dir).items() if any(p.suffix == ".yaml" for p in paths))


def remove_iterations_above(policy_dir, iteration) -> list[int]:
    """Delete a member's iteration files above `iteration`, each .yaml before its .pth; returns
    the iterations removed. They belong to a run the member doesn't continue (an earlier run, or
    the future of a relaunch from an older checkpoint): peers would read them as its current state."""
    groups = _numbered_files(policy_dir)
    above = sorted(it for it in groups if it > iteration)
    for it in above:
        _unlink_iteration(groups[it])
    return above


# the networks of a learner checkpoint: the actor-critic with its normalizers, the central value net
_NETWORK_KEYS = ("model", "assymetric_vf_nets")


def _check_finite_weights(state, what) -> None:
    """Raise ValueError when a network in learner checkpoint `state` holds NaN or inf.

    A diverged policy must not reach the workspace or a restart: a peer would copy it as its
    donor, and a NaN policy trains on silently. Floating-point tensors only (integer counters
    hold no NaN).
    """
    bad = []
    for key in _NETWORK_KEYS:
        value = state.get(key) if isinstance(state, Mapping) else None
        for name, tensor in (value.items() if isinstance(value, Mapping) else [(None, value)]):
            if torch.is_tensor(tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                bad.append(key if name is None else f"{key}.{name}")
    if bad:
        more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
        raise ValueError(f"{what}: NaN or inf in {', '.join(bad[:3])}{more} (a diverged policy)")


def save_pbt_checkpoint(workspace_dir, curr_policy_score, curr_iter, algo, params, *, replaceable=True):
    """Save a PBT checkpoint (.pth and .yaml) with policy state, score, and metadata (rank 0 only).

    The yaml is plain YAML: a value it can't hold plainly (a numpy scalar from a custom mutation
    function) raises before anything is written. A state whose networks hold NaN or inf raises
    ValueError before anything is written too, so the member's newest checkpoint stays its last
    finite one. Both files are written to a temp name and renamed, the .pth first, so readers
    never see partial files or a yaml without its checkpoint. `replaceable: false` keeps the
    member out of its peers' worst group while it can't act on a replacement (see `member_status`).
    """
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.pth")
        pbt_checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.yaml")
        text = yaml.safe_dump({
            "iteration": curr_iter,
            "true_objective": to_float(curr_policy_score),
            "frame": int(algo.frame),
            "params": params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": algo.experiment_name,
            "replaceable": bool(replaceable),
        })
        state = algo.get_full_state_weights()
        if "epoch" in state:
            # called in after_steps, after update_epoch and before the epoch's update and frame
            # increment: the weights and frame belong to the last completed epoch
            state["epoch"] = int(algo.epoch_num) - 1
        _check_finite_weights(state, f"PBT checkpoint {checkpoint_file}")
        _atomic_torch_save(state, checkpoint_file)
        _atomic_write_text(pbt_checkpoint_file, text)


def load_pbt_ckpts(workspace_dir, cur_policy_id, num_policies, pbt_iteration) -> dict | None:
    """
    Load the latest available PBT checkpoint for each policy (≤ current iteration).
    Returns a dict mapping policy_idx → checkpoint dict or None. (rank 0 only)

    Each member directory is listed first, then its yamls are read newest first. A yaml that is
    gone by then (its owner's cleanup) or malformed (not plain YAML, a key missing or of the
    wrong type) is skipped for the next older one, without retrying. `checkpoint` is set to the
    .pth next to the yaml when that exists, so a workspace mounted at another path, or moved,
    still works; else the stored path stays. `pbt_checkpoint` is the yaml read.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    checkpoints = dict()
    for policy_idx in range(num_policies):
        checkpoints[policy_idx] = None
        policy_dir = os.path.join(workspace_dir, f"{policy_idx:03d}")

        if not os.path.isdir(policy_dir):
            continue

        candidates = []
        for name in os.listdir(policy_dir):
            stem, ext = os.path.splitext(name)
            if ext == ".yaml" and _ITERATION_STEM.fullmatch(stem) and int(stem) <= pbt_iteration:
                candidates.append((int(stem), name))
        for _, pbt_checkpoint_file in sorted(candidates, reverse=True):
            path = os.path.join(policy_dir, pbt_checkpoint_file)
            loaded = safe_filesystem_op(_read_text_and_ctime, path)
            if loaded is None:
                print(f"Policy {cur_policy_id}: {path} is gone (cleaned up by its owner); trying an older one")
                continue
            text, ctime_ts = loaded
            created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"Policy {cur_policy_id} [{now_str}]: Loading"
                f" policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})"
            )
            try:
                ckpt = yaml.safe_load(text)
            except yaml.YAMLError:
                ckpt = None
            if not _valid_pbt_checkpoint(ckpt):
                print(f"Policy {cur_policy_id}: skipping malformed PBT checkpoint {path}")
                continue
            local_checkpoint = os.path.join(policy_dir, os.path.splitext(pbt_checkpoint_file)[0] + ".pth")
            if os.path.isfile(local_checkpoint):
                ckpt["checkpoint"] = os.path.abspath(local_checkpoint)
            ckpt["pbt_checkpoint"] = os.path.abspath(path)
            checkpoints[policy_idx] = ckpt
            break

    return checkpoints


def mark_not_replaceable(policy_dir) -> str | None:
    """Rewrite this member's newest workspace yaml with `replaceable: false` (temp + rename) and
    return its path; None without a readable one. For a member that decided to restart: peers
    read that yaml as its state until the restarted process saves, and the member can't act on
    a replacement meanwhile. Only the member writes its own yamls."""
    yamls = {it: p for it, paths in _numbered_files(policy_dir).items() for p in paths if p.suffix == ".yaml"}
    if not yamls:
        return None
    path = str(yamls[max(yamls)])
    with open(path, encoding="utf-8") as fobj:
        data = yaml.safe_load(fobj)
    if not isinstance(data, dict):
        return None
    data["replaceable"] = False
    _atomic_write_yaml(path, data)
    return path


# a member's status in a reader's view (`member_status`); only REPLACEABLE members take worst slots
REPLACEABLE, NOT_REPLACEABLE, BEHIND, NOT_REPORTED = "OK", "not replaceable", "behind", "—"


def member_status(checkpoints: dict[int, dict | None], reader_iteration: int) -> dict[int, str]:
    """Each member's status as the reader of `checkpoints` (`load_pbt_ckpts` at `reader_iteration`)
    sees it, for the replacement rule and the checkpoint summary:

    - "—": no checkpoint with an objective at or below the reader's iteration;
    - "not replaceable": its newest yaml says `replaceable: false` (restarting, in a grace window
      or near the end of its run);
    - "behind": its newest yaml is 2 or more iterations behind the reader's (a restart starting
      up, a stalled member);
    - "OK": replaceable. A yaml without the field (1e6966a) is replaceable.
    """
    status = {}
    for policy, ckpt in checkpoints.items():
        ckpt = ckpt if isinstance(ckpt, Mapping) else {}
        objective, iteration = ckpt.get("true_objective"), ckpt.get("iteration")
        if not _is_number(objective) or objective <= UNINITIALIZED_VALUE:
            status[policy] = NOT_REPORTED
        elif ckpt.get("replaceable") is False:
            status[policy] = NOT_REPLACEABLE
        elif isinstance(iteration, int) and iteration < reader_iteration - 1:
            status[policy] = BEHIND
        else:
            status[policy] = REPLACEABLE
    return status


def replaceable_members(checkpoints: dict[int, dict | None], reader_iteration: int) -> set[int]:
    """The members that may take a worst slot in this reader's view: status "OK" in `member_status`."""
    return {policy for policy, status in member_status(checkpoints, reader_iteration).items()
            if status == REPLACEABLE}


_KEEP_NEWEST = 10  # cleanup never thins a member's newest iterations (IsaacGymEnvs)


def cleanup(checkpoints: dict[int, dict], policy_dir, keep_back: int = 20, max_yaml: int = 50) -> None:
    """
    Cleanup old checkpoints of this member (rank 0 only), as IsaacGymEnvs' DexPBT does.
    - Delete files at or below (slowest member's iteration - keep_back), where a member without a
      checkpoint counts as iteration 0, except the newest yaml at or below the slowest member's
      iteration. A live member however far behind keeps finding this member at or below its own
      iteration.
    - Above `max_yaml` iterations, thin the older ones, never the newest 10 or the oldest: delete
      the iteration whose removal leaves the smallest gap, until max(max_yaml, 12) are left. Disk
      use stays bounded when a member stops, and a laggard still finds a file below it.
    Each iteration's .yaml is deleted before its .pth.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return
    slowest = min((ckpt["iteration"] if ckpt else 0 for ckpt in checkpoints.values()), default=0)
    threshold = slowest - keep_back
    _remove_stale_tmp(policy_dir)
    groups = _numbered_files(policy_dir)
    yaml_iters = sorted(it for it, paths in groups.items() if any(p.suffix == ".yaml" for p in paths))

    # 1) what every member is past: keep the file the slowest member reads, though
    readable = max((it for it in yaml_iters if it <= slowest), default=None)
    for it in sorted(it for it in groups if it <= threshold and it != readable):
        _unlink_iteration(groups.pop(it))
    yaml_iters = [it for it in yaml_iters if it in groups]

    # 2) thin the old ones so that a stopped member doesn't make the directory grow
    while len(yaml_iters) > max_yaml:
        candidates = yaml_iters[:-_KEEP_NEWEST]
        if len(candidates) < 3:
            break
        # the inner candidate whose removal leaves the smallest gap (the oldest on ties)
        k = min(range(1, len(candidates) - 1), key=lambda j: candidates[j + 1] - candidates[j - 1])
        _unlink_iteration(groups.pop(candidates[k]))
        yaml_iters.remove(candidates[k])


def prepare_restart_checkpoint(source_checkpoint, policy_dir, frame, history_entry, epoch=None, *,
                               config_hparams=None, last_mean_rewards=None) -> str:
    """Copy the source checkpoint into this member's workspace for the restart.

    The copy survives the source's cleanup, keeps this member's frame (DexPBT) and epoch
    counters, and appends `history_entry` to the checkpoint's `pbt_history`. When given,
    `config_hparams` replaces the stored configured values key by key (the restart's
    pre-mutation values: the restore then moves an inherited adaptive LR by this restart's
    mutation factor only), and `last_mean_rewards` the best-reward watermark (the member's
    own, when it keeps writing into the same run directory). A source whose networks hold NaN
    or inf (published by an older member) raises ValueError, and nothing is written.
    """
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    _check_finite_weights(state, f"restart source {source_checkpoint}")
    state["frame"] = int(frame)
    if epoch is not None:
        state["epoch"] = int(epoch)
    if config_hparams:
        stored = dict(state.get("config_hparams") or {})
        stored.update({key: float(value) for key, value in config_hparams.items()})
        state["config_hparams"] = stored
    if last_mean_rewards is not None:
        state["last_mean_rewards"] = float(last_mean_rewards)
    state["pbt_history"] = list(state.get("pbt_history", [])) + [history_entry]
    restart_checkpoint = os.path.join(policy_dir, "restart.pth")
    _atomic_torch_save(state, restart_checkpoint)
    return os.path.abspath(restart_checkpoint)


# <workspace>/best: per-member records `p<idx>_<it>.yaml` naming copies `best_p<idx>_<it>.pth`,
# each written once by its member; `best.yaml` is a view with the best record's content
_BEST_VIEW = "best.yaml"
_BEST_RECORD = re.compile(r"p([0-9]+)_([0-9]+)\.yaml")
_BEST_COPY = re.compile(r"best_p([0-9]+)_([0-9]+)\.pth")


def _best_record(text, best_dir) -> dict | None:
    """A best record (or the view) loaded from `text`; None when malformed. `checkpoint` becomes
    the copy of that name in `best_dir` when it exists (the workspace may be mounted elsewhere)."""
    try:
        record = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not (isinstance(record, dict) and _is_number(record.get("true_objective"))
            and isinstance(record.get("checkpoint"), str)):
        return None
    local_copy = os.path.join(best_dir, os.path.basename(record["checkpoint"]))
    if os.path.isfile(local_copy):
        record["checkpoint"] = os.path.abspath(local_copy)
    return record


def _best_records(best_dir) -> tuple[dict[str, dict], set[str]]:
    """(readable records by file name, names listed but gone) in `best_dir`.

    The view `best.yaml` counts as a record too: it holds one, and it is the only record 1e6966a
    wrote. The directory is listed before any file is opened. A listed file that is gone was
    replaced by its owner's newer record, so the directory is listed once more for that one:
    a missing file is skipped, never taken as "no best".
    """
    records, seen, gone = {}, set(), set()
    for _ in range(2):
        try:
            names = os.listdir(best_dir)
        except FileNotFoundError:
            break
        gone = set()
        for name in names:
            if name in seen or not (name == _BEST_VIEW or _BEST_RECORD.fullmatch(name)):
                continue
            loaded = safe_filesystem_op(_read_text_and_ctime, os.path.join(best_dir, name))
            if loaded is None:
                gone.add(name)
                continue
            seen.add(name)
            record = _best_record(loaded[0], best_dir)
            if record is not None:
                records[name] = record
        if not gone:
            break
    return records, gone


def _top_record(records) -> dict | None:
    """The highest-objective record whose checkpoint copy exists (on a tie, a record before the view)."""
    present = [(record["true_objective"], name != _BEST_VIEW, name) for name, record in records.items()
               if os.path.isfile(record["checkpoint"])]
    return records[max(present)[2]] if present else None


def population_best(workspace_dir) -> dict | None:
    """The best record in <workspace>/best: the highest objective whose checkpoint copy exists, or
    None. A dict with the keys of `best.yaml` (policy_idx, iteration, frame, true_objective, params,
    checkpoint, source_checkpoint); `checkpoint` is the copy's path on this host's mount.

    Authoritative: `best.yaml` is a view of it, rebuilt by several members, that can lag behind
    for a while on a shared filesystem.
    """
    return _top_record(_best_records(os.path.join(workspace_dir, "best"))[0])


def _rebuild_best_view(best_dir, tag) -> tuple[dict[str, dict], set[str]]:
    """Point `best.yaml` at the best record; returns the last listing (see `_best_records`).

    Other members rebuild it too, and a view built from an older listing may land after this
    one: list again and rewrite until the view holds the best record (a few rounds). A view that
    is listed but can't be read stays: it may hold a record (1e6966a's) that exists nowhere else.
    """
    for attempt in range(4):
        records, gone = _best_records(best_dir)
        top = _top_record(records)
        if attempt == 3 or top is None or _BEST_VIEW in gone or records.get(_BEST_VIEW) == top:
            return records, gone
        _atomic_write_yaml(os.path.join(best_dir, _BEST_VIEW), top, tag=tag)


def _drop_old_best(best_dir, policy_idx, records, gone) -> None:
    """Delete this member's best records and copies, except its two best records (the new one,
    and the previous one, which a view written concurrently may still name) and the copy the
    view names. Records go before their copies. A copy without a record (1e6966a's, or a write
    cut short) only goes when the view could be read."""
    own = {}
    for name, record in records.items():
        match = _BEST_RECORD.fullmatch(name)
        if match and int(match.group(1)) == policy_idx:
            # a member's records grow in objective as it writes them, also across a relaunch
            # that numbers its iterations lower again
            own[name[:-len(".yaml")]] = (record["true_objective"], int(match.group(2)))
    keep = set(sorted(own, key=own.get)[-2:])
    view = records.get(_BEST_VIEW)
    if view is not None:
        keep.add(os.path.basename(view["checkpoint"])[len("best_"):-len(".pth")])
    stale_records, stale_copies = [], []
    for name in os.listdir(best_dir):
        record_match, copy_match = _BEST_RECORD.fullmatch(name), _BEST_COPY.fullmatch(name)
        if record_match and int(record_match.group(1)) == policy_idx and name[:-len(".yaml")] not in keep:
            stale_records.append(name)
        elif copy_match and int(copy_match.group(1)) == policy_idx:
            stem = name[len("best_"):-len(".pth")]
            if stem not in keep and (stem in own or _BEST_VIEW not in gone):
                stale_copies.append(name)
    for name in stale_records + stale_copies:
        try:
            os.remove(os.path.join(best_dir, name))
        except FileNotFoundError:
            pass


def maybe_save_best(workspace_dir, policy_idx, objective, iteration, frame, checkpoint, params) -> bool:
    """Record a new population best in <workspace>/best (called by the population-best member).

    Returns whether `objective` beat `population_best()` and was recorded. The member writes its
    own files under new names: the copy `best_p<idx>_<it>.pth`, then the record `p<idx>_<it>.yaml`
    naming it (each to a temp name, then renamed), so no member replaces another's file and a
    listed record's copy exists. Then it rebuilds the view `best.yaml` and deletes its own older
    records and copies (see `_drop_old_best`).
    """
    best_dir = os.path.join(workspace_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    _remove_stale_tmp(best_dir)
    objective = to_float(objective)
    current = population_best(workspace_dir)
    if not math.isfinite(objective) or (current is not None and current["true_objective"] >= objective):
        return False
    tag = f"p{policy_idx:03d}"
    stem = f"{tag}_{iteration:06d}"
    best_checkpoint = os.path.join(best_dir, f"best_{stem}.pth")
    # a value plain YAML can't hold raises here, before any file is written
    text = yaml.safe_dump({
        "policy_idx": policy_idx, "iteration": iteration, "frame": int(frame),
        "true_objective": objective, "params": params,
        "checkpoint": os.path.abspath(best_checkpoint), "source_checkpoint": os.path.abspath(checkpoint),
    })
    tmp = _tmp_path(best_checkpoint, tag)
    shutil.copyfile(checkpoint, tmp)
    os.replace(tmp, best_checkpoint)
    _atomic_write_text(os.path.join(best_dir, f"{stem}.yaml"), text, tag=tag)
    records, gone = _rebuild_best_view(best_dir, tag)
    _drop_old_best(best_dir, policy_idx, records, gone)
    return True


def tie_break_order(workspace: str, iteration: int, num_policies: int) -> list[int]:
    """The member indices shuffled by a generator seeded with (workspace name, iteration): every
    reader at `iteration` ranks equal objectives in this order, and the next iteration reorders
    them, so a tie doesn't pin one member to the worst slot or the population best. A local
    generator: the global `random` stream is untouched."""
    order = list(range(num_policies))
    random.Random(f"{workspace}:{iteration}").shuffle(order)
    return order


def rank_members(objectives: dict[int, float | None], order: list[int] | None = None) -> list[int]:
    """Members from the best objective to the worst, those without one (None) last. Equal
    objectives keep their order in `order` (a permutation of the members, see `tie_break_order`);
    without it the higher index ranks first, as IsaacGymEnvs sorts."""
    members = list(order) if order is not None else sorted(objectives, reverse=True)

    def value(policy):
        objective = objectives.get(policy)
        return UNINITIALIZED_VALUE if objective is None else objective

    return sorted(members, key=value, reverse=True)


def select_dexpbt(objectives: dict[int, float | None], policy_idx: int, current_objective: float,
                  best_in_iteration: float | None, cfg, *, eligible=None, order=None) -> tuple[int | None, str]:
    """DexPBT replacement rule.

    The members are ranked by objective (see `rank_members`; `order` breaks ties). The worst
    group is the last ceil(replace_fraction_worst * n) members of the ranking that reported and
    are in `eligible` (every member that reported when None; see `replaceable_members`):
    IsaacGymEnvs takes the last of the whole ranking, where a member that never reported, or
    one that can't act on a replacement now, holds a slot a member that can act needs. Every
    member that reported stays ranked, counts for the `> max(2, n // 2)` gate and can be the
    donor.

    Returns (source, reason): None keeps training; `policy_idx` restarts from its own
    checkpoint with mutated parameters; any other index restarts from that member.
    """
    n = cfg.num_policies
    values = {p: UNINITIALIZED_VALUE if objectives.get(p) is None else objectives[p] for p in range(n)}
    ranked_policies = rank_members(values, order)
    reported = [p for p in ranked_policies if values[p] > UNINITIALIZED_VALUE]
    initialized = [values[p] for p in reported]
    replaceable = [p for p in reported if eligible is None or p in eligible]

    replace_worst = math.ceil(cfg.replace_fraction_worst * n)
    replace_best = math.ceil(cfg.replace_fraction_best * n)
    best_policies = [p for p in ranked_policies[:replace_best] if values[p] > UNINITIALIZED_VALUE]
    worst_policies = replaceable[-replace_worst:]

    if policy_idx not in worst_policies:
        return None, f"not among the worst {worst_policies}"
    if len(initialized) <= max(2, n // 2) or not best_policies:
        return None, f"only {len(initialized)} of {n} members reported"
    if best_in_iteration is not None and best_in_iteration >= min(values[p] for p in best_policies):
        return None, f"best objective this iteration {best_in_iteration:.6g} reaches the top {best_policies}"

    candidate = random.choice(best_policies)
    candidate_objective = objectives[candidate]
    delta = candidate_objective - current_objective
    # members ranked below this one that can't be replaced now: it may sit mid-pack
    below = [p for p in reported[reported.index(policy_idx) + 1:] if p not in replaceable]
    note = f" (the worst replaceable member; {below} rank lower)" if below else ""

    # drop the worst 20% from the spread: members that stopped improving would inflate it
    num_outliers = int(math.floor(0.2 * len(initialized)))
    trimmed = sorted(initialized)[num_outliers:] if len(initialized) > num_outliers else initialized
    std_threshold = cfg.replace_threshold_frac_std * float(np.std(trimmed))
    abs_threshold = cfg.replace_threshold_frac_absolute * abs(candidate_objective)
    if delta > std_threshold and delta > abs_threshold:
        return candidate, (f"replace from {candidate}: lead {delta:.6g} > {std_threshold:.6g}, "
                           f"{abs_threshold:.6g}{note}")
    return policy_idx, f"lead of {candidate} ({delta:.6g}) too small: mutate own parameters{note}"


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


def _random_per_launch(seed) -> bool:
    """None or -1: the run takes a new random seed at every launch."""
    try:
        return seed is None or int(seed) == -1
    except (TypeError, ValueError):
        return False


def draw_rng(seed, policy_idx: int, restart_count: int, initial_frame: int) -> random.Random:
    """The generator of a member's PBT draws (donor pick, leader-params coin, mutations).

    Seeded from (seed, member, restart count, frame the process started at): the draws differ
    across members, restarts and resumes, and repeat for the same ones. A seed of None or -1
    (random per launch) takes OS entropy instead. Train scripts seed the global `random` with
    the config seed at every start: drawn from it, every restart without a seed on its command
    line would draw the same donor and factors, and so would every member launched with the
    same seed.
    """
    if _random_per_launch(seed):
        seed = os.urandom(16).hex()
    return random.Random(f"pbt-draws:{seed}:{policy_idx}:{restart_count}:{initial_frame}")


@contextlib.contextmanager
def swap_random_state(rng: random.Random):
    """Run the block with `rng`'s state in the global `random` module, then store the state the
    block left back into `rng` and restore the global one: code that calls `random` directly
    (select_*, mutate(), custom MUTATION_FUNCS, subclass overrides) draws from `rng`, and the
    global stream stays where it was."""
    outer = random.getstate()
    random.setstate(rng.getstate())
    try:
        yield rng
    finally:
        rng.setstate(random.getstate())
        random.setstate(outer)


def member_seed(seed, policy_idx: int, stride: int = 1_000_000):
    """Launch seed of population member `policy_idx`: `seed + policy_idx * stride`; -1 (random
    per launch) and None are returned unchanged.

    Members launched with the same seed start as copies (initial weights, env and sampling
    streams), so the train script launches each member with `member_seed(seed, policy_idx)`.
    Apply it to the seed the run uses, after the command-line override, e.g.
    `params['seed'] = member_seed(params['seed'], policy_idx)`, not to an optional `--seed`
    that defaults to None: a None seed leaves rl_games' Runner to seed with `int(time.time())`,
    so members launched in the same second start as copies.
    With `multi_gpu`, rl_games adds the global rank to the seed itself; scripts that add it
    as well get `seed + 2 * rank`, so `stride` must exceed 2 * (GPUs per member - 1).
    """
    if seed is None:
        return None
    seed = int(seed)
    return seed if seed == -1 else seed + int(policy_idx) * int(stride)


def build_launch_restart_args(launch_argv, new_params, restart_from_checkpoint, checkpoint_arg="--checkpoint"):
    """Restart command from the original launch argv: everything verbatim except the checkpoint
    argument and the mutated `key=value` overrides (exact key match)."""
    out = [launch_argv[0]]
    rest = remove_cli_arg(list(launch_argv[1:]), checkpoint_arg)
    out += [arg for arg in rest if not ("=" in arg and _arg_name(arg) in new_params)]
    out.append(f"{checkpoint_arg}={restart_from_checkpoint}")
    out += [f"{param}={value}" for param, value in new_params.items()]
    return out


def argv_rewritten() -> bool:
    """Whether `sys.argv` is no longer the tail of `sys.orig_argv`: the script replaced its
    command line (Isaac Lab style `sys.argv = [sys.argv[0]] + hydra_args`), or it runs as
    `python -m` (`sys.argv[0]` is then the module's file). False when the command is unknown."""
    orig = list(getattr(sys, "orig_argv", None) or [])
    if not orig:
        return False
    argv = list(sys.argv)
    return len(argv) > len(orig) or orig[len(orig) - len(argv):] != argv


def launch_dirs() -> tuple[str, str | None]:
    """(directory the job was launched from, Hydra run dir a restart must reuse, or None).

    Hydra with 1.1 job semantics (`version_base` omitted or '1.1', or `hydra.job.chdir=True`)
    runs the job inside its run dir. A restart then starts from the launch directory, where
    the relative paths of the command point, and passes that run dir on: the child would
    otherwise nest a new run dir inside it and move a relative train_dir and its workspace.
    Without a Hydra job this is (os.getcwd(), None); Hydra is not imported for the check.
    A best-effort probe: whatever Hydra raises here is not a reason to stop the member.
    """
    cwd = os.getcwd()
    if "hydra" not in sys.modules:
        return cwd, None
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd

        if not HydraConfig.initialized():
            return cwd, None
        original = get_original_cwd()
    except Exception:
        return cwd, None
    try:
        run_dir = os.path.join(original, HydraConfig.get().runtime.output_dir)
    except Exception:
        # Hydra < 1.2 has no runtime.output_dir and always runs the job in its run dir; a config
        # set by hand (compose API, HydraConfig.instance().set_config) leaves it unset (omegaconf
        # raises MissingMandatoryValue): no job ran, and the cwd is normally the original one
        run_dir = cwd
    changed = os.path.realpath(run_dir) == os.path.realpath(cwd) != os.path.realpath(original)
    return original, (os.path.abspath(run_dir) if changed else None)


def absolute_script_path(args: list[str], launch_cwd: str) -> list[str]:
    """`args` with the script (`args[0]`) as an absolute path when it is a relative path that
    exists under `launch_cwd`; a module launch (`-m pkg.mod`) and anything else stay as given."""
    if not args or args[0] == "-m" or os.path.isabs(args[0]):
        return list(args)
    path = os.path.join(launch_cwd, args[0])
    return [os.path.abspath(path)] + list(args[1:]) if os.path.exists(path) else list(args)


_HYDRA_PLAIN_VALUE = re.compile(r"[A-Za-z0-9_/.\-+@%*?|:]+")


def hydra_override_value(value: str) -> str:
    """`value` written for a Hydra `key=value` override: as is when the override grammar takes it
    unquoted, else single-quoted (spaces, parentheses, `=`, `,`, brackets, quotes...)."""
    if _HYDRA_PLAIN_VALUE.fullmatch(value):
        return value
    # in a quoted value a backslash escapes only a quote or the closing quote
    escaped = re.sub(r"(\\+)(?='|\Z)", r"\1\1", value).replace("'", "\\'")
    return f"'{escaped}'"


def pin_hydra_run_dir(args: list[str], run_dir: str) -> list[str]:
    """`args` with every `hydra.run.dir` override replaced by one naming `run_dir`, so the
    restarted Hydra job reuses the run dir instead of nesting a new one (as IsaacGymEnvs did)."""
    kept = [arg for arg in args[1:] if _arg_name(arg).lstrip("+~") != "hydra.run.dir"]
    return [args[0]] + kept + [f"hydra.run.dir={hydra_override_value(run_dir)}"]


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

    def print_ckpt_summary(self, sumry: dict[int, dict | None], *, status: dict[int, str] | None = None):
        """One row per member; `status` fills the Status column (see `member_status`; default: OK,
        or — without a checkpoint)."""
        status = status or {}
        headers = ["Policy", "Status", "Objective", "Iter", "Frame", "Experiment", "Checkpoint", "YAML"]
        rows = []
        for p in sorted(sumry.keys()):
            c = sumry[p]
            if c is None:
                rows.append([p, status.get(p, NOT_REPORTED), "", "", "", "", "", ""])
            else:
                rows.append([
                    p,
                    status.get(p, REPLACEABLE),
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
