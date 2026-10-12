"""Per-process reports of the PBT example (`train_member.py --report`), for tests and the GPU
qualification (`check_qualification.py` reads them).

Each rank prints one line `PBT_MEMBER {json}` at the first env step of every process (a launch
or a PBT restart): its restart info, counters, the hyperparameters the learner trains with,
hashes of its model, normalizers, optimizer and loss scaler, and what the checkpoint it started
from holds. The first env step comes after the restore and the multi-GPU broadcast and before
any update, so all ranks of a member must report the same state. One more line at the normal
end of training.
"""

import hashlib
import json
import os
import sys
from collections.abc import Mapping

import torch

PREFIX = "PBT_MEMBER "

# learner attributes holding the values the agent.params.config.<key> overrides set
EFFECTIVE = {"learning_rate": "last_lr", "entropy_coef": "entropy_coef", "e_clip": "e_clip",
             "critic_coef": "critic_coef", "grad_norm": "grad_norm", "gamma": "gamma", "tau": "tau"}


def _feed(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor{tensor.dtype}{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _feed(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _feed(digest, item)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode())


def tree_hash(value) -> str:
    """A short hash of nested dicts/lists of tensors and plain values (a state_dict)."""
    digest = hashlib.sha256()
    _feed(digest, value)
    return digest.hexdigest()[:16]


def model_hash(state_dict) -> str:
    """Hash of a model state_dict, without torch.compile's `_orig_mod.` key prefix."""
    return tree_hash({key.replace("_orig_mod.", ""): value for key, value in state_dict.items()})


def state_hashes(algo) -> dict:
    model = getattr(algo.model, "_orig_mod", algo.model)
    normalizers = {name: getattr(model, name).state_dict()
                   for name in ("running_mean_std", "value_mean_std") if hasattr(model, name)}
    return {"model": model_hash(algo.model.state_dict()), "normalizers": tree_hash(normalizers),
            "optimizer": tree_hash(algo.optimizer.state_dict()), "scaler": tree_hash(algo.scaler.state_dict())}


def checkpoint_summary(path) -> dict | None:
    """What the checkpoint a process started from holds (a PBT restart's restart.pth)."""
    if not path:
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    scaler = state.get("scaler") or {}
    return {"path": path, "epoch": state.get("epoch"), "frame": state.get("frame"), "last_lr": state.get("last_lr"),
            "entropy_coef": state.get("entropy_coef"), "config_hparams": state.get("config_hparams"),
            "scaler_scale": scaler.get("scale"), "model": model_hash(state["model"])}


def _ranks():
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def start_report(observer, *, seed, checkpoint, launch_argv) -> dict:
    """The report of a process at its first env step (see the module docstring)."""
    import rl_games
    from rl_games.common.pbt import restart_info

    algo = observer.algo
    rank, world_size = _ranks()
    effective = {key: float(getattr(algo, attr)) for key, attr in EFFECTIVE.items() if hasattr(algo, attr)}
    effective["optimizer_lr"] = [float(group["lr"]) for group in algo.optimizer.param_groups]
    effective["weight_decay"] = [float(group.get("weight_decay", 0.0)) for group in algo.optimizer.param_groups]
    return {
        "event": "start", "policy_idx": observer.cfg.policy_idx, "rank": rank, "world_size": world_size,
        "pid": os.getpid(), "rl_games": os.path.dirname(os.path.abspath(rl_games.__file__)),
        "argv": list(launch_argv), "restart": restart_info(), "restart_count": observer.restart_count,
        "seed": seed, "frame": int(algo.frame), "epoch": int(algo.epoch_num), "rnn": bool(algo.is_rnn),
        # the configured LR schedule, the same on every rank, which the restore decides on: multi-GPU
        # ranks other than 0 run with lr_schedule None, and rank 0's scheduler sets their LR
        "lr_schedule": getattr(algo, "_configured_lr_schedule", algo.config.get("lr_schedule")),
        "params": dict(observer.pbt_params), "mutation": dict(observer.cfg.mutation),
        "config": {key: algo.config.get(key) for key in EFFECTIVE},
        "effective": effective,
        "state": state_hashes(algo),
        "scaler_scale": float(algo.scaler.get_scale()) if algo.scaler.is_enabled() else None,
        "checkpoint": checkpoint_summary(checkpoint),
    }


def exit_report(observer) -> dict:
    """The report at the normal end of training."""
    rank, world_size = _ranks()
    algo = observer.algo
    return {"event": "exit", "policy_idx": observer.cfg.policy_idx, "rank": rank, "world_size": world_size,
            "pid": os.getpid(), "restart_count": observer.restart_count, "frame": int(algo.frame),
            "epoch": int(algo.epoch_num)}


def _plain(value):
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def emit(report) -> None:
    """Print `report` as a line of its own: one write, starting with a newline. The ranks of a
    member share its log file, and print() writes a line's text and its newline apart (torchrun
    runs the workers unbuffered): without that newline, a report written in between would continue
    another rank's line. Other writes to the file don't split the report's one write."""
    sys.stdout.write("\n" + PREFIX + json.dumps(report, sort_keys=True, default=_plain) + "\n")
    sys.stdout.flush()


def read_reports(lines) -> tuple[list[dict], int]:
    """(reports, malformed) in `lines`. A report is the JSON object after `PREFIX` to the end of
    its line, wherever the prefix starts (a log written without `emit`'s leading newline); a line
    holding the prefix without one counts as malformed."""
    reports, bad = [], 0
    for line in lines:
        start = line.find(PREFIX)
        if start < 0:
            continue
        try:
            report = json.loads(line[start + len(PREFIX):])
        except json.JSONDecodeError:
            report = None
        if isinstance(report, dict):
            reports.append(report)
        else:
            bad += 1
    return reports, bad
