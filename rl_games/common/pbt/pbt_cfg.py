# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import difflib
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import shutil
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields

from rl_games.common.pbt.mutation import MUTATION_FUNCS
from rl_games.common.pbt.pbt_utils import flatten_dict, launch_dirs

REPLACE_RULES = ("dexpbt", "threshold")

# the margins each replacement rule reads; the other rule ignores them
RULE_KEYS = {
    "dexpbt": ("replace_fraction_worst", "replace_fraction_best", "replace_threshold_frac_std",
               "replace_threshold_frac_absolute", "replace_threshold_absolute", "leader_params_prob"),
    "threshold": ("threshold_std", "threshold_abs"),
}

# the settings every member of a population shares, with its rule's (RULE_KEYS): see settings_hash. A key
# added here later needs a default that keeps the earlier behaviour: configs at it keep their hash
POPULATION_KEYS = ("num_policies", "interval_steps", "objective", "workspace", "replace_rule", "objective_fn",
                   "interval_unit")

# IsaacGymEnvs pbt keys without a counterpart of the same name
_IGE_KEYS = {
    "change_min": "use change_range: [min, max]",
    "change_max": "use change_range: [min, max]",
    "dbg_mode": "remove it; force_outcome: self, best or donor:<idx> is the counterpart of dbg_mode: True, "
                "for qualifying an integration",
}


@dataclass
class PbtCfg:
    """
    Population-Based Training (PBT) configuration; docs/PBT.md has the full reference.

    Two replacement rules are available (`replace_rule`):
      - "dexpbt" (default): rank the population; a member in the worst `replace_fraction_worst`
        of the members that can be replaced now restarts from a random member of the best
        `replace_fraction_best` when the gap is large enough, otherwise it restarts from itself
        with mutated parameters (DexPBT), unless the mutation changed no value.
      - "threshold": underperformers (score < min(mean - threshold_std*std, mean - threshold_abs))
        restart from a random leader (score > max(mean + threshold_std*std, mean + threshold_abs)).
    On a replacement each key of `mutation` is mutated with probability `mutation_rate` by its
    mutation function (docs/PBT.md lists them; not all use `change_range`).

    `interval_steps`, `start_after` and `initial_delay` count frames, the learner's `frame`: agent
    steps summed over the member's ranks; or learner epochs (`interval_unit`).

    Numeric fields are converted by their declared type and range-checked at construction, so
    every rank fails the same way at init: YAML loaders deliver `2e7` and `1e-3` as strings and
    Hydra delivers `2e7` as a float. Invalid values raise ValueError naming the key, and
    `from_dict` rejects unknown keys. `validate()` checks a whole PBT setup.
    """

    enabled: bool = False
    """Read by train scripts to decide whether to attach the observer. The observer runs PBT
    whenever it is attached (it warns when this is False)."""

    policy_idx: int = 0
    """Index of this learner in the population (unique in [0, num_policies-1])."""

    num_policies: int = 8
    """Total number of learners participating in PBT (>= 1)."""

    directory: str = ""
    """Not read by the observer, which keeps the workspace in `<train_dir>/<workspace>`. Train
    scripts may use it to place the run (Isaac Lab-style scripts put their log root under it)."""

    workspace: str = "pbt_workspace"
    """The population's workspace directory, `<train_dir>/<workspace>`: every member of a
    population runs with the same train_dir and workspace."""

    objective: str = ""
    """Dotted address of the objective inside env infos — required unless `objective_fn` computes
    it (`validate()` and the observer raise without either); there is no portable default (info
    layouts differ per backend). Examples:
    'episode.Episode_Reward/success' (Isaac Lab-style nested infos), 'scores' (flat dicts).
    A per-env tensor is read at the finished envs; a scalar is taken as the mean over the
    envs that finished on that step. Episodes with a NaN or infinite value are left out, and
    an iteration waits until every rank has a finite value since the previous one (see
    `max_consecutive_errors`). If reward is stationary, a term that corresponds to
    task success is usually enough; with non-stationary rewards or curricula, prefer a true
    task objective that ranks curriculum progress first."""

    objective_window: int = 0
    """Number of most recent finished episodes (per rank) averaged into the objective.
    0 = the rank's number of environments: the size of DexPBT's estimator, which averages each
    env's latest episode instead (envs weigh equally there, episodes here)."""

    interval_steps: int = 100_000
    """Frames (or epochs, `interval_unit`) between PBT iterations (save, compare, replace/mutate); > 0."""

    start_after: int = 0
    """Frames (or epochs) a member trains after each process start (a fresh launch, a resume or a
    restart), or a self-mutation applied in process, before it can be replaced. Scores right after
    a restart still mostly measure the source policy, not the mutation. No replacement is decided
    either with fewer than max(start_after, interval_steps) frames (or epochs) left before
    max_epochs or max_frames end the run."""

    initial_delay: int = 0
    """Frames (or epochs) from the start of training (the member's frame and epoch, which a restart
    keeps) before the member can be replaced."""

    replace_rule: str = ""
    """'dexpbt' (rank-based) or 'threshold' (mean ± band), see the class docstring. Empty:
    'threshold' when threshold_std/threshold_abs are set (configs predating the DexPBT rule),
    else 'dexpbt'."""

    replace_fraction_worst: float = 0.125
    """dexpbt: fraction of the population (rounded up) eligible for replacement, in (0, 1]: the
    members ranked last among those that reported and can be replaced now (not restarting, in a
    grace window or near the end of their run), leaving out a member 2+ iterations behind that
    was not below the deciding member at its iteration (see `pbt_utils.member_status`)."""

    replace_fraction_best: float = 0.3
    """dexpbt: fraction of the population (rounded up) that replacements are drawn from, in (0, 1]."""

    replace_threshold_frac_std: float = 0.5
    """dexpbt: a transfer from the drawn donor needs a lead of more than this many population stds
    (std over the objectives with the worst 20% trimmed); otherwise the member self-mutates; >= 0."""

    replace_threshold_frac_absolute: float = 0.05
    """dexpbt: a transfer from the drawn donor needs a lead of more than this fraction of |donor
    objective| (unless `replace_threshold_absolute` is set); otherwise the member self-mutates; >= 0."""

    leader_params_prob: float = 0.5
    """dexpbt: probability that a replaced member mutates the source's hyperparameters rather
    than its own, in [0, 1]."""

    threshold_std: float | None = None
    """threshold: std-based margin k: leaders are above max(mean + k·std, mean + threshold_abs),
    underperformers below min(mean - k·std, mean - threshold_abs). Default 0.10."""

    threshold_abs: float | None = None
    """threshold: absolute margin A: leaders are above max(mean + threshold_std·std, mean + A),
    underperformers below min(mean - threshold_std·std, mean - A). Default 0.05."""

    mutation_rate: float = 0.25
    """Per-parameter probability of mutation when a policy is replaced, in [0, 1]."""

    change_range: tuple[float, float] = (1.1, 2.0)
    """Lower and upper bound of the multiplicative change factor: two finite numbers > 0 (sorted).
    mutate_discount ignores it (it steps 1 - x by a factor in [1.1, 1.2])."""

    mutation: dict[str, str] = field(default_factory=dict)
    """Which parameters to mutate on restart, mapping flattened param address to mutation function:
        {
            "agent.params.config.learning_rate": "mutate_float",
            "agent.params.config.grad_norm": "mutate_float",
            "agent.params.config.gamma": "mutate_discount",
        }
    Must not be empty. Every key must resolve to a config value (or an `extra_params` entry of
    the observer).
    """

    launcher: str = ""
    """Executable used to re-exec the training process on restart. Empty = sys.executable.
    An existing file (relative to the launch directory, as os.execv sees it) or a command on
    PATH; `validate()` replaces it with its absolute path. Isaac Lab / Isaac Sim workflows
    should point this at their python.sh wrapper."""

    checkpoint_arg: str = "--checkpoint"
    """CLI name of the train script's checkpoint argument; restarts pass `<name>=<path>`."""

    reseed_on_restart: bool = True
    """Replace the value of `seed_arg` on each restart's command line with a seed derived from
    (that seed, policy_idx, restart count), so a member does not replay its initial random
    streams after each replacement. Only when the command has the argument: without it every
    restart seeds as the launch did, except that a restart without launch_argv passes a seed
    derived from the config seed (none for a config seed of None or -1). A seed of -1 (random per
    launch) is kept as is. PBT's own draws (donor, mutations) come from a private generator
    either way."""

    seed_arg: str = "--seed"
    """CLI name of the train script's seed argument (used by `reseed_on_restart`)."""

    max_consecutive_errors: int = 5
    """Raise after this many consecutive PBT iterations fail to save/load checkpoints (a state
    whose weights hold NaN or inf is not saved), or after this many iteration boundaries in a
    row at which a rank had no finite objective since its last iteration (or the process
    start) although every env of the rank had ended an episode, and at least a window of
    episodes ended (a diverged member), instead of silently training without PBT (>= 1).
    Waiting for episodes to end never counts."""

    wandb_finish_timeout: float = 120.0
    """Seconds a restart waits for the open wandb run to finish before re-executing (a finite
    number >= 0). After a clean finish the restarted process resumes the same run; after a
    timeout or an error it logs to a new run `<run id>-r<restart count>` (the old run may
    still be uploading)."""

    reuse_workspace: bool = False
    """Let a fresh launch (no checkpoint, not a PBT restart) start in a member directory that
    holds numbered checkpoints of an earlier run. Workspace files are named by iteration only,
    so peers would read those as the member's current state: by default the member raises at
    its first step. With true it deletes its own numbered files and starts over; `best/` keeps
    the earlier run's records, so a new best is recorded once it beats them."""

    replace_threshold_absolute: float = 0.0
    """dexpbt: when > 0, a transfer needs a lead of more than this many objective units in place
    of `replace_threshold_frac_absolute` × |donor objective|, which changes with a constant added
    to the objective; >= 0. 0 keeps DexPBT's fractional margin."""

    mutation_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    """[lo, hi] per mutation key (two finite numbers, lo < hi): every replacement clamps the new
    values into them, also the values it doesn't mutate (a donor's may predate the bounds). An
    integer param (an int that mutate_mini_epochs or a custom function mutates) is clamped to the
    integers inside: `validate()` raises when there is none, and warns about a configured value
    outside. The base `_pbt_iteration` and `initial_mutation` apply them; a subclass that
    replaces `_pbt_iteration` calls `mutation.clamp_to_bounds` itself."""

    force_outcome: str = ""
    """For qualifying an integration, not for training: 'self', 'best' or 'donor:<idx>' makes
    every iteration outside the grace windows and the end-of-run margin restart the member from
    itself (a self-mutation), the best member or member idx, whatever the rule says; a
    self-mutation that changes no value still keeps training. It takes the rule's place in
    `pbt_utils.select_dexpbt` and `select_threshold`, so a subclass whose `_pbt_iteration` calls
    them gets it too. Every rank says so at init, and the member's workspace yamls, best records,
    restart.yaml and decisions.jsonl lines carry `forced: <value>`. Empty: off."""

    initial_mutation: bool = False
    """Mutate a fresh member's `agent.params.config` values once at init, in the params the
    observer is built with (runner.load reads them afterwards), as DexPBT does at launch: a
    population launched from one config starts diverse. With `mutation_rate`, drawn again until
    a value changes (at most 20 draws), from (the launch command's `seed_arg` value, policy_idx):
    the same on every rank. Fresh: no restart payload and no numbered checkpoint in the member's
    workspace directory, not even an earlier run's that `reuse_workspace` lets it delete (rank 0
    says so)."""

    restart_drop_args: dict[str, int | str] = field(default_factory=dict)
    """Launch-only arguments every restart removes from the command, as {name: nargs}: 0 for a
    switch, N for N values, '?' for an optional one (the next argument when it starts with no '-'
    and holds no '='); `name=value` is one argument. E.g. {'--reset-env-state': 0} for a
    warm-start switch that must not act on restart.pth."""

    self_mutation: str = "restart"
    """How a self-mutation (the member keeps its weights) takes effect: 'restart' re-execs the
    member from restart.pth, as a transfer does; 'in_process' sets the new values on the running
    learner, on every rank before the decision epoch's update (`algo.set_param`), when every
    changed key is an `agent.params.config` value set_param takes, and restarts otherwise (env
    and extra_params values, central_value_config values). Either way the member's grace window
    and objective window start over."""

    transfer: str = "restart"
    """How a transfer (the member takes another member's weights) takes effect: 'restart' re-execs
    the member from restart.pth; 'in_process' loads restart.pth into the running learner on every
    rank at the start of the next epoch, after the decision epoch's update, as the restarted
    process would restore it, keeping the member's epoch, frame and best reward, when the running
    learner can take every changed value (as for self_mutation), and restarts otherwise. The
    learner's observer chain must forward before_steps (rl_games' MultiObserver does)."""

    max_member_lag: int = 10
    """Liveness, in the deciding member's median iteration wall times (>= 0). A member without a
    sign of life for longer, neither a heartbeat in its owner record (owner.yaml, renewed at every
    iteration boundary) nor a new workspace yaml, is stale: left out of the ranking, the donors,
    the spread and the reported count. One with a recent heartbeat but no new yaml for longer,
    whose yaml at the deciding member's iteration is 2+ iterations behind, waits (for its
    objective, or at a start-up): ranked and a donor, but not replaceable. A member whose exit
    record (exit.yaml) is newer than its heartbeat has finished: ranked and a donor, never
    replaceable. 0 judges no member by time. Members of an older rl_games (no owner record) are
    never stale. Clocks are assumed synchronized."""

    force_owner: bool = False
    """Start even when the member's owner record names a process that may still run it: on this
    host its pid with the recorded start time, on another host a heartbeat within max_member_lag
    of its iteration times (10 minutes when unknown), unless the member's exit record says that
    launch ended. A plain launch refuses then, as two processes of one member would write the
    same files; a PBT restart or a relaunch with the member's restart payload (member_resume)
    takes over without the check. The other process, if it still runs, stops at its next
    iteration boundary."""

    stop_with_first: bool = False
    """End the population with its first member to finish: in its last epoch (by max_epochs or
    max_frames) that member writes <workspace>/finished.yaml, and every other member saves the
    custom-stop checkpoint and stops at the end of its next epoch, through the learner's stop_fn
    (a stop_fn set before train() is called first). No replacement is decided once the marker
    exists. A marker with other max_epochs, max_frames or population settings (another run) is
    ignored with a warning; a fresh launch under reuse_workspace removes one written before it."""

    objective_fn: str = ""
    """'pkg.module:function' (dotted attributes after ':', as in entry points), imported at init: a
    function that computes the objective in place of the lookup of `objective`, which may then be
    empty (and is not read). It is called at every env step on every rank, as process_infos is,
    as fn(infos, done_indices, num_agents, algo) with the learner as `algo`, and returns the envs'
    objective as a looked-up value would be (a per-env tensor or array, or a scalar), or None for
    no objective on this step (the episodes that ended count as unreadable). Its exceptions
    propagate. The observer's `objective_fn` argument takes a function object instead."""

    interval_unit: str = "frames"
    """What `interval_steps`, `start_after` and `initial_delay` count: 'frames' (the learner's
    `frame`) or 'epochs' (the learner's completed epochs, the same on every rank; a restart keeps
    them). Workspace iterations count it, so a population shares it: it is one of the settings in
    `settings_hash`, and workspace yamls record it. A subclass that replaces `_pbt_iteration`
    compares its grace windows in it: `_progress(frame)`."""

    def __post_init__(self):
        hints = typing.get_type_hints(type(self))
        for f in fields(self):
            kind, optional = _numeric_kind(hints[f.name])
            value = getattr(self, f.name)
            if kind is not None and not (optional and value is None):
                setattr(self, f.name, _to_int(f.name, value) if kind is int else _to_float(f.name, value))
        self.change_range = _to_change_range(self.change_range)
        if not isinstance(self.mutation, Mapping):
            raise ValueError(f"pbt.mutation must map flattened param names to mutation functions, "
                             f"got {self.mutation!r}")
        self.mutation = dict(self.mutation)
        self.mutation_bounds = _to_bounds(self.mutation_bounds, self.mutation)
        self.restart_drop_args = _to_drop_args(self.restart_drop_args)
        legacy_thresholds = self.threshold_std is not None or self.threshold_abs is not None
        if not self.replace_rule:
            self.replace_rule = "threshold" if legacy_thresholds else "dexpbt"
        self.threshold_std = 0.10 if self.threshold_std is None else self.threshold_std
        self.threshold_abs = 0.05 if self.threshold_abs is None else self.threshold_abs
        if self.replace_rule not in REPLACE_RULES:
            raise ValueError(f"pbt.replace_rule must be one of {REPLACE_RULES}, got {self.replace_rule!r}")
        self._check_ranges()

    def _check_ranges(self):
        def require(ok, name, rule):
            if not ok:
                raise ValueError(f"pbt.{name} must be {rule}, got {getattr(self, name)!r}")

        require(self.num_policies >= 1, "num_policies", ">= 1")
        require(0 <= self.policy_idx < self.num_policies, "policy_idx", f"in [0, num_policies={self.num_policies})")
        require(self.interval_steps > 0, "interval_steps", "> 0")
        for name in ("start_after", "initial_delay", "objective_window", "max_member_lag"):
            require(getattr(self, name) >= 0, name, ">= 0")
        require(self.max_consecutive_errors >= 1, "max_consecutive_errors", ">= 1")
        for name in ("replace_fraction_worst", "replace_fraction_best"):
            require(0.0 < getattr(self, name) <= 1.0, name, "in (0, 1]")
        for name in ("mutation_rate", "leader_params_prob"):
            require(0.0 <= getattr(self, name) <= 1.0, name, "in [0, 1]")
        for name in ("replace_threshold_frac_std", "replace_threshold_frac_absolute", "replace_threshold_absolute"):
            require(getattr(self, name) >= 0.0, name, ">= 0")
        require(math.isfinite(self.wandb_finish_timeout) and self.wandb_finish_timeout >= 0,
                "wandb_finish_timeout", "a finite number >= 0")
        # a string such as 'false' would be truthy and switch the option on or off
        for name in ("reuse_workspace", "initial_mutation", "force_owner", "stop_with_first"):
            require(isinstance(getattr(self, name), bool), name, "true or false")
        donor = re.fullmatch(r"donor:([0-9]+)", self.force_outcome) if isinstance(self.force_outcome, str) else None
        require(self.force_outcome in ("", "self", "best") or (donor and int(donor.group(1)) < self.num_policies),
                "force_outcome", "'', 'self', 'best' or 'donor:<idx>' with idx in [0, num_policies)")
        for name in ("self_mutation", "transfer"):
            require(getattr(self, name) in ("restart", "in_process"), name, "'restart' or 'in_process'")
        require(self.interval_unit in ("frames", "epochs"), "interval_unit", "'frames' or 'epochs'")

    @classmethod
    def from_dict(cls, d):
        """Build from a config dict. Unknown keys raise, each listed with the key likely meant
        when one is close."""
        known = [f.name for f in fields(cls)]
        unknown = sorted((key for key in d if key not in known), key=str)
        if unknown:
            raise ValueError("unknown pbt config keys: " + ", ".join(_unknown_key_hint(key, known) for key in unknown))
        return cls(**dict(d))


def _unknown_key_hint(key, known):
    if key in _IGE_KEYS:
        return f"{key} (IsaacGymEnvs key: {_IGE_KEYS[key]})"
    # 0.75: typos match at 0.8+; the default 0.6 also suggests 'launcher' for 'another'
    close = difflib.get_close_matches(str(key), known, n=1, cutoff=0.75)
    return f"{key} (did you mean {close[0]!r}?)" if close else str(key)


def _numeric_kind(hint):
    """(int, float or None, whether None is allowed) for a field annotation."""
    args = typing.get_args(hint)
    optional = type(None) in args
    if optional:
        rest = [arg for arg in args if arg is not type(None)]
        hint = rest[0] if len(rest) == 1 else None
    return (hint if hint is int or hint is float else None), optional


def _number(name, value, what):
    """int or float from a number or a numeric string ('2e7', '20_000_000'); bool and None raise."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(value, str):
        for parse in (int, float):
            try:
                return parse(value.strip())
            except ValueError:
                pass
    raise ValueError(f"pbt.{name} must be {what}, got {value!r}")


def _to_int(name, value):
    number = _number(name, value, "an integer")
    if isinstance(number, float):
        if not number.is_integer():  # also inf and nan
            raise ValueError(f"pbt.{name} must be an integer, got {value!r}")
        number = int(number)
    return number


def _to_float(name, value):
    number = float(_number(name, value, "a number"))
    if math.isnan(number):
        raise ValueError(f"pbt.{name} must be a number, got {value!r}")
    return number


def _to_change_range(value):
    """Two finite numbers > 0 as a sorted tuple (mutate_float takes the bounds in either order).

    A lower bound of 0 gives unbounded factors and negative bounds flip signs; a string or a
    third element would only fail at the first mutation, on rank 0.
    """
    message = f"pbt.change_range must be two finite numbers > 0, e.g. [1.1, 2.0], got {value!r}"
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(message)
    bounds = list(value)
    if len(bounds) != 2:
        raise ValueError(message)
    try:
        bounds = [float(_number("change_range", bound, "a number")) for bound in bounds]
    except ValueError:
        raise ValueError(message) from None
    if not all(math.isfinite(bound) and bound > 0 for bound in bounds):
        raise ValueError(message)
    return tuple(sorted(bounds))


def _to_bounds(value, mutation):
    """`mutation_bounds` as {mutation key: (lo, hi)}: two finite numbers with lo < hi per key."""
    if not isinstance(value, Mapping):
        raise ValueError(f"pbt.mutation_bounds must map mutation keys to [lo, hi], got {value!r}")
    bounds = {}
    for key, pair in value.items():
        if key not in mutation:
            raise ValueError(f"pbt.mutation_bounds key {key!r} is not a key of pbt.mutation")
        message = f"pbt.mutation_bounds of {key} must be two finite numbers lo < hi, got {pair!r}"
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Iterable):
            raise ValueError(message)
        try:
            lo, hi = (_to_float("mutation_bounds", bound) for bound in pair)
        except ValueError:
            raise ValueError(message) from None
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(message)
        bounds[key] = (lo, hi)
    return bounds


def _to_drop_args(value):
    """`restart_drop_args` as {argument name: nargs}, nargs an int >= 0 or '?'."""
    message = f"pbt.restart_drop_args must map argument names to an int >= 0 or '?', got {value!r}"
    if not isinstance(value, Mapping):
        raise ValueError(message)
    drop = {}
    for name, nargs in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(message)
        if nargs != "?":
            try:
                nargs = _to_int("restart_drop_args", nargs)
            except ValueError:
                raise ValueError(message) from None
            if nargs < 0:
                raise ValueError(message)
        drop[name] = nargs
    return drop


def settings_hash(cfg) -> str:
    """A short hash of `cfg`'s population settings (POPULATION_KEYS and its rule's RULE_KEYS).

    Members write it into their workspace yamls; one that reads another hash from a peer leaves
    that peer out of its ranking: they run different populations (e.g. `interval_steps` edited
    for some members of a running population). Per-member settings (policy_idx, mutation, grace
    windows) don't enter it, nor do settings at their PbtCfg default: restarts exec the current
    checkout, and a setting a later version adds must not split a population running across the
    upgrade.
    """
    defaults = {f.name: f.default for f in fields(cfg)}
    settings = {key: getattr(cfg, key) for key in POPULATION_KEYS + RULE_KEYS[cfg.replace_rule]}
    text = json.dumps({key: value for key, value in settings.items() if value != defaults[key]}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def rule_notices(cfg, section) -> list[str]:
    """Init-output lines about the replacement rule of `cfg`, built from the pbt `section` as given.

    The resolved rule and the margins in use when `replace_rule` was not set (master's #359
    configs without thresholds silently moved to dexpbt), a warning for keys of the other rule
    (they have no effect), and dexpbt's minimum population.
    """
    lines = []
    # replace_threshold_absolute > 0 replaces the fractional margin
    unused = "replace_threshold_frac_absolute" if cfg.replace_threshold_absolute > 0 else "replace_threshold_absolute"
    margins = ", ".join(f"{key} {getattr(cfg, key):g}" for key in RULE_KEYS[cfg.replace_rule] if key != unused)
    if not section.get("replace_rule"):
        if cfg.replace_rule == "threshold":
            lines.append(f"replace_rule not set: using 'threshold' because threshold_std/threshold_abs are set "
                         f"({margins}); set replace_rule explicitly ('dexpbt' is the DexPBT rank-based rule)")
        else:
            lines.append(f"replace_rule not set: using 'dexpbt' ({margins})")
    inactive = [key for rule, keys in RULE_KEYS.items() if rule != cfg.replace_rule
                for key in keys if section.get(key) is not None]
    if inactive:
        lines.append(f"WARNING {', '.join(inactive)} {'has' if len(inactive) == 1 else 'have'} no effect under "
                     f"replace_rule '{cfg.replace_rule}'")
    if cfg.replace_rule == "dexpbt" and cfg.num_policies <= 2 and not cfg.force_outcome:
        lines.append("WARNING the dexpbt rule needs more than max(2, num_policies // 2) members to report, so it "
                     "never replaces with num_policies <= 2; use 3+ members or replace_rule: threshold")
    return lines


# --- validate(): the whole PBT setup, before anything is built ----------------------------------

_CONFIG = "agent.params.config."
# integer settings, by the last component of their key. Counts are stepped by mutate_mini_epochs;
# no built-in function keeps a buffer or batch size valid (mutate_mini_epochs clamps to [1, 8],
# the others return floats). Both accept a custom function registered in MUTATION_FUNCS.
_COUNT_KEYS = frozenset({"mini_epochs", "mini_epochs_num"})
_SIZE_KEYS = frozenset({"horizon_length", "minibatch_size", "minibatch_size_per_env", "seq_length", "num_actors"})
_BUILTIN_FUNCS = dict(MUTATION_FUNCS)  # what mutation.py ships, before any registration


def _is_builtin(fn_name):
    return fn_name in _BUILTIN_FUNCS and MUTATION_FUNCS.get(fn_name) is _BUILTIN_FUNCS[fn_name]


def _mutable_value(key, value, fn_name):
    """A plain Python int or float for a mutable param, checked against its mutation function."""
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        value = value.item()  # numpy scalars, 0-d arrays and tensors
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"pbt.mutation key {key}: the value {value!r} is not a finite number")
    value = int(value) if isinstance(value, int) else float(value)

    leaf = key.rsplit(".", 1)[-1]
    builtin = _is_builtin(fn_name)
    integer_step = builtin and fn_name == "mutate_mini_epochs"
    if integer_step or leaf in _COUNT_KEYS or leaf in _SIZE_KEYS:
        if not float(value).is_integer():
            how = " for mutate_mini_epochs, which steps it by ±1 within [1, 8]" if integer_step else ""
            raise ValueError(f"pbt.mutation key {key} must be an integer{how}, got {value!r}")
        value = int(value)
    if leaf in _COUNT_KEYS and builtin and not integer_step:
        raise ValueError(f"pbt.mutation key {key} is an integer setting and {fn_name} would make it a float: "
                         f"use mutate_mini_epochs (or a custom function registered in MUTATION_FUNCS)")
    if leaf in _SIZE_KEYS and builtin:
        raise ValueError(f"pbt.mutation key {key} sizes buffers and batches, and no built-in function keeps it "
                         f"valid (mutate_mini_epochs clamps it to [1, 8], the others make it a float): register "
                         f"a custom function in MUTATION_FUNCS")
    return value


def _no_effect_notes(mutation, flat):
    """Mutations that don't do what their name suggests under the configured LR schedule."""
    schedule = flat.get(_CONFIG + "lr_schedule")
    lines = []
    if _CONFIG + "kl_threshold" in mutation and schedule != "adaptive":
        lines.append(f"WARNING {_CONFIG}kl_threshold is mutated but lr_schedule is {schedule!r}: only the "
                     f"adaptive schedule reads kl_threshold, so the mutation has no effect")
    if _CONFIG + "learning_rate" in mutation and schedule == "adaptive":
        lines.append(f"note: {_CONFIG}learning_rate is mutated under lr_schedule 'adaptive': a mutation only "
                     f"moves the learning rate the KL-driven schedule restarts from; mutate kl_threshold to "
                     f"change its target")
    return lines


def _resolve_launcher(launcher, cwd=None):
    """Absolute path of what os.execv will run for `launcher`: an existing file relative to `cwd`
    (default: the launch directory restarts exec from, see `pbt_utils.launch_dirs`: Hydra's
    original cwd inside a Hydra job, else the current directory), else a command on PATH.
    Empty stays empty (sys.executable)."""
    if not launcher:
        return launcher
    if not isinstance(launcher, str):
        raise ValueError(f"pbt.launcher must be a path or a command name, got {launcher!r}")
    base = os.fspath(cwd) if cwd else launch_dirs()[0]
    path = os.path.join(base, launcher)
    if os.path.isfile(path):
        if not os.access(path, os.X_OK):
            raise ValueError(f"pbt.launcher {launcher!r} is not executable: {os.path.abspath(path)}")
        return os.path.abspath(path)
    found = shutil.which(launcher) if os.sep not in launcher else None
    if found is None:
        raise ValueError(f"pbt.launcher {launcher!r} is neither a file relative to {base} nor a command on PATH")
    return os.path.abspath(found)


def load_objective_fn(path):
    """The function `pbt.objective_fn` names ('pkg.module:function'), imported; None for ''.
    ValueError when the module or the attribute can't be imported, or it is not callable."""
    if path == "":
        return None
    if not isinstance(path, str) or not re.fullmatch(r"\w+(\.\w+)*:\w+(\.\w+)*", path):
        raise ValueError(f"pbt.objective_fn must be 'pkg.module:function', got {path!r}")
    module, attributes = path.split(":")
    try:
        function = importlib.import_module(module)
        for name in attributes.split("."):
            function = getattr(function, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"pbt.objective_fn {path!r} does not import: {exc!r}") from exc
    if not callable(function):
        raise ValueError(f"pbt.objective_fn {path!r} is not callable")
    return function


def validate(params, extra_params=None, launcher_cwd=None, *, objective_fn=None):
    """Check a PBT setup without building anything; returns (cfg, mutable params).

    Builds the PbtCfg from `params["pbt"]` and flattens `{"agent": params}` plus `extra_params`
    as PbtAlgoObserver does. `objective` is required unless an objective function is set:
    `pbt.objective_fn`, which must import (`load_objective_fn`), or the observer's
    `objective_fn` argument, passed here too (not both). Every mutation key must resolve to a
    finite number, every mutation function must exist and suit its value (integer settings need
    an integer function), and `launcher` must resolve the way os.execv will see it: an existing
    file relative to `launcher_cwd` (default: the directory restarts exec from: Hydra's original
    cwd inside a Hydra job, else the current directory), else a command on PATH; the cfg keeps
    the absolute path. Raises ValueError naming the key, prints a note for mutations that don't
    do what their name suggests, and warns about a value outside its `mutation_bounds`, which
    become the integers inside them for an integer param (an int that mutate_mini_epochs or a
    custom function mutates). The params come back as plain Python numbers (numpy scalars, 0-d
    tensors and numeric strings converted), in config order.

    It touches no tensors and no environment (it imports `pbt.objective_fn`'s module), so a
    train script can call it before building the env; PbtAlgoObserver calls it on every rank.
    """
    section = params.get("pbt") if isinstance(params, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError("params need a 'pbt' section with the PBT config")
    cfg = PbtCfg.from_dict(section)
    if objective_fn is not None and cfg.objective_fn:
        raise ValueError(f"pbt.objective_fn is {cfg.objective_fn!r} and the observer got an objective_fn as well: "
                         f"set one of them")
    function = objective_fn if objective_fn is not None else load_objective_fn(cfg.objective_fn)
    if not isinstance(cfg.objective, str) or not (cfg.objective or function):
        raise ValueError("pbt.objective is required unless an objective function is set (pbt.objective_fn or the "
                         "observer's objective_fn): a dotted address into env infos, e.g. "
                         "'episode.Episode_Reward/success' (Isaac Lab) or 'scores' (flat infos)")
    if function is not None and cfg.objective:
        print(f"PbtCfg: note: pbt.objective {cfg.objective!r} is not read: the objective function computes the "
              f"objective")
    if not cfg.mutation:
        raise ValueError("pbt.mutation is empty: map the params to mutate to mutation functions, "
                         "e.g. {'agent.params.config.learning_rate': 'mutate_float'}")
    flat = flatten_dict({"agent": params})
    flat.update(extra_params or {})
    missing = sorted(set(cfg.mutation) - set(flat), key=str)
    if missing:
        raise ValueError(f"pbt.mutation keys not found in the params or extra_params: {missing}")
    unknown = sorted({str(fn) for fn in cfg.mutation.values() if not isinstance(fn, str) or fn not in MUTATION_FUNCS})
    if unknown:
        raise ValueError(f"unknown pbt.mutation functions {unknown}; available: {sorted(MUTATION_FUNCS)}")
    pbt_params = {key: _mutable_value(key, value, cfg.mutation[key])
                  for key, value in flat.items() if key in cfg.mutation}
    cfg.launcher = _resolve_launcher(cfg.launcher, launcher_cwd)
    for line in _no_effect_notes(cfg.mutation, flat):
        print(f"PbtCfg: {line}")
    for key, (lo, hi) in cfg.mutation_bounds.items():
        value, fn_name = pbt_params[key], cfg.mutation[key]
        if isinstance(value, int) and (fn_name == "mutate_mini_epochs" or not _is_builtin(fn_name)):
            # an integer param (the other built-in functions return floats): the integers inside
            if math.ceil(lo) > math.floor(hi):
                raise ValueError(f"pbt.mutation_bounds of {key} must hold an integer ({key} is an integer param), "
                                 f"got [{lo:g}, {hi:g}]")
            lo, hi = cfg.mutation_bounds[key] = math.ceil(lo), math.floor(hi)
        if not lo <= value <= hi:
            print(f"PbtCfg: WARNING {key} is {value!r}, outside its mutation_bounds [{lo:g}, {hi:g}]: "
                  f"a replacement clamps it into them")
    return cfg, pbt_params
