# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field, fields

REPLACE_RULES = ("dexpbt", "threshold")


@dataclass
class PbtCfg:
    """
    Population-Based Training (PBT) configuration.

    Two replacement rules are available (`replace_rule`):
      - "dexpbt" (default): rank the population; a member in the worst `replace_fraction_worst`
        restarts from a random member of the best `replace_fraction_best` when the gap is large
        enough, otherwise it restarts from itself with mutated parameters (DexPBT).
      - "threshold": underperformers (score < min(mean - threshold_std*std, mean - threshold_abs))
        restart from a random leader (score > max(mean + threshold_std*std, mean + threshold_abs)).
    On replacement, selected hyperparameters are mutated multiplicatively in [change_min, change_max].
    """

    enabled: bool = False
    """Enable/disable PBT logic."""

    policy_idx: int = 0
    """Index of this learner in the population (unique in [0, num_policies-1])."""

    num_policies: int = 8
    """Total number of learners participating in PBT."""

    directory: str = ""
    """Root directory for PBT artifacts (checkpoints, metadata)."""

    workspace: str = "pbt_workspace"
    """Subfolder under the training dir to isolate this PBT run."""

    objective: str = ""
    """Dotted address of the scalar objective inside env infos — required when PBT is
    enabled; there is no portable default (info layouts differ per backend). Examples:
    'episode.Episode_Reward/success' (Isaac Lab-style nested infos), 'scores' (flat dicts).
    A per-env tensor is read at the finished envs; a scalar is taken as the mean over the
    envs that finished on that step. If reward is stationary, a term that corresponds to
    task success is usually enough; with non-stationary rewards or curricula, prefer a true
    task objective that ranks curriculum progress first."""

    objective_window: int = 0
    """Number of most recent finished episodes (per rank) averaged into the objective.
    0 = the rank's number of environments (DexPBT)."""

    interval_steps: int = 100_000
    """Environment steps between PBT iterations (save, compare, replace/mutate)."""

    start_after: int = 0
    """Environment steps a member trains after each (re)start before it can be replaced.
    Scores right after a restart still mostly measure the source policy, not the mutation."""

    initial_delay: int = 0
    """Environment steps from the start of training before any member can be replaced."""

    replace_rule: str = ""
    """'dexpbt' (rank-based) or 'threshold' (mean ± band), see the class docstring. Empty:
    'threshold' when threshold_std/threshold_abs are set (configs predating the DexPBT rule),
    else 'dexpbt'."""

    replace_fraction_worst: float = 0.125
    """dexpbt: fraction of the population (rounded up) eligible for replacement."""

    replace_fraction_best: float = 0.3
    """dexpbt: fraction of the population (rounded up) that replacements are drawn from."""

    replace_threshold_frac_std: float = 0.5
    """dexpbt: replace only if the candidate leads by more than this many population stds
    (std over the objectives with the worst 20% trimmed)."""

    replace_threshold_frac_absolute: float = 0.05
    """dexpbt: replace only if the candidate leads by more than this fraction of |candidate objective|."""

    leader_params_prob: float = 0.5
    """dexpbt: probability that a replaced member mutates the source's hyperparameters rather
    than its own."""

    threshold_std: float | None = None
    """threshold: std-based margin k in max(mean ± k·std, mean ± threshold_abs). Default 0.10."""

    threshold_abs: float | None = None
    """threshold: absolute margin A in max(mean ± threshold_std·std, mean ± A). Default 0.05."""

    mutation_rate: float = 0.25
    """Per-parameter probability of mutation when a policy is replaced."""

    change_range: tuple[float, float] = (1.1, 2.0)
    """Lower and upper bound of the multiplicative change factor."""

    mutation: dict[str, str] = field(default_factory=dict)
    """Which parameters to mutate on restart, mapping flattened param address to mutation function:
        {
            "agent.params.config.learning_rate": "mutate_float",
            "agent.params.config.grad_norm": "mutate_float",
            "agent.params.config.gamma": "mutate_discount",
        }
    Every key must resolve to a config value (or an `extra_params` entry of the observer).
    """

    launcher: str = ""
    """Executable used to re-exec the training process on restart. Empty = sys.executable.
    Isaac Lab / Isaac Sim workflows should point this at their python.sh wrapper."""

    checkpoint_arg: str = "--checkpoint"
    """CLI name of the train script's checkpoint argument; restarts pass `<name>=<path>`."""

    reseed_on_restart: bool = True
    """Give every restart a new seed derived from (seed, policy_idx, restart count), so a
    member does not replay its initial random streams after each replacement. A seed of -1
    (random per launch) is kept as is."""

    seed_arg: str = "--seed"
    """CLI name of the train script's seed argument (used by `reseed_on_restart`)."""

    max_consecutive_errors: int = 5
    """Raise after this many consecutive PBT iterations fail to save/load checkpoints,
    instead of silently training without PBT."""

    def __post_init__(self):
        self.change_range = tuple(self.change_range)
        legacy_thresholds = self.threshold_std is not None or self.threshold_abs is not None
        if not self.replace_rule:
            self.replace_rule = "threshold" if legacy_thresholds else "dexpbt"
            if legacy_thresholds:
                print("PbtCfg: threshold_std/threshold_abs set without replace_rule: using the 'threshold' "
                      "rule; set replace_rule explicitly ('dexpbt' is the DexPBT rank-based rule)")
        self.threshold_std = 0.10 if self.threshold_std is None else self.threshold_std
        self.threshold_abs = 0.05 if self.threshold_abs is None else self.threshold_abs
        if self.replace_rule not in REPLACE_RULES:
            raise ValueError(f"pbt.replace_rule must be one of {REPLACE_RULES}, got {self.replace_rule!r}")
        for name in ("replace_fraction_worst", "replace_fraction_best"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"pbt.{name} must be in (0, 1], got {value}")

    @classmethod
    def from_dict(cls, d):
        """Build from a config dict, ignoring (and reporting) unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            print(f"PbtCfg: ignoring unknown config keys {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})
