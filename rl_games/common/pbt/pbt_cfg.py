# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field, fields


@dataclass
class PbtCfg:
    """
    Population-Based Training (PBT) configuration.

    Leaders are policies with score > max(mean + threshold_std*std, mean + threshold_abs).
    Underperformers are policies with score < min(mean - threshold_std*std, mean - threshold_abs).
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
    If reward is stationary, a term that corresponds to task success is usually enough; with
    non-stationary rewards, prefer a true task objective."""

    interval_steps: int = 100_000
    """Environment steps between PBT iterations (save, compare, replace/mutate)."""

    threshold_std: float = 0.10
    """Std-based margin k in max(mean ± k·std, mean ± threshold_abs) for leader/underperformer cuts."""

    threshold_abs: float = 0.05
    """Absolute margin A in max(mean ± threshold_std·std, mean ± A) for leader/underperformer cuts."""

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
    """

    launcher: str = ""
    """Executable used to re-exec the training process on restart. Empty = sys.executable.
    Isaac Lab / Isaac Sim workflows should point this at their python.sh wrapper."""

    def __post_init__(self):
        self.change_range = tuple(self.change_range)

    @classmethod
    def from_dict(cls, d):
        """Build from a config dict, ignoring (and reporting) unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            print(f"PbtCfg: ignoring unknown config keys {unknown}")
        return cls(**{k: v for k, v in d.items() if k in known})
