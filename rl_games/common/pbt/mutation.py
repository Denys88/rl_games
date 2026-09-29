# Ported from the Isaac Lab rl_games integration (isaaclab_rl); original DexPBT
# implementation from NVIDIA-Omniverse/IsaacGymEnvs (https://arxiv.org/abs/2305.12127).
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import random
from collections.abc import Callable
from typing import Any


def mutate_float(x: float, change_min: float = 1.1, change_max: float = 1.5) -> float:
    """Multiply or divide by a random factor in [change_min, change_max]."""
    k = random.uniform(change_min, change_max)
    return x / k if random.random() < 0.5 else x * k


def mutate_float_min_1(x: float, **kwargs) -> float:
    """mutate_float, floored at 1.0."""
    return max(1.0, mutate_float(x, **kwargs))


def mutate_eps_clip(x: float, **kwargs) -> float:
    """mutate_float, clamped to the PPO clip range [0.01, 0.3]."""
    return min(0.3, max(0.01, mutate_float(x, **kwargs)))


def mutate_mini_epochs(x: int, **kwargs) -> int:
    """Step the number of PPO passes by ±1, clamped to [1, 8]."""
    new_value = x + 1 if random.random() < 0.5 else x - 1
    return int(min(8, max(1, new_value)))


def mutate_discount(x: float, **kwargs) -> float:
    """Conservative change near 1.0 by mutating (1 - x) in [1.1, 1.2].

    The configured change_range is intentionally ignored: gamma-like params
    need much smaller steps than regular floats (DexPBT behavior).
    """
    inv = 1.0 - x
    new_inv = mutate_float(inv, change_min=1.1, change_max=1.2)
    return 1.0 - new_inv


MUTATION_FUNCS: dict[str, Callable[..., Any]] = {
    "mutate_float": mutate_float,
    "mutate_float_min_1": mutate_float_min_1,
    "mutate_eps_clip": mutate_eps_clip,
    "mutate_mini_epochs": mutate_mini_epochs,
    "mutate_discount": mutate_discount,
}


def mutate(
    params: dict[str, Any],
    mutations: dict[str, str],
    mutation_rate: float,
    change_range: tuple[float, float],
) -> dict[str, Any]:
    """Mutate whitelisted params. Integer params need an integer rule (mutate_mini_epochs)."""
    cmin, cmax = change_range
    out: dict[str, Any] = {}
    for name, val in params.items():
        fn_name = mutations.get(name)
        # skip if no rule or coin flip says "no"
        if fn_name is None or random.random() > mutation_rate:
            out[name] = val
            continue
        fn = MUTATION_FUNCS.get(fn_name)
        if fn is None:
            raise KeyError(f"Unknown mutation function: {fn_name!r}")
        out[name] = fn(val, change_min=cmin, change_max=cmax)
    return out
