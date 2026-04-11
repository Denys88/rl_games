"""Global Triton kernel configuration for rl_games.

Controls whether custom Triton kernels are used in place of PyTorch
implementations. Triton kernels can provide significant speedups for
GPU-heavy workloads (large batch sizes, many environments).

Usage:
    from rl_games.triton_config import USE_TRITON
    if USE_TRITON:
        result = triton_gae(...)
    else:
        result = torch_gae(...)

Set via environment variable or modify directly:
    export RL_GAMES_USE_TRITON=1
"""

import os

USE_TRITON = bool(int(os.environ.get('RL_GAMES_USE_TRITON', '0')))

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    if USE_TRITON:
        import warnings
        warnings.warn('RL_GAMES_USE_TRITON=1 but triton is not installed. Falling back to PyTorch.')
        USE_TRITON = False
