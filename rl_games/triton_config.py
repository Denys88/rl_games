"""Global Triton kernel configuration for rl_games.

Triton kernels are enabled by default when triton is installed.
To disable, set the environment variable:

    export RLG_NO_TRITON=1
"""

import os

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

USE_TRITON = TRITON_AVAILABLE and not bool(int(os.environ.get('RLG_NO_TRITON', '0')))
