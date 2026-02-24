"""
PPO sweep on Ant-v4 (envpool): different architectures and gamma values.

3 architectures x 2 gammas x 3 seeds = 18 total runs.

Usage:
    cd config_manager
    python run_ant_sweep.py
"""
import os
from config import Config
from experiment_manager import ExperimentManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(BASE_DIR, "..", "rl_games", "configs", "mujoco", "ant_envpool.yaml")
RUNNER_SCRIPT = os.path.join(BASE_DIR, "..", "runner.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "ant_ppo_sweep")

em = ExperimentManager(
    base_config_path=BASE_CONFIG,
    output_dir=OUTPUT_DIR,
    runner_script=RUNNER_SCRIPT,
)

# Architecture sweep
em.sweep.grid("params.network.mlp.units", [
    [256, 128, 64],   # original
    [256, 256],        # wider 2-layer
    [512, 256],        # large 2-layer
])

# Gamma sweep
em.sweep.grid("params.config.gamma", [0.98, 0.99])

# Short runs for testing — remove this line for full training
em.sweep.grid("params.config.max_epochs", [500])

# 3 architectures x 2 gammas x 3 seeds = 18 runs
em.run(seeds=[42, 123, 777])

print(em.leaderboard())
