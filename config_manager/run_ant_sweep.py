"""
PPO sweep on Ant-v4 (envpool): different architectures and gamma values.

With default parameters: 3 architectures x 3 activations x 5 seeds = 45 total runs.

Usage:
    cd config_manager
    python run_ant_sweep.py
    python run_ant_sweep.py --max_epochs 2000
    python run_ant_sweep.py --track --wandb_project ant_ppo_sweep
    python run_ant_sweep.py --config path/to/other_config.yaml --output_dir ./my_sweep
"""
import os
import argparse
from experiment_manager import ExperimentManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="PPO architecture + activation sweep on Ant-v4")
parser.add_argument("--config", default=os.path.join(BASE_DIR, "..", "rl_games", "configs", "mujoco", "ant_envpool.yaml"),
                    help="Path to base config YAML")
parser.add_argument("--runner", default=os.path.join(BASE_DIR, "..", "runner.py"),
                    help="Path to rl_games runner.py")
parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "experiments", "ant_ppo_sweep"),
                    help="Directory for experiment outputs")
parser.add_argument("--max_epochs", type=int, default=1000,
                    help="Training budget per run")
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42, 123, 777],
                    help="Random seeds for each variant")
parser.add_argument("--track", action="store_true",
                    help="Enable Weights & Biases experiment tracking")
parser.add_argument("--wandb_project", type=str, default="rl_games_sweeps",
                    help="W&B project name")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity (team or username)")
args = parser.parse_args()

# Build extra args for runner.py
extra_args = []
if args.track:
    extra_args += ["--track", "--wandb-project-name", args.wandb_project]
    if args.wandb_entity:
        extra_args += ["--wandb-entity", args.wandb_entity]

em = ExperimentManager(
    base_config_path=args.config,
    output_dir=args.output_dir,
    runner_script=args.runner,
    extra_args=extra_args,
)

# Architecture sweep
em.sweep.grid("params.network.mlp.units", [
    [256, 128],         # small 2-layer
    [256, 128, 64],     # original
    [256, 256],         # wider 2-layer
    # [512, 256],       # large 2-layer
    # [512, 256, 64],   # large 3-layer
])

# Activations sweep
em.sweep.grid("params.network.mlp.activation", ["relu", "elu", "swish"])

# Gamma sweep
# em.sweep.grid("params.config.gamma", [0.98, 0.99])

# Training budget (excluded from experiment names automatically)
em.sweep.grid("params.config.max_epochs", [args.max_epochs])

em.run(seeds=args.seeds)

em.print_leaderboard()
