"""Run 3 Walker2d experiments: baseline PPO, DPG-old, DPG-curr, then plot comparison."""
import subprocess
import sys
import os
import glob
from collections import defaultdict

def run_training(config_path, name):
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    # Use inline script with Runner API instead of rl_games.runner module
    script = f"""
import yaml
from rl_games.torch_runner import Runner
with open('{config_path}') as f:
    config = yaml.safe_load(f)
r = Runner()
r.load(config)
r.reset()
r.run({{'train': True}})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    return result.returncode

def plot_comparison():
    """Extract reward curves from TensorBoard logs and plot."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dirs = {
        'PPO Baseline': 'Walker2d_baseline',
        'DPG (old probs)': 'Walker2d_dpg_old',
        'DPG (curr probs)': 'Walker2d_dpg_curr',
    }

    plt.figure(figsize=(12, 6))

    for label, prefix in run_dirs.items():
        # Find the run directory
        matches = sorted(glob.glob(f"runs/{prefix}_*/summaries/events.out.tfevents.*"))
        if not matches:
            print(f"No TensorBoard log found for {label} (prefix: {prefix})")
            continue

        event_file = matches[-1]  # latest run
        run_dir = os.path.dirname(os.path.dirname(event_file))
        print(f"{label}: {run_dir}")

        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()

        # Try common reward tag names
        tags = ea.Tags().get('scalars', [])
        reward_tag = None
        for candidate in ['rewards/mean', 'reward', 'episode_rewards/mean',
                          'rewards/iter', 'episode_lengths/mean']:
            if candidate in tags:
                reward_tag = candidate
                break

        if reward_tag is None:
            # Find any tag with 'reward' in it
            reward_tags = [t for t in tags if 'reward' in t.lower()]
            if reward_tags:
                reward_tag = reward_tags[0]
            else:
                print(f"  Available tags: {tags[:20]}")
                print(f"  No reward tag found for {label}")
                continue

        print(f"  Using tag: {reward_tag}")
        events = ea.Scalars(reward_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        plt.plot(steps, values, label=label, alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title('Walker2d: PPO vs Delightful Policy Gradient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dpg_comparison.png', dpi=150)
    print(f"\nPlot saved to dpg_comparison.png")


if __name__ == "__main__":
    configs = [
        ("rl_games/configs/mujoco/walker2d_baseline.yaml", "PPO Baseline"),
        ("rl_games/configs/mujoco/walker2d_dpg_old.yaml", "DPG (old probs)"),
        ("rl_games/configs/mujoco/walker2d_dpg_curr.yaml", "DPG (curr probs)"),
    ]

    for config_path, name in configs:
        run_training(config_path, name)

    print("\n\nAll training runs complete. Generating comparison plot...")
    plot_comparison()
