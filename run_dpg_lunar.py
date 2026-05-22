"""Run 3 LunarLander experiments: baseline PPO, DPG-old, DPG-curr, then plot comparison."""
import subprocess
import sys
import os
import glob

def run_training(config_path, name):
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
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
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, prefix, color in [
        ('PPO Baseline', 'Lunar_baseline', 'blue'),
        ('DPG (old probs)', 'Lunar_dpg_old', 'red'),
        ('DPG (curr probs)', 'Lunar_dpg_curr', 'green'),
    ]:
        matches = sorted(glob.glob(f"runs/{prefix}_*/summaries/events.out.tfevents.*"))
        if not matches:
            print(f"No data for {label}")
            continue
        ea = EventAccumulator(os.path.dirname(matches[-1]))
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        rtags = [t for t in tags if t == 'rewards/step']
        if not rtags:
            rtags = [t for t in tags if 'reward' in t.lower()]
        if not rtags:
            print(f"No reward tag for {label}")
            continue
        evts = ea.Scalars(rtags[0])
        epochs = list(range(1, len(evts)+1))
        values = [e.value for e in evts]
        ax.plot(epochs, values, label=f'{label} ({values[-1]:.1f})', color=color, alpha=0.8)
        print(f"{label}: {len(evts)} epochs, final={values[-1]:.1f}")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Reward')
    ax.set_title('LunarLander: PPO vs Delightful Policy Gradient (seed=42)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('dpg_lunar_comparison.png', dpi=150)
    print(f"\nPlot saved to dpg_lunar_comparison.png")


if __name__ == "__main__":
    configs = [
        ("rl_games/configs/lunar_baseline.yaml", "Lunar Baseline"),
        ("rl_games/configs/lunar_dpg_old.yaml", "Lunar DPG (old)"),
        ("rl_games/configs/lunar_dpg_curr.yaml", "Lunar DPG (curr)"),
    ]

    for config_path, name in configs:
        run_training(config_path, name)

    print("\n\nAll training runs complete. Generating comparison plot...")
    plot_comparison()
