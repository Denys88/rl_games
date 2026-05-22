"""Plot G1 flat velocity training curves from tensorboard logs."""
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("pip install tensorboard")
    exit(1)


def read_tb_scalar(log_dir, tag='rewards/iter'):
    """Read scalar from tensorboard events."""
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # Try different tag names
    available = ea.Tags().get('scalars', [])

    # Find the rewards tag
    reward_tag = None
    for t in available:
        if 'reward' in t.lower():
            reward_tag = t
            break

    if reward_tag is None:
        print(f"  Available tags: {available[:10]}")
        return None, None

    events = ea.Scalars(reward_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)


# Map of run directories to labels
RUNS = {
    'runs/MJLab_G1_Velocity_Flat_28-23-44-35': 'Baseline (11.5)',
    'runs/MJLab_G1_Velocity_Flat_CV_29-03-32-02': 'Central Value (8.4)',
}

# Find curriculum runs
for d in sorted(glob.glob('runs/MJLab_G1_Velocity_Flat_Curriculum*')):
    if os.path.exists(os.path.join(d, 'summaries')):
        RUNS[d] = 'Curriculum'

for d in sorted(glob.glob('runs/MJLab_G1_Velocity_Flat_RSL_*')):
    if os.path.exists(os.path.join(d, 'summaries')):
        name = os.path.basename(d)
        if 'NoCV' in name:
            RUNS[d] = 'RSL-style (no CV, ent=0.01)'
        elif 'v2' in name:
            RUNS[d] = 'RSL v2 (CV, ent=0.001, 30k)'
        else:
            RUNS[d] = 'RSL-style (CV, ent=0.01)'

print("Runs found:")
for d, label in RUNS.items():
    print(f"  {label}: {d}")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']

for i, (run_dir, label) in enumerate(RUNS.items()):
    summary_dir = os.path.join(run_dir, 'summaries')
    if not os.path.exists(summary_dir):
        print(f"  Skipping {label} - no summaries dir")
        continue

    steps, values = read_tb_scalar(summary_dir)
    if steps is None:
        print(f"  Skipping {label} - no reward data")
        continue

    # Smooth with rolling average
    window = max(1, len(values) // 100)
    if len(values) > window:
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        smooth_steps = steps[:len(smoothed)]
    else:
        smoothed = values
        smooth_steps = steps

    color = colors[i % len(colors)]
    ax.plot(smooth_steps, smoothed, label=label, color=color, linewidth=2)
    # Light original data
    ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.5)

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Reward', fontsize=13)
ax.set_title('G1 Humanoid Flat Velocity - Training Comparison', fontsize=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('g1_flat_comparison.png', dpi=150)
print("\nSaved: g1_flat_comparison.png")
