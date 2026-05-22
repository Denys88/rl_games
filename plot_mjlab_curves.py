"""Plot training curves for all MJLab runs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

run_dirs = sorted(glob.glob("runs/MJLab*Go1*/summaries"))
print(f"Found {len(run_dirs)} runs")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MJLab Training Curves", fontsize=16, fontweight='bold')

metrics = [
    ('rewards/iter', 'Reward per Iteration'),
    ('losses/a_loss', 'Actor Loss'),
    ('losses/c_loss', 'Critic Loss'),
    ('episode_lengths/iter', 'Episode Length'),
]

for run_dir in run_dirs:
    run_name = run_dir.split('/')[1]
    short_name = run_name.replace('_Velocity_', ' ').replace('MJLab_', '')
    print(f"Loading {run_name}...")

    ea = EventAccumulator(run_dir)
    ea.Reload()
    available_tags = ea.Tags().get('scalars', [])
    print(f"  Tags: {available_tags[:10]}...")

    for ax, (tag, title) in zip(axes.flat, metrics):
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, label=short_name, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        else:
            # Try partial match
            matches = [t for t in available_tags if tag.split('/')[-1] in t]
            if matches:
                events = ea.Scalars(matches[0])
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=short_name, alpha=0.8)
                ax.set_title(title + f" ({matches[0]})")
                ax.set_xlabel('Epoch')
                ax.grid(True, alpha=0.3)

for ax in axes.flat:
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("mjlab_training_curves.png", dpi=150, bbox_inches='tight')
print("Saved mjlab_training_curves.png")
