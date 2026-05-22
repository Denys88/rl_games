"""Generate two clean Go1 training reward charts: Flat and Rough."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob


def load_reward(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    tag = next((t for t in tags if 'rewards/iter' in t or t.endswith('iter')), None)
    if tag is None:
        return None, None
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def plot_set(run_patterns, title, outfile):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, pattern in run_patterns:
        for d in sorted(glob.glob(pattern)):
            steps, vals = load_reward(d + '/summaries')
            if steps:
                ax.plot(steps, vals, label=label, alpha=0.85)
                break
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches='tight')
    print(f"Saved {outfile}")


plot_set(
    [
        ('Go1 Flat (v5, no curriculum)', 'runs/MJLab_Go1_Velocity_v5_no_curriculum_28-22-43-11'),
        ('Go1 Flat (v4)', 'runs/MJLab_Go1_Velocity_v4_28-15-29-57'),
        ('Go1 Flat CV', 'runs/MJLab_Go1_Velocity_Flat_CV_29-02-14-11'),
    ],
    'Go1 Flat Velocity — Training Reward',
    'go1_flat_training.png',
)

plot_set(
    [
        ('Go1 Rough', 'runs/MJLab_Go1_Velocity_Rough_28-23-09-02'),
        ('Go1 Rough (Central Value)', 'runs/MJLab_Go1_Velocity_Rough_CV_29-02-48-07'),
    ],
    'Go1 Rough Velocity — Training Reward',
    'go1_rough_training.png',
)
