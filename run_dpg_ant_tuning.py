"""Hyperparameter tuning for DG on Ant-v4."""
import subprocess
import sys
import os
import glob
import yaml
import copy

BASE_CONFIG = {
    'params': {
        'torch_threads': 4,
        'seed': 42,
        'algo': {'name': 'a2c_continuous'},
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {
            'name': 'actor_critic',
            'separate': False,
            'space': {
                'continuous': {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0},
                    'fixed_sigma': True,
                }
            },
            'mlp': {
                'units': [256, 128, 64],
                'activation': 'elu',
                'initializer': {'name': 'default'},
            },
        },
        'config': {
            'env_name': 'openai_gym',
            'normalize_input': True,
            'normalize_value': True,
            'value_bootstrap': True,
            'reward_shaper': {'scale_value': 0.1},
            'normalize_advantage': True,
            'gamma': 0.99,
            'tau': 0.95,
            'learning_rate': 3e-4,
            'lr_schedule': 'adaptive',
            'kl_threshold': 0.008,
            'grad_norm': 1.0,
            'entropy_coef': 0.0,
            'truncate_grads': True,
            'e_clip': 0.2,
            'clip_value': False,
            'num_actors': 64,
            'horizon_length': 128,
            'minibatch_size': 2048,
            'critic_coef': 2,
            'use_smooth_clamp': True,
            'bound_loss_type': 'regularisation',
            'bounds_loss_coef': 0.0,
            'device': 'cuda:0',
            'max_epochs': 1000,
            'env_config': {'name': 'Ant-v4', 'seed': 42},
            'player': {'render': True},
        },
    }
}

# Ant needs stronger KL and maybe higher temp (softer gate)
EXPERIMENTS = [
    # (name, temp, mini_epochs, kl_coef, lr)
    # Strong KL — maybe Ant needs much more regularization
    ('Ant_dg_t01_me5_kl10', 0.1, 5, 1.0, 3e-4),
    # Soft gate + moderate KL
    ('Ant_dg_t50_me5_kl01', 5.0, 5, 0.1, 3e-4),
    # Very soft gate (almost PPO-like) + KL
    ('Ant_dg_t100_me5_kl01', 100.0, 5, 0.1, 3e-4),
    # Lower LR + best Walker2d params
    ('Ant_dg_t01_me5_kl01_lr1e4', 0.1, 5, 0.1, 1e-4),
    # 1 mini epoch (pure on-policy) + gate
    ('Ant_dg_t01_me1_kl0', 0.1, 1, 0.0, 3e-4),
]


def run_experiment(name, temp, mini_epochs, kl_coef, lr):
    config = copy.deepcopy(BASE_CONFIG)
    config['params']['config']['name'] = name
    config['params']['config']['delightful_pg'] = 'vanilla'
    config['params']['config']['dpg_temperature'] = temp
    config['params']['config']['dpg_kl_coef'] = kl_coef
    config['params']['config']['mini_epochs'] = mini_epochs
    config['params']['config']['learning_rate'] = lr

    config_path = f'/tmp/{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'='*60}")
    print(f"{name}: temp={temp}, me={mini_epochs}, kl={kl_coef}, lr={lr}")
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
    result = subprocess.run([sys.executable, "-c", script],
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")


def plot_comparison():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))

    # Include baseline and previous runs
    all_names = ['Ant_baseline', 'Ant_dg_t01_me5_kl01'] + [e[0] for e in EXPERIMENTS]
    colors = ['blue', 'gray', 'red', 'green', 'purple', 'orange', 'brown']

    for i, name in enumerate(all_names):
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 10:
                    vals = [e.value for e in evts]
                    lw = 2.5 if i == 0 else 2
                    ls = '-' if i <= 1 else '-'
                    ax.plot(range(1, len(evts)+1), vals,
                            label=f'{name} ({vals[-1]:.0f})',
                            color=colors[i % len(colors)],
                            linewidth=lw, alpha=0.8)
                    print(f'{name}: {len(evts)} epochs, final={vals[-1]:.1f}')
                    break
        else:
            print(f'{name}: no data')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.set_title('Ant-v4: DG Hyperparameter Tuning (seed=42)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('dpg_ant_tuning.png', dpi=150)
    print('\nSaved dpg_ant_tuning.png')


if __name__ == "__main__":
    for name, temp, me, kl, lr in EXPERIMENTS:
        run_experiment(name, temp, me, kl, lr)

    print("\n\nAll experiments done. Plotting...")
    plot_comparison()
