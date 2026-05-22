"""Run DPG experiments on Ant-v4."""
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
            'mini_epochs': 5,
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

EXPERIMENTS = [
    # (name, delightful_pg, dpg_temperature, mini_epochs, dpg_kl_coef)
    ('Ant_baseline', None, 1.0, 5, 0.0),
    ('Ant_dg_t05_me5', 'vanilla', 0.5, 5, 0.0),
    ('Ant_dg_t01_me5_kl01', 'vanilla', 0.1, 5, 0.1),
]


def run_experiment(name, dpg, temp, mini_epochs, kl_coef):
    config = copy.deepcopy(BASE_CONFIG)
    config['params']['config']['name'] = name
    config['params']['config']['mini_epochs'] = mini_epochs
    if dpg:
        config['params']['config']['delightful_pg'] = dpg
        config['params']['config']['dpg_temperature'] = temp
        config['params']['config']['dpg_kl_coef'] = kl_coef

    config_path = f'/tmp/{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'='*60}")
    print(f"Starting: {name}")
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

    colors = ['blue', 'green', 'red']
    for i, (name, dpg, temp, me, kl) in enumerate(EXPERIMENTS):
        label = name.replace('Ant_', '')
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 10:
                    vals = [e.value for e in evts]
                    ax.plot(range(1, len(evts)+1), vals,
                            label=f'{label} ({vals[-1]:.0f})', color=colors[i],
                            linewidth=2, alpha=0.8)
                    print(f'{name}: {len(evts)} epochs, final={vals[-1]:.1f}')
                    break
        else:
            print(f'{name}: no data')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.set_title('Ant-v4: PPO vs Vanilla DG (seed=42)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('dpg_ant_comparison.png', dpi=150)
    print('\nSaved dpg_ant_comparison.png')


if __name__ == "__main__":
    for name, dpg, temp, me, kl in EXPERIMENTS:
        run_experiment(name, dpg, temp, me, kl)

    print("\n\nAll experiments done. Plotting...")
    plot_comparison()
