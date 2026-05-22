"""Fair Pong comparison — same LR schedule for all."""
import subprocess
import sys
import os
import glob
import yaml
import copy

PONG_BASE = {
    'params': {
        'seed': 42,
        'algo': {'name': 'a2c_discrete'},
        'model': {'name': 'discrete_a2c'},
        'network': {
            'name': 'actor_critic',
            'separate': False,
            'space': {'discrete': None},
            'cnn': {
                'type': 'conv2d',
                'activation': 'elu',
                'permute_input': False,
                'initializer': {'name': 'default'},
                'convs': [
                    {'filters': 32, 'kernel_size': 8, 'strides': 4, 'padding': 0},
                    {'filters': 64, 'kernel_size': 4, 'strides': 2, 'padding': 0},
                    {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 0},
                ],
            },
            'mlp': {
                'units': [512],
                'activation': 'elu',
                'initializer': {'name': 'orthogonal_initializer', 'gain': 1.41421356237},
            },
        },
        'config': {
            'env_name': 'atari_gymnasium',
            'score_to_win': 20.0,
            'normalize_value': True,
            'normalize_input': True,
            'reward_shaper': {'min_val': -1, 'max_val': 1},
            'normalize_advantage': True,
            'gamma': 0.99,
            'tau': 0.95,
            'grad_norm': 1.0,
            'entropy_coef': 0.01,
            'truncate_grads': True,
            'e_clip': 0.2,
            'clip_value': False,
            'num_actors': 64,
            'horizon_length': 128,
            'minibatch_size': 2048,
            'critic_coef': 2,
            'max_epochs': 500,
            'device': 'cuda:0',
            'env_config': {
                'env_name': 'ALE/Pong-v5',
                'frameskip': 1,
                'use_async': False,
            },
            'player': {'render': True, 'games_num': 10, 'deterministic': True},
        },
    }
}

EXPERIMENTS = [
    # All constant LR
    ('Pong_ppo_const', None, 1.0, 0.0, 'None', 3e-4, 4),
    ('Pong_dg_const_me4', 'vanilla', 1.0, 0.0, 'None', 3e-4, 4),
    ('Pong_dg_kl01_const', 'vanilla', 1.0, 0.1, 'None', 3e-4, 4),
    # All adaptive LR
    ('Pong_dg_adaptive_me4', 'vanilla', 1.0, 0.0, 'adaptive', 3e-4, 4),
    ('Pong_dg_kl01_adaptive', 'vanilla', 1.0, 0.1, 'adaptive', 3e-4, 4),
]


def run_experiment(name, dpg, temp, kl, lr_sched, lr, me):
    config = copy.deepcopy(PONG_BASE)
    config['params']['config']['name'] = name
    config['params']['config']['learning_rate'] = lr
    config['params']['config']['lr_schedule'] = lr_sched
    config['params']['config']['mini_epochs'] = me
    config['params']['config']['kl_threshold'] = 0.01

    if dpg:
        config['params']['config']['delightful_pg'] = dpg
        config['params']['config']['dpg_temperature'] = temp
        config['params']['config']['dpg_kl_coef'] = kl

    config_path = f'/tmp/{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'='*60}")
    print(f"{name}: dpg={dpg}, kl={kl}, sched={lr_sched}, me={me}")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Constant LR comparison
    const_runs = [
        ('Pong_ppo_const', 'PPO', 'blue'),
        ('Pong_dg_const_me4', 'DG vanilla', 'red'),
        ('Pong_dg_kl01_const', 'DG + KL=0.1', 'green'),
    ]
    for name, label, color in const_runs:
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 5:
                    vals = [e.value for e in evts]
                    ax1.plot(range(1, len(evts)+1), vals, label=f'{label} ({vals[-1]:.1f})', color=color, linewidth=2, alpha=0.8)
                    print(f'{name}: final={vals[-1]:.1f}')
                    break
    ax1.set_title('Constant LR (3e-4)'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Reward'); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    # Adaptive LR comparison
    adapt_runs = [
        ('Pong_ppo_v2', 'PPO', 'blue'),
        ('Pong_dg_adaptive_me4', 'DG vanilla', 'red'),
        ('Pong_dg_kl01_adaptive', 'DG + KL=0.1', 'green'),
    ]
    for name, label, color in adapt_runs:
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 5:
                    vals = [e.value for e in evts]
                    ax2.plot(range(1, len(evts)+1), vals, label=f'{label} ({vals[-1]:.1f})', color=color, linewidth=2, alpha=0.8)
                    print(f'{name}: final={vals[-1]:.1f}')
                    break
    ax2.set_title('Adaptive LR'); ax2.set_xlabel('Epoch'); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    fig.suptitle('Pong: Fair LR Comparison — PPO vs DG', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('dpg_pong_fair.png', dpi=150)
    print('\nSaved dpg_pong_fair.png')


if __name__ == "__main__":
    for name, dpg, temp, kl, lrs, lr, me in EXPERIMENTS:
        run_experiment(name, dpg, temp, kl, lrs, lr, me)

    print("\n\nAll experiments done. Plotting...")
    plot_comparison()
