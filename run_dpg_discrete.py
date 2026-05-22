"""Run vanilla DG on discrete envs — the paper's intended setting."""
import subprocess
import sys
import os
import glob
import yaml
import copy

CARTPOLE_BASE = {
    'params': {
        'torch_threads': 4,
        'seed': 42,
        'algo': {'name': 'a2c_discrete'},
        'model': {'name': 'discrete_a2c'},
        'network': {
            'name': 'actor_critic',
            'separate': True,
            'space': {'discrete': None},
            'mlp': {
                'units': [64, 64],
                'activation': 'relu',
                'initializer': {'name': 'default'},
            },
        },
        'config': {
            'env_name': 'CartPole-v1',
            'reward_shaper': {'scale_value': 1.0},
            'normalize_advantage': True,
            'gamma': 0.99,
            'tau': 0.9,
            'grad_norm': 1.0,
            'entropy_coef': 0.01,
            'truncate_grads': True,
            'e_clip': 0.2,
            'clip_value': True,
            'num_actors': 16,
            'horizon_length': 128,
            'minibatch_size': 512,
            'critic_coef': 1,
            'normalize_input': False,
            'device': 'cuda',
            'max_epochs': 300,
            'score_to_win': 500,
        },
    }
}

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
            'mini_epochs': 4,
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

# (name, base, dpg, temp, kl, lr_schedule, lr, me)
EXPERIMENTS = [
    # CartPole
    ('CP_ppo', CARTPOLE_BASE, None, 1.0, 0.0, 'None', 8e-4, 4),
    ('CP_dg_vanilla', CARTPOLE_BASE, 'vanilla', 1.0, 0.0, 'None', 8e-4, 4),
    ('CP_dg_kl01', CARTPOLE_BASE, 'vanilla', 1.0, 0.1, 'None', 8e-4, 4),
    ('CP_dg_me1', CARTPOLE_BASE, 'vanilla', 1.0, 0.0, 'None', 8e-4, 1),
    # Pong — paper's method: vanilla DG, eta=1, no KL, single epoch
    ('Pong_ppo_v2', PONG_BASE, None, 1.0, 0.0, 'adaptive', 3e-4, 4),
    ('Pong_dg_vanilla_me1', PONG_BASE, 'vanilla', 1.0, 0.0, 'None', 3e-4, 1),
    ('Pong_dg_vanilla_me4', PONG_BASE, 'vanilla', 1.0, 0.0, 'None', 3e-4, 4),
    ('Pong_dg_kl01_me4', PONG_BASE, 'vanilla', 1.0, 0.1, 'None', 3e-4, 4),
]


def run_experiment(name, base, dpg, temp, kl, lr_sched, lr, me):
    config = copy.deepcopy(base)
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
    print(f"{name}: dpg={dpg}, temp={temp}, kl={kl}, lr={lr}, sched={lr_sched}, me={me}")
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

    # CartPole
    cp_runs = [e for e in EXPERIMENTS if e[0].startswith('CP_')]
    colors = ['blue', 'red', 'green', 'purple']
    for i, (name, _, dpg, temp, kl, lrs, lr, me) in enumerate(cp_runs):
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 5:
                    vals = [e.value for e in evts]
                    ax1.plot(range(1, len(evts)+1), vals,
                            label=f'{name} ({vals[-1]:.0f})', color=colors[i],
                            linewidth=2, alpha=0.8)
                    print(f'{name}: {len(evts)} epochs, final={vals[-1]:.1f}')
                    break
        else:
            print(f'{name}: no data')
    ax1.set_title('CartPole-v1 (discrete)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Reward'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Pong
    pong_runs = [e for e in EXPERIMENTS if e[0].startswith('Pong_')]
    colors2 = ['blue', 'red', 'green', 'purple']
    for i, (name, _, dpg, temp, kl, lrs, lr, me) in enumerate(pong_runs):
        matches = sorted(glob.glob(f'runs/{name}_*/summaries/events.out.tfevents.*'))
        for m in reversed(matches):
            ea = EventAccumulator(os.path.dirname(m))
            ea.Reload()
            rtags = [t for t in ea.Tags().get('scalars', []) if t == 'rewards/step']
            if rtags:
                evts = ea.Scalars(rtags[0])
                if len(evts) > 5:
                    vals = [e.value for e in evts]
                    ax2.plot(range(1, len(evts)+1), vals,
                            label=f'{name} ({vals[-1]:.1f})', color=colors2[i],
                            linewidth=2, alpha=0.8)
                    print(f'{name}: {len(evts)} epochs, final={vals[-1]:.1f}')
                    break
        else:
            print(f'{name}: no data')
    ax2.set_title('Pong (discrete, Atari)')
    ax2.set_xlabel('Epoch'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle('Delightful Policy Gradient — Discrete Envs (paper setting)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('dpg_discrete_comparison.png', dpi=150)
    print('\nSaved dpg_discrete_comparison.png')


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_experiment(*exp)

    print("\n\nAll experiments done. Plotting...")
    plot_comparison()
