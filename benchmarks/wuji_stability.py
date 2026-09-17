"""Generate controlled Wuji stability ablations; this script never starts training.

Run from the repository root:
    python benchmarks/wuji_stability.py --output experiments/wuji/stability/configs

All arms use the original task, raw actions/rewards and 8192*40*5000 frames.
The reference_like arm matches major recipe choices, not every implementation
detail (normalizer schedule, minibatch ordering and optimizer coupling differ).
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'control': 'September E/F recipe on corrected code, with diagnostics enabled',
    'hard_clip': 'Only replace smooth PPO clipping with the reference hard clamp',
    'no_actor_value': 'Only remove the auxiliary critic loss from the actor trunk',
    'fp32': 'Only disable autocast for both actor and central critic',
    'rollout_kl': 'Only retain the rollout Gaussian as the adaptive KL reference',
    'no_cv_clip': 'Only disable central critic value clipping',
    'fixed_lr': 'Only replace adaptive actor learning rate with fixed 1e-4',
    'reference_like': 'Reference widths, geometry, hard clipping, raw values, fp32 and fixed LR',
}


def make_config(base, arm, seed, device, train_dir):
    params = copy.deepcopy(base['params'])
    params['seed'] = seed
    params['network']['mlp']['units'] = [1024, 512, 256]
    cfg = params['config']
    cfg.update({
        'name': f'wuji_stability_{arm}_seed{seed}',
        'device': device, 'num_actors': 8192, 'horizon_length': 40,
        'max_epochs': 5000, 'minibatch_size': 16384, 'mini_epochs': 4,
        'learning_rate': 1e-4, 'lr_schedule': 'adaptive',
        'min_lr': 5e-5, 'max_lr': 2e-4, 'kl_threshold': .01,
        'schedule_type': 'per_minibatch', 'kl_reference': 'previous_mini_epoch',
        'mixed_precision': True, 'use_experimental_cv': True,
        'use_smooth_clamp': True, 'normalize_value': True,
        'torch_compile': False,
        'use_diagnostics': True, 'multi_gpu': False,
        'clip_actions': False, 'train_dir': str(train_dir),
    })
    cfg['reward_shaper'] = {'scale_value': 1.0}
    cfg['env_config'].update({'device': device, 'seed': seed})
    cv = cfg['central_value_config']
    cv.update({'mixed_precision': True, 'minibatch_size': 16384,
               'mini_epochs': 4, 'learning_rate': 1e-4, 'clip_value': True})
    cv['network']['mlp']['units'] = [1024, 1024, 512, 256]
    if arm == 'hard_clip':
        cfg['use_smooth_clamp'] = False
    elif arm == 'no_actor_value':
        cfg['use_experimental_cv'] = False
    elif arm == 'fp32':
        cfg['mixed_precision'] = cv['mixed_precision'] = False
    elif arm == 'rollout_kl':
        cfg['kl_reference'] = 'rollout'
    elif arm == 'no_cv_clip':
        cv['clip_value'] = False
    elif arm == 'fixed_lr':
        cfg['lr_schedule'] = None
    elif arm == 'reference_like':
        params['network']['mlp']['units'] = [512, 256, 128]
        cv['network']['mlp']['units'] = [512, 512, 256, 128]
        cfg.update({'use_smooth_clamp': False, 'use_experimental_cv': False,
                    'normalize_value': False, 'lr_schedule': None,
                    'bounds_loss_coef': 0.0, 'kl_reference': 'rollout'})
        cfg['mixed_precision'] = cv['mixed_precision'] = False
        cfg['minibatch_size'] = cv['minibatch_size'] = 10240
        cv['clip_value'] = False
    elif arm != 'control':
        raise ValueError(f'Unknown arm: {arm}')
    return {'params': params}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-config', type=Path,
                        default=ROOT / 'rl_games/configs/mjlab/ppo_wujihand_reorient.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
    parser.add_argument('--arms', choices=list(ARMS), nargs='+', default=list(ARMS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-dir', default='runs/wuji-stability')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.base_config.read_text())
    commands, runs = [], []
    for arm in args.arms:
        for seed in args.seeds:
            path = args.output / f'{arm}_seed{seed}.yaml'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}; choose a fresh output directory')
            config = make_config(base, arm, seed, args.device, args.train_dir)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            commands.append(shlex.join([sys.executable, 'runner.py', '--train', '--file', str(path)]))
            runs.append({'arm': arm, 'seed': seed, 'config': str(path), 'description': ARMS[arm]})
    # These hashes distinguish corrected code from an older clean commit with
    # the same name; do not confuse a new-code control with a historical run.
    sources = ['rl_games/algos_torch/models.py', 'rl_games/algos_torch/torch_ext.py',
               'rl_games/common/a2c_common.py', 'rl_games/algos_torch/a2c_continuous.py',
               'rl_games/common/diagnostics.py']
    manifest = {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'frames_per_run': 8192 * 40 * 5000,
        'task_changed': False,
        'training_started': False,
        'runs': runs,
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'commands.txt').write_text('\n'.join(commands) + '\n')
    print(f'Wrote {len(runs)} configs and commands to {args.output}; no training started.')


if __name__ == '__main__':
    main()
