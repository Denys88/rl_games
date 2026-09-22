"""Compare one real PPO minibatch update at several learning rates.

Collects ONE rollout in a fresh Wuji environment, prepares its first minibatch,
then restores identical model/Adam/normalizer/RNG state before each LR trial.
An optional checkpoint supplies learner state, not the historical environment
or onset batch. This is a local response probe, not performance or causality
evidence. Smaller rollout/minibatch geometry also differs from full training.

Example (requires the MJLab/Wuji environment dependencies and a free GPU):
    .venv/bin/python benchmarks/wuji_lr_replay.py --device cuda:0 \
        --output /tmp/wuji_lr_replay.json

--warmup-updates performs optimizer updates on the SAME captured minibatch
before taking the shared snapshot; it does not collect more rollouts. This
can populate fresh Adam moments, but is not a trained-policy substitute.
"""

import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_games.algos_torch.torch_ext import policy_kl
from rl_games.torch_runner import Runner


def rng_state(device):
    device = torch.device(device)
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        # Only the selected GPU participates; do not initialize other GPUs.
        'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        'device': device,
    }


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], state['device'])


def capture_distribution(model, _inputs, output):
    """Detach at the forward boundary, before later steps mutate parameters."""
    result = {k: output[k].detach().float().clone()
              for k in ('mus', 'sigmas', 'prev_neglogp')}
    normalizer = getattr(model, 'running_mean_std', None)
    result['normalizer'] = (copy.deepcopy(normalizer.state_dict())
                            if normalizer is not None else {})
    return result


def replay_minibatch(agent, minibatch, learning_rates):
    """Use the production calc_gradients; no replacement loss or optimizer."""
    snapshot = {
        'model': copy.deepcopy(agent.model.state_dict()),
        'optimizer': copy.deepcopy(agent.optimizer.state_dict()),
        'rng': rng_state(agent.ppo_device),
    }
    masks = minibatch.get('rnn_masks')
    valid = (masks.reshape(-1).bool() if masks is not None else
             torch.ones(minibatch['actions'].shape[0], dtype=torch.bool,
                        device=minibatch['actions'].device))
    if not valid.any():
        raise ValueError('The captured minibatch has no valid rows.')

    def mean_kl(new_mu, new_sigma, old_mu, old_sigma):
        values = policy_kl(new_mu[valid].float(), new_sigma[valid].float(),
                           old_mu[valid].float(), old_sigma[valid].float(), False)
        return values.mean().item()

    results = []
    first_pre = None
    for lr in learning_rates:
        agent.model.load_state_dict(snapshot['model'])
        # load_state_dict can alias optimizer tensors: clone on EVERY restore.
        agent.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        agent.update_lr(lr)
        restore_rng(snapshot['rng'])
        agent.set_train()
        captured = []
        handle = agent.model.register_forward_hook(
            lambda model, inputs, output: captured.append(
                capture_distribution(model, inputs, output)))
        try:
            agent.calc_gradients(copy.deepcopy(minibatch))
        finally:
            handle.remove()
        if len(captured) != 2:
            raise RuntimeError(f'Expected pre/post optimizer forwards, got {len(captured)}.')
        pre, post = captured
        if first_pre is None:
            first_pre = pre
        pre_identical = all(torch.equal(pre[k], first_pre[k])
                            for k in ('mus', 'sigmas', 'prev_neglogp'))
        stats_unchanged = all(torch.equal(v, post['normalizer'][k])
                              for k, v in pre['normalizer'].items())
        if not pre_identical or not stats_unchanged:
            raise RuntimeError('Isolation failed: pre-policy or pre/post normalizer differs.')
        step_kl = mean_kl(post['mus'], post['sigmas'], pre['mus'], pre['sigmas'])
        scheduler_kl = agent.kl_schedule_value.item()
        if not math.isclose(step_kl, scheduler_kl, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError('Captured step KL does not match the production scheduler signal.')
        parameter_delta_sq = sum(
            (p.detach().float() - snapshot['model'][name].float()).square().sum().item()
            for name, p in agent.model.named_parameters())
        logratio = minibatch['old_logp_actions'][valid].float() - pre['prev_neglogp'][valid]
        results.append({
            'learning_rate': lr,
            'pre_vs_rollout_kl': mean_kl(pre['mus'], pre['sigmas'],
                                       minibatch['mu'], minibatch['sigma']),
            'post_vs_pre_kl': step_kl,
            'scheduler_kl': scheduler_kl,
            'post_vs_rollout_kl': mean_kl(post['mus'], post['sigmas'],
                                        minibatch['mu'], minibatch['sigma']),
            'parameter_delta_l2': math.sqrt(parameter_delta_sq),
            'mu_delta_abs_mean': (post['mus'][valid] - pre['mus'][valid]).abs().mean().item(),
            'sigma_delta_abs_mean': (post['sigmas'][valid] - pre['sigmas'][valid]).abs().mean().item(),
            'pre_logratio_abs_max': logratio.abs().max().item(),
            'pre_policy_identical_across_lrs': pre_identical,
            'normalizer_unchanged_pre_to_post': stats_unchanged,
        })
    return results


def minibatch_size(config, batch_size, num_envs):
    requested = int(config.get('minibatch_size',
                               num_envs * config.get('minibatch_size_per_env', 1)))
    if requested < 1:
        raise ValueError('Configured minibatch size must be positive.')
    # Preserve configured size when possible; otherwise take a valid divisor.
    return next(n for n in range(min(requested, batch_size), 0, -1)
                if batch_size % n == 0)


def run(args, train_dir):
    payload = yaml.safe_load(args.config.read_text())
    cfg = payload['params']['config']
    if payload['params']['algo']['name'] != 'a2c_continuous':
        raise ValueError('This probe requires continuous PPO.')
    batch_size = args.num_envs * args.horizon
    cfg.update({
        'name': 'wuji_lr_replay', 'device': args.device, 'num_actors': args.num_envs,
        'horizon_length': args.horizon, 'multi_gpu': False, 'torch_compile': False,
        'use_diagnostics': False, 'kl_schedule_source': 'optimizer_step',
        'lr_schedule': None, 'train_dir': train_dir, 'save_frequency': 0,
        'seq_length': 1,  # feedforward only; no sequence padding required
    })
    cfg['minibatch_size'] = minibatch_size(cfg, batch_size, args.num_envs)
    cfg.setdefault('env_config', {})['device'] = args.device
    if 'central_value_config' in cfg:
        cv = cfg['central_value_config']
        cv['minibatch_size'] = minibatch_size(cv, batch_size, args.num_envs)
        cv['torch_compile'] = False
    effective_config = copy.deepcopy(payload)
    runner = Runner()
    runner.load(payload)
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    checkpoint_epoch = None
    try:
        if args.checkpoint is not None:
            # Same trusted local checkpoint format as the training runner.
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            checkpoint_epoch = checkpoint.get('epoch')
            checkpoint['env_state'] = None  # never pretend this is an onset replay
            agent.set_full_state_weights(checkpoint, set_epoch=False)
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.set_eval()
        with torch.no_grad():
            rollout = agent.play_steps()
        agent.set_train()
        agent.prepare_dataset(rollout)
        # No central-critic fit is needed: targets/advantages are already fixed,
        # and its parameters are separate from the actor optimizer under test.
        minibatch = copy.deepcopy(agent.dataset[0])
        agent.update_lr(float(cfg['learning_rate']))
        for _ in range(args.warmup_updates):
            agent.set_train()
            agent.calc_gradients(copy.deepcopy(minibatch))
        results = replay_minibatch(agent, minibatch, args.lrs)
        return {
            'interpretation': (
                'One fresh-environment rollout; identical captured minibatch and learner '
                'state per LR. Not a historical onset replay or training-performance result.'),
            'config_path': str(args.config.resolve()),
            'effective_config': effective_config,
            'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
            'checkpoint_epoch': checkpoint_epoch,
            'device': args.device, 'seed': runner.seed,
            'torch_version': str(torch.__version__),
            'mixed_precision': agent.mixed_precision,
            'num_envs': args.num_envs, 'horizon': args.horizon,
            'rollouts_collected': 1, 'rollout_rows': batch_size,
            'minibatch_rows': int(minibatch['actions'].shape[0]),
            'warmup_updates_on_same_minibatch': args.warmup_updates,
            'warmup_learning_rate': float(cfg['learning_rate']),
            'central_critic_updates': 0,
            'advantage_abs_max': minibatch['advantages'].abs().max().item(),
            'results': results,
        }
    finally:
        agent.writer.close()
        agent.vec_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'rl_games/configs/mjlab/ppo_wujihand_reorient.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--lrs', type=float, nargs='+', default=[0, 5e-5, 1e-4, 2e-4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--warmup-updates', type=int, default=0)
    args = parser.parse_args()
    if args.num_envs < 1 or args.horizon < 1 or args.warmup_updates < 0:
        parser.error('Environment count/horizon must be positive; warmup updates nonnegative.')
    if any(not math.isfinite(lr) or lr < 0 for lr in args.lrs):
        parser.error('Learning rates must be finite and nonnegative.')
    print('Fresh-environment LR response probe; not a historical onset replay.', file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix='wuji-lr-replay-') as train_dir:
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args, train_dir)
    output = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
