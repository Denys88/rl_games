"""Isolate auxiliary critic gradients in the Wuji policy's shared trunk.

This is a synthetic batch, not a Wuji environment run or proof of the cause of
observed training failures. The shipped Wuji network recipe and real PPO loss
are used, with zero advantages, entropy/bounds coefficients and normalization
disabled. Only the auxiliary value loss can change the policy. Hard and smooth
PPO clipping are both exercised against the same immutable rollout log-probs.

Example (JSON on stdout, model construction messages on stderr):
    python benchmarks/wuji_aux_critic_probe.py --steps 80 --device cpu
"""

import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch.torch_ext import policy_kl
from rl_games.common import common_losses


def run_arm(initial_model, observations, rollout, cfg, steps, auxiliary, actor_loss):
    model = copy.deepcopy(initial_model)
    # calc_losses uses only these agent fields for this feedforward probe.
    agent = SimpleNamespace(
        model=model, has_value_loss=auxiliary, _ddp_model=None, ppo=True,
        clip_value=cfg['clip_value'], bound_loss_type='none',
        ppo_device=observations.device, critic_coef=cfg['critic_coef'],
        entropy_coef=0.0, bounds_loss_coef=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=1e-8)
    advantages = torch.zeros(observations.shape[0], device=observations.device)
    targets = torch.full_like(rollout['values'], -3.0)

    for _ in range(steps):
        output = model({'obs': observations, 'prev_actions': rollout['actions']})
        ratio = torch.exp(rollout['neglogpacs'] - output['prev_neglogp'])
        if not torch.isfinite(ratio).all():
            raise RuntimeError('Nonfinite likelihood ratio invalidates the zero-advantage isolation.')
        loss, actor_term, *_ = A2CAgent.calc_losses(
            agent, actor_loss, rollout['neglogpacs'], output['prev_neglogp'],
            advantages, cfg['e_clip'], rollout['values'], output['values'],
            targets, output['mus'], output['entropy'], None,
        )
        if actor_term.item() != 0.0:
            raise RuntimeError('Expected exactly zero PPO surrogate loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg['truncate_grads']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_norm'])
        optimizer.step()

    with torch.no_grad():
        final = model({'obs': observations, 'prev_actions': rollout['actions']})
        drift = (final['mus'] - rollout['mus']).abs()
        metrics = {
            'mean_abs_mu_drift': drift.mean().item(),
            'max_abs_mu_drift': drift.max().item(),
            'mean_abs_sigma_drift': (final['sigmas'] - rollout['sigmas']).abs().mean().item(),
            'rollout_kl_new_to_old': policy_kl(
                final['mus'], final['sigmas'], rollout['mus'], rollout['sigmas']).item(),
        }
    return metrics, {key: value.detach().clone() for key, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config_path = ROOT / 'rl_games/configs/mjlab/ppo_wujihand_reorient.yaml'
    params = yaml.safe_load(config_path.read_text())['params']
    cfg = params['config']
    with contextlib.redirect_stdout(sys.stderr):
        model = ModelBuilder().load(params).build({
            'input_shape': (64,), 'actions_num': 20, 'num_seqs': 128,
            'value_size': 1, 'normalize_input': False, 'normalize_value': False,
        }).to(device)
    observations = torch.randn(128, 64, device=device)
    with torch.no_grad():
        rollout = model({'obs': observations, 'is_train': False})
        rollout = {key: value.detach().clone() for key, value in rollout.items()
                   if isinstance(value, torch.Tensor)}
    snapshot = {key: value.clone() for key, value in rollout.items()}
    results, states = {}, {}
    for auxiliary in (False, True):
        for name, loss_fn in (('hard', common_losses.actor_loss),
                              ('smooth', common_losses.smoothed_actor_loss)):
            arm = f'{name}_auxiliary_{str(auxiliary).lower()}'
            results[arm], states[arm] = run_arm(
                model, observations, rollout, cfg, args.steps, auxiliary, loss_fn)
    rollout_unchanged = all(torch.equal(rollout[key], value) for key, value in snapshot.items())
    hard_smooth_equal = all(
        torch.equal(states[f'hard_auxiliary_{enabled}'][key], value)
        for enabled in ('false', 'true')
        for key, value in states[f'smooth_auxiliary_{enabled}'].items()
    )
    if not rollout_unchanged or not hard_smooth_equal:
        raise RuntimeError('Isolation failed: rollout mutated or zero-advantage clipping variants diverged.')
    print(json.dumps({
        'interpretation': 'Synthetic gradient interference probe; not evidence of observed run causality.',
        'seed': 42, 'device': str(device), 'steps': args.steps,
        'recipe': str(config_path.relative_to(ROOT)),
        'actor_units': params['network']['mlp']['units'],
        'batch_size': 128, 'synthetic_observation_dim': 64, 'actions': 20,
        'learning_rate': cfg['learning_rate'], 'critic_coef': cfg['critic_coef'],
        'grad_norm': cfg['grad_norm'], 'synthetic_return_target': -3.0,
        'advantages': 0.0, 'entropy_coef': 0.0, 'bounds_loss_coef': 0.0,
        'normalize_input': False, 'normalize_value': False,
        'precision': 'float32', 'rollout_tensors_unchanged': rollout_unchanged,
        'hard_smooth_final_parameters_equal': hard_smooth_equal,
        'results': results,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
