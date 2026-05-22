#!/usr/bin/env python
"""Fast goal-counting eval for dm_soccer ant (no rendering).

Use this to quickly score a checkpoint — runs N matches, prints win/draw/loss,
mean goals/episode, mean shaped reward, episode length. Roughly 10x faster
than the rendering eval (no MuJoCo OpenGL pass, no mp4 encode).

Usage:
    python eval_dm_soccer_ant.py CHECKPOINT [--episodes N] [--opponent KIND]

Examples:
    # Quick health check on a Phase-1 (random opp) checkpoint:
    python eval_dm_soccer_ant.py runs/.../nn/dm_soccer_ant_random_opp.pth --episodes 20

    # Score Phase-2 (self-play) checkpoint vs frozen self:
    python eval_dm_soccer_ant.py runs/.../nn/dm_soccer_ant_selfplay.pth \
        --episodes 20 --opponent self \
        --config rl_games/configs/dm_control/ppo_soccer_ant_selfplay.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml


def _build_player(config_path: str, checkpoint: str):
    from rl_games.torch_runner import Runner
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['params']['config']['num_actors'] = 1
    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True   # see play_dm_soccer_ant.py for why
    return player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default='rl_games/configs/dm_control/ppo_soccer_ant.yaml')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--opponent', choices=['random', 'noop', 'self'], default='random')
    ap.add_argument('--opponent-checkpoint', default=None,
                    help='When set, use this ckpt for the opponent (different snapshot). '
                         'Implies --opponent self. Lets you do head-to-head matches like '
                         "'phaseB2 latest vs cur3 bootstrap' to prove self-play improved.")
    ap.add_argument('--time-limit', type=float, default=30.0)
    ap.add_argument('--no-field-box', action='store_true')
    ap.add_argument('--no-shaping', action='store_true',
                    help='turn off shaped reward to score on raw goals only')
    # Default is multi-turn (ball respawns after each goal). dm_soccer's native
    # single-goal-then-done mode loses scoring rate information — multi-turn is
    # what every trained config in this branch uses, so it's the right default.
    ap.add_argument('--terminate-on-goal', action='store_true',
                    help='end episode at first goal (rarely correct — only use '
                         'to match a config trained with terminate_on_goal=True).')
    ap.add_argument('--field-size', nargs=2, type=float, default=None,
                    metavar=('W', 'H'),
                    help='override pitch size, e.g. "--field-size 15 10" '
                         'to match v3 training (default: dm_soccer 32-48m × 24-36m).')
    ap.add_argument('--goal-size', nargs=3, type=float, default=None,
                    metavar=('D', 'W', 'H'),
                    help='override goal opening (depth, width, height); e.g. '
                         '"--goal-size 1.0 6.0 0.6" to widen goals to 60%% of '
                         'a 10m-wide pitch.')
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f'checkpoint not found: {args.checkpoint}')

    if args.opponent_checkpoint:
        # --opponent-checkpoint means head-to-head with a *different* snapshot.
        # Force --opponent self so the env loads weights into FrozenA2COpponent.
        args.opponent = 'self'

    import torch
    from rl_games.envs.dm_soccer import DMSoccerAntEnv

    # Read the training config — needed for the network arch AND the env_config
    # block (walker_type, field/goal size, team_size) so eval matches training.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_env_cfg = dict(cfg['params']['config'].get('env_config', {}))
    train_env_cfg.pop('ray_config', None)  # training-only, not an env arg

    if args.opponent == 'self':
        opponent_cfg = {
            'opponent_kind': 'policy',
            'opponent_factory_config': {
                'network_params': cfg['params']['network'],
                'model_name': cfg['params']['model']['name'],
                'normalize_input': cfg['params']['config'].get('normalize_input', True),
                'normalize_value': cfg['params']['config'].get('normalize_value', False),
                'pool_size': 1,
                'deterministic': True,
                'device': 'cpu',
            },
        }
    else:
        opponent_cfg = {'opponent_kind': args.opponent}

    # Start from the trained env_config, then apply eval-time overrides.
    env_kwargs = dict(train_env_cfg)
    env_kwargs.update(opponent_cfg)
    env_kwargs.update(
        time_limit=args.time_limit,
        enable_field_box=not args.no_field_box,
        terminate_on_goal=args.terminate_on_goal,
        shaped_reward=not args.no_shaping,
        seed=int(time.time()) & 0xFFFF,
    )
    if args.field_size is not None:
        env_kwargs['field_size'] = tuple(args.field_size)
    if args.goal_size is not None:
        env_kwargs['goal_size'] = tuple(args.goal_size)
    env = DMSoccerAntEnv(**env_kwargs)

    player = _build_player(args.config, args.checkpoint)

    if args.opponent == 'self':
        if args.opponent_checkpoint:
            opp_ckpt = torch.load(args.opponent_checkpoint, map_location='cpu',
                                   weights_only=False)
            opp_sd = opp_ckpt.get('model', opp_ckpt)
            opp_sd = {k.replace('_orig_mod.', ''): v for k, v in opp_sd.items()}
            env.update_weights({'model': opp_sd})
            print(f'opponent loaded from {args.opponent_checkpoint}')
        else:
            env.update_weights({'model': player.model.state_dict()})

    home_goals_total = 0
    away_goals_total = 0
    sum_rewards = []
    ep_lens = []
    wins = draws = losses = 0

    t0 = time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_len = 0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=player.device)
            with torch.no_grad():
                action = player.get_action(obs_t, is_deterministic=True)
            obs, r, term, trunc, info = env.step(action.cpu().numpy())
            ep_reward += float(np.asarray(r).sum())
            ep_len += 1
            if bool(term.any()) or bool(trunc.any()):
                home = int(info.get('home_goals', 0))
                away = int(info.get('away_goals', 0))
                home_goals_total += home
                away_goals_total += away
                sum_rewards.append(ep_reward)
                ep_lens.append(ep_len)
                if home > away:
                    wins += 1
                elif away > home:
                    losses += 1
                else:
                    draws += 1
                print(f'  ep {ep+1:2d}: len={ep_len:4d}  reward={ep_reward:+7.2f}  '
                      f'goals={home}-{away}')
                break

    elapsed = time.time() - t0
    n = args.episodes
    print('=' * 60)
    print(f'OPPONENT: {args.opponent}   EPISODES: {n}   TIME: {elapsed:.1f}s')
    print(f'  W/D/L:               {wins}/{draws}/{losses}   '
          f'win_rate={100*wins/n:.1f}%')
    print(f'  goals home/away:     {home_goals_total}/{away_goals_total}   '
          f'(mean home={home_goals_total/n:.2f}/ep, away={away_goals_total/n:.2f}/ep)')
    print(f'  goal_diff/ep:        {(home_goals_total - away_goals_total)/n:+.2f}')
    print(f'  shaped reward sum/ep: mean={np.mean(sum_rewards):+.2f}   '
          f'std={np.std(sum_rewards):.2f}')
    print(f'  ep_len:              mean={np.mean(ep_lens):.0f}   '
          f'min={min(ep_lens)}   max={max(ep_lens)}')


if __name__ == '__main__':
    main()
