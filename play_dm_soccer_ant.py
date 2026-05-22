#!/usr/bin/env python
"""Play and record dm_control 2v2 ant soccer with a trained rl_games policy.

Usage:
    python play_dm_soccer_ant.py CHECKPOINT [--out OUT.mp4] [--episodes N]
                                 [--opponent {random,noop,self}]
                                 [--config CFG.yaml]

By default we render against random opponents. ``--opponent self`` loads the
same checkpoint into a frozen opponent (the natural "vs self" replay).

The script loads the rl_games config to get the network/model architecture
right; the checkpoint must come from a run trained with that config.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml


def _build_player(config_path: str, checkpoint: str):
    """Boot a torch_runner Player; we'll call its model directly for inference."""
    from rl_games.torch_runner import Runner

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Player only needs minimal infra — small actor count, no Ray.
    cfg['params']['config']['num_actors'] = 1
    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    player.restore(checkpoint)
    # Our env emits (num_agents, obs_dim) per step — already batched.
    # Without this flag, get_action would unsqueeze to (1, num_agents, obs_dim)
    # and the linear layer would see a 2*obs_dim input (mat-shape mismatch).
    player.has_batch_dimension = True
    return player


def _act(player, obs_np):
    """Run the trained policy deterministically on one batch of obs.

    rl_games' player.get_action expects a torch tensor on the right device.
    """
    import torch
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=player.device)
    with torch.no_grad():
        action = player.get_action(obs, is_deterministic=True)
    return action.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', help='path to trained .pth')
    ap.add_argument('--config', default='rl_games/configs/dm_control/ppo_soccer_ant.yaml',
                    help='rl_games YAML used to train the checkpoint')
    ap.add_argument('--out', default='dm_soccer_ant.mp4', help='output mp4 path')
    ap.add_argument('--episodes', type=int, default=3, help='number of matches to record')
    ap.add_argument('--opponent', choices=['random', 'noop', 'self'], default='random')
    ap.add_argument('--opponent-checkpoint', default=None,
                    help='When set, use this checkpoint for the opponent instead '
                         "of args.checkpoint. Implies --opponent self. Lets you "
                         "render 'current policy vs older snapshot' demos.")
    ap.add_argument('--time-limit', type=float, default=30.0)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--no-field-box', action='store_true',
                    help='cinematic mode — disable invisible walls')
    # Multi-turn (ball respawns) is the default — matches every trained config
    # in this branch and produces longer, more interesting demos.
    ap.add_argument('--terminate-on-goal', action='store_true',
                    help='end episode at first goal (rarely wanted — only for '
                         'configs trained with terminate_on_goal=True).')
    ap.add_argument('--field-size', nargs=2, type=float, default=None,
                    metavar=('W', 'H'),
                    help='custom pitch (default: dm_soccer big field)')
    ap.add_argument('--goal-size', nargs=3, type=float, default=None,
                    metavar=('D', 'W', 'H'),
                    help='custom goal opening (depth, width, height) — bigger '
                         'goals make scoring easier for shallow-trained policies')
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f'checkpoint not found: {args.checkpoint}')

    # Build the env directly (no Ray) so we can render frames.
    from rl_games.envs.dm_soccer import DMSoccerAntEnv

    # Read the training config — needed both for the network arch and for the
    # env_config block (walker_type, field/goal size, team_size, shaping) so
    # the render env matches what the policy was actually trained on.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_env_cfg = dict(cfg['params']['config'].get('env_config', {}))
    train_env_cfg.pop('ray_config', None)  # training-only, not an env arg

    # --opponent-checkpoint implies self-play with a separate weight source.
    if args.opponent_checkpoint:
        args.opponent = 'self'

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

    # Start from the trained env_config, then apply render-time overrides.
    env_kwargs = dict(train_env_cfg)
    env_kwargs.update(opponent_cfg)
    env_kwargs.update(
        time_limit=args.time_limit,
        enable_field_box=not args.no_field_box,
        terminate_on_goal=args.terminate_on_goal,
        shaped_reward=True,
        seed=0,
    )
    if args.field_size is not None:
        env_kwargs['field_size'] = tuple(args.field_size)
    if args.goal_size is not None:
        env_kwargs['goal_size'] = tuple(args.goal_size)
    env = DMSoccerAntEnv(**env_kwargs)

    player = _build_player(args.config, args.checkpoint)

    # If --opponent self, push checkpoint weights into the env's opponent.
    # When --opponent-checkpoint is given, load THOSE weights for the opponent
    # so it's a *different* snapshot than the home agent (older self, etc).
    if args.opponent == 'self':
        if args.opponent_checkpoint:
            import torch as _t
            opp_ckpt = _t.load(args.opponent_checkpoint, map_location='cpu',
                               weights_only=False)
            opp_sd = opp_ckpt.get('model', opp_ckpt)
            # Strip torch.compile prefix if present (training-time artefact).
            opp_sd = {k.replace('_orig_mod.', ''): v for k, v in opp_sd.items()}
            env.update_weights({'model': opp_sd})
            print(f'opponent loaded from {args.opponent_checkpoint}')
        else:
            env.update_weights({'model': player.model.state_dict()})

    # Lazy import: imageio handles mp4 muxing via ffmpeg.
    try:
        import imageio
    except ImportError:
        sys.exit('imageio is required for video output: pip install imageio[ffmpeg]')

    writer = imageio.get_writer(args.out, fps=30, codec='libx264', quality=8)
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_len = 0
            while True:
                action = _act(player, obs)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += float(np.asarray(r).sum())
                ep_len += 1
                frame = env.render(height=args.height, width=args.width)
                writer.append_data(frame)
                if bool(term.any()) or bool(trunc.any()):
                    print(f'episode {ep+1}: len={ep_len} sum_reward={ep_reward:.2f} '
                          f"home={info.get('home_goals', 0)} away={info.get('away_goals', 0)}")
                    break
    finally:
        writer.close()
        env.close()
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
