#!/usr/bin/env python
"""Hybrid scripted-scorer evaluator for dm_soccer ant.

Wraps a trained chase-policy in an obs-rewriter that *lies* to it about
where the ball is. The fake ball position is set to a point on the
own-goal side of the real ball, on the line through real-ball→opp-goal —
i.e. the position from which a forward push moves the ball toward goal.

When the ant gets to the fake-ball point, the real ball is directly in
front of it; the ant's chase-and-bump reflex then drives the real ball
toward the opponent goal. No retraining needed.

Usage:
    python eval_dm_soccer_ant_scripted.py CHECKPOINT [--episodes N] [--offset M]
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
    player.has_batch_dimension = True
    return player


def _rewrite_obs_for_scoring(env, raw_obs_per_agent, offset: float):
    """Replace each agent's ball_ego_position with a 'behind-ball' target.

    raw_obs_per_agent: list of OrderedDicts (per home player) — the
        original dm_env observations.
    offset: distance behind ball (along ball→opp_goal axis) where the
        fake target sits. ~0.5–1.0 m works for an ant.
    """
    rewritten = []
    for o in raw_obs_per_agent:
        o = dict(o)  # copy keys
        ball = np.asarray(o['ball_ego_position']).reshape(-1)
        goal = np.asarray(o['opponent_goal_mid']).reshape(-1)
        # Direction from ball toward opp goal in ego frame.
        d = goal - ball
        n = float(np.linalg.norm(d))
        if n > 1e-3:
            unit = d / n
            # Fake ball = real ball minus offset along the goal direction
            # (i.e. on the own-goal side of the ball).
            fake_ball = ball - offset * unit
        else:
            fake_ball = ball
        # Preserve original shape (1, 3).
        o['ball_ego_position'] = fake_ball.reshape(np.asarray(o['ball_ego_position']).shape)
        rewritten.append(o)
    return rewritten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default='rl_games/configs/dm_control/ppo_soccer_ant_v3.yaml')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--time-limit', type=float, default=30.0)
    ap.add_argument('--field-size', nargs=2, type=float, default=[15.0, 10.0])
    ap.add_argument('--goal-size', nargs=3, type=float, default=[1.0, 6.0, 0.6])
    ap.add_argument('--offset', type=float, default=0.6,
                    help='How far (m) behind the real ball to place the fake target.')
    ap.add_argument('--no-rewrite', action='store_true',
                    help='Disable the obs rewrite — straight chase-policy baseline.')
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f'checkpoint not found: {args.checkpoint}')

    import torch
    # We need direct access to the dm_env to read each player's raw obs
    # (the rl_games wrapper drops `stats_*` keys and stacks players).
    from dm_control.locomotion import soccer
    rs = np.random.RandomState(int(time.time()) & 0xFFFF)
    if args.field_size:
        from dm_control import composer
        from dm_control.locomotion.soccer import (
            MultiturnTask, RandomizedPitch, SoccerBall, _make_players,
        )
        w, h = args.field_size
        arena = RandomizedPitch(
            min_size=(float(w), float(h)),
            max_size=(float(w), float(h)),
            keep_aspect_ratio=False,
            field_box=True,
            goal_size=tuple(args.goal_size) if args.goal_size else None,
        )
        task = MultiturnTask(
            players=_make_players(2, soccer.WalkerType.ANT),
            arena=arena, ball=SoccerBall(), disable_walker_contacts=False,
        )
        dm_env = composer.Environment(task=task, time_limit=args.time_limit, random_state=rs)
    else:
        dm_env = soccer.load(team_size=2, time_limit=args.time_limit, random_state=rs,
                              walker_type=soccer.WalkerType.ANT, terminate_on_goal=False)

    player = _build_player(args.config, args.checkpoint)

    # Same flat-obs builder as the rl_games wrapper.
    from rl_games.envs.dm_soccer import _flatten_obs, _STAT_KEYS

    def home_obs(per_player_obs, rewrite: bool):
        home = per_player_obs[:2]  # team_size=2
        if rewrite:
            home = _rewrite_obs_for_scoring(None, home, args.offset)
        return np.stack([
            _flatten_obs(o, skip_keys=_STAT_KEYS) for o in home
        ]).astype(np.float32)

    home_goals_total = away_goals_total = 0
    wins = draws = losses = 0
    for ep in range(args.episodes):
        ts = dm_env.reset()
        last_home, last_away = 0.0, 0.0
        ep_h = ep_a = 0
        steps = 0
        while True:
            obs_arr = home_obs(ts.observation, rewrite=not args.no_rewrite)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=player.device)
            with torch.no_grad():
                home_actions = player.get_action(obs_t, is_deterministic=True).detach().cpu().numpy()
            # Random away team (matches eval_dm_soccer_ant.py default).
            away_actions = [rs.uniform(-1.0, 1.0, size=(8,)) for _ in range(2)]
            all_actions = [np.clip(a, -1.0, 1.0) for a in home_actions] + away_actions
            ts = dm_env.step(all_actions)
            steps += 1
            # Goal accounting from stats_*.
            obs0 = ts.observation[0]
            hs = float(np.asarray(obs0['stats_home_score']).ravel()[0])
            as_ = float(np.asarray(obs0['stats_away_score']).ravel()[0])
            if hs > last_home:
                ep_h += int(round(hs - last_home))
                last_home = hs
            if as_ > last_away:
                ep_a += int(round(as_ - last_away))
                last_away = as_
            if ts.last():
                break
        home_goals_total += ep_h
        away_goals_total += ep_a
        if ep_h > ep_a: wins += 1
        elif ep_a > ep_h: losses += 1
        else: draws += 1
        print(f'  ep {ep+1:2d}: len={steps:4d}  goals={ep_h}-{ep_a}')

    n = args.episodes
    print('=' * 60)
    print(f'EPISODES: {n}   REWRITE: {not args.no_rewrite}   OFFSET: {args.offset:.2f} m')
    print(f'  W/D/L:                {wins}/{draws}/{losses}   win_rate={100*wins/n:.1f}%')
    print(f'  goals home/away:      {home_goals_total}/{away_goals_total}')
    print(f'  mean goals/ep:        home={home_goals_total/n:.2f}  away={away_goals_total/n:.2f}  diff={(home_goals_total-away_goals_total)/n:+.2f}')


if __name__ == '__main__':
    main()
