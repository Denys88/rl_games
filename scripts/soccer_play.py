#!/usr/bin/env python
"""Play full envpool soccer matches with a checkpoint, print health stats,
optionally render one match to mp4.

    python scripts/soccer_play.py -f <config.yaml> -c <ckpt.pth> --opponent random --matches 8 --video out.mp4
    python scripts/soccer_play.py -f ... -c ... --opponent self
    python scripts/soccer_play.py -f ... -c ... --opponent checkpoint --opponent-checkpoint <older.pth>
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch, yaml


def load_model(params, obs_dim, act_dim, ckpt, device):
    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.envs.envpool_soccer import _strip_prefix
    net = ModelBuilder().load(params)
    cfg = params['config']
    model = net.build({'actions_num': act_dim, 'input_shape': (obs_dim,), 'num_seqs': 1, 'value_size': 1,
                       'normalize_value': cfg.get('normalize_value', False),
                       'normalize_input': cfg.get('normalize_input', False)}).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)['model']
    model.load_state_dict(_strip_prefix(sd))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('--opponent', default='random', choices=['random', 'self', 'checkpoint'])
    ap.add_argument('--opponent-checkpoint', default=None, help='away team weights (opponent=checkpoint)')
    ap.add_argument('--matches', type=int, default=8)
    ap.add_argument('--video', default=None)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frame-skip', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1234, help='env seed (match 0 is the one filmed)')
    ap.add_argument('--deterministic', action='store_true', default=True)
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions (both teams) instead of using the mean')
    ap.add_argument('--opponent-sigma-scale', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                    help='per-match multiplier range on the away policy sigma (implies --stochastic)')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.file))['params']
    env_cfg = dict(cfg['config']['env_config'])
    env_cfg.update(opponent='random' if args.opponent == 'random' else 'pool', num_threads=8, seed=args.seed)
    if args.opponent_sigma_scale is not None:
        env_cfg['opponent_sigma_scale'] = list(args.opponent_sigma_scale)
        env_cfg['opponent_deterministic'] = False
    if args.stochastic:
        args.deterministic = False
        env_cfg['opponent_deterministic'] = False
    if args.video:
        env_cfg.update(render_mode='rgb_array', render_width=640, render_height=480, render_camera_id=args.camera)
    from rl_games.envs.envpool_soccer import EnvpoolSoccerVecEnv, MAIN_ID
    env = EnvpoolSoccerVecEnv('envpool_soccer', args.matches, **env_cfg)
    N, T = env.num_envs, env.team_size
    model = load_model(cfg, env.obs_dim, env.act_dim, args.checkpoint, env.device)
    if args.opponent == 'self':
        env.set_opponent_template(model)
        env.set_pool_assignment(np.full(N, MAIN_ID), {MAIN_ID: model.state_dict()})
    elif args.opponent == 'checkpoint':
        opp = load_model(cfg, env.obs_dim, env.act_dim, args.opponent_checkpoint, env.device)
        env.set_opponent_template(model)
        env.set_pool_assignment(np.full(N, 1), {1: opp.state_dict()})

    obs = env.reset()
    frames, finished = [], 0
    speed, near_ball, vbg, closest_v = [], [], [], []
    goal_events = []
    t0 = time.time(); step = 0
    while finished < N:
        with torch.no_grad():
            out = model({'obs': obs, 'is_train': False})
            act = out['mus'] if args.deterministic else out['actions']
        obs, rew, done, info = env.step(act)
        raw, rinfo = env.last_raw_obs, env.last_raw_info
        vel = env._ordered(np.asarray(raw['sensors_velocimeter'])).reshape(N, 2 * T, 3)[:, :T, :2]
        speed.append(np.linalg.norm(vel, axis=-1).mean())
        bp = env._ordered(np.asarray(raw['ball_ego_position'])).reshape(N, 2 * T, 3)[:, :T, :2]
        d = np.linalg.norm(bp, axis=-1)
        near_ball.append((d.min(axis=1) < 1.0).mean())
        vbg.append(env._stat(raw, rinfo, 'stats_vel_ball_to_goal')[:, 0].mean())
        closest_v.append(env._stat(raw, rinfo, 'stats_closest_vel_to_ball')[:, :T].sum(1).mean())
        gs = np.sign(rew.view(N, T)[:, 0].cpu().numpy())
        for i in np.nonzero(np.abs(rew.view(N, T)[:, 0].cpu().numpy()) >= env.shaping['goal'] * 0.5)[0]:
            goal_events.append((int(i), step, 'HOME' if gs[i] > 0 else 'AWAY'))
        if args.video and step % args.frame_skip == 0 and finished == 0:
            frames.append(env.env.render(env_ids=[0])[0].copy())
        if done.any():
            dm = done.view(N, T)[:, 0].cpu().numpy()
            for i in np.nonzero(dm)[0]:
                print(f'match {i}: home {int(info["home_goals"][i])} - away {int(info["away_goals"][i])}  len {int(info["match_len"][i])}')
            finished += int(dm.sum())
        step += 1
    dt = time.time() - t0
    print(f'\n{N} matches vs {args.opponent}, {step} steps in {dt:.1f}s')
    print(f'mean home speed (m/s):            {np.mean(speed):.2f}   (early {np.mean(speed[:100]):.2f}, late {np.mean(speed[-100:]):.2f})')
    print(f'frac steps a home robot <1 m ball:{np.mean(near_ball):.2f}')
    print(f'mean ball vel to opp goal (m/s):  {np.mean(vbg):+.3f}')
    print(f'mean closest-teammate vel->ball:  {np.mean(closest_v):+.3f}')
    print('goal timeline (match, step, scorer):', goal_events[:40])
    if args.video and frames:
        import imageio
        imageio.mimwrite(args.video, frames, fps=int(40 / args.frame_skip), quality=7)
        print('video', args.video, len(frames), 'frames', frames[0].shape)
    env.close()


if __name__ == '__main__':
    main()
