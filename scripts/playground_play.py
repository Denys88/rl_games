#!/usr/bin/env python
"""Play a mujoco_playground checkpoint and record world 0 to mp4 (big render
+ the policy's own pixel view as an inset when the obs is pixels).

    python scripts/playground_play.py -f rl_games/configs/playground/ppo_panda_pick_pixels_distill.yaml \
        -c runs/<run>/nn/panda_pick_pixels_distill.pth --video panda_distill.mp4 --episodes 3
"""
import argparse
import os
import sys

WSL_LIB = '/usr/lib/wsl/lib'
if os.path.isdir(WSL_LIB) and WSL_LIB not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
    os.environ['LD_LIBRARY_PATH'] = WSL_LIB + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('--envs', type=int, default=64)
    ap.add_argument('--episodes', type=int, default=3, help='consecutive episodes of world 0 to film')
    ap.add_argument('--video', default=None)
    ap.add_argument('--camera', default=None)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--stochastic', action='store_true')
    args = ap.parse_args()

    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.envs.playground_vecenv import PlaygroundVecEnv
    params = yaml.safe_load(open(args.file))['params']
    cfg = params['config']
    env_cfg = dict(cfg['env_config'])
    env_cfg.update(seed=args.seed)
    env = PlaygroundVecEnv('playground', args.envs, **env_cfg)
    info = env.get_env_info()
    net = ModelBuilder().load(params)
    model = net.build({'actions_num': info['action_space'].shape[0], 'input_shape': info['observation_space'].shape,
                       'num_seqs': 1, 'value_size': 1, 'normalize_value': cfg.get('normalize_value', False),
                       'normalize_input': cfg.get('normalize_input', False)}).to(env.device).eval()
    sd = torch.load(args.checkpoint, map_location=env.device, weights_only=False)['model']
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    mj = env.raw_env.mj_model
    cams = [mj.camera(i).name for i in range(mj.ncam)]
    camera = args.camera if args.camera is not None else (cams[0] if cams else None)
    print('cameras:', cams, '-> using', camera)

    def policy_obs(o):
        o = o['obs'] if isinstance(o, dict) else o
        return o.float() / 255.0 if o.dtype == torch.uint8 else o

    obs = env.reset()
    traj, insets = [], []
    successes, lengths, finished = [], [], 0
    ep_len = np.zeros(args.envs, dtype=int)
    while True:
        if args.video:
            traj.append(env.env_state(0))
            o0 = obs['obs'][0] if isinstance(obs, dict) else obs[0]
            if o0.dtype == torch.uint8:
                insets.append(o0.cpu().numpy())
        with torch.no_grad():
            out = model({'obs': policy_obs(obs), 'is_train': False})
            act = out['actions'] if args.stochastic else out['mus']
        obs, r, d, inf = env.step(act)
        ep_len += 1
        if d.any():
            m = inf['metrics']
            for i in torch.nonzero(d).flatten().tolist():
                successes.append(float(m['reward/success'][i]) if 'reward/success' in m else float('nan'))
                lengths.append(int(ep_len[i]))
                ep_len[i] = 0
                if i == 0:
                    finished += 1
        if finished >= args.episodes or len(traj) > 400 * args.episodes:
            break
    print(f'{len(successes)} episodes: success rate {np.nanmean(successes):.2f}, mean length {np.mean(lengths):.0f}')
    if args.video and traj:
        import imageio
        frames = env.raw_env.render(traj, height=480, width=640, camera=camera)
        out = []
        for k, fr in enumerate(frames):
            fr = np.asarray(fr).copy()
            if insets:
                ins = np.kron(insets[k], np.ones((3, 3, 1), dtype=np.uint8))       # 64 -> 192 px
                fr[8:8 + ins.shape[0], 8:8 + ins.shape[1]] = ins
            out.append(fr)
        imageio.mimwrite(args.video, out, fps=args.fps, quality=7)
        print('video', args.video, len(out), 'frames', out[0].shape)


if __name__ == '__main__':
    main()
