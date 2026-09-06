#!/usr/bin/env python
"""Play Craftax-Classic with a pixel (or symbolic) checkpoint and record env 0 to mp4.

    python scripts/craftax_play.py -f rl_games/configs/craftax/ppo_craftax_classic_pixels_distill.yaml \
        -c runs/<run>/nn/craftax_classic_pixels_distill.pth --video craftax_distill.mp4
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('-c', '--checkpoint', required=True)
    ap.add_argument('--envs', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--video', default=None)
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()

    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.envs.craftax_vecenv import CraftaxVecEnv, ACHIEVEMENTS, crafter_score
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
    import jax

    params = yaml.safe_load(open(args.file))['params']
    cfg = params['config']
    env_cfg = dict(cfg['env_config']); env_cfg.update(seed=args.seed)
    env = CraftaxVecEnv('craftax', args.envs, **env_cfg)
    info = env.get_env_info()
    obs_space = info['observation_space']
    net = ModelBuilder().load(params)
    model = net.build({'actions_num': info['action_space'].n, 'input_shape': obs_space.shape, 'num_seqs': 1,
                       'value_size': 1, 'normalize_value': cfg.get('normalize_value', False),
                       'normalize_input': cfg.get('normalize_input', False)}).to(env.device).eval()
    sd = torch.load(args.checkpoint, map_location=env.device, weights_only=False)['model']
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    render_big = jax.jit(make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_HUMAN))

    def policy_obs(o):
        o = o['obs'] if isinstance(o, dict) else o
        return o.float() / 255.0 if o.dtype == torch.uint8 else o

    obs = env.reset()
    frames, finished, results = [], np.zeros(args.envs, dtype=bool), []
    rewards = np.zeros(args.envs); lengths = np.zeros(args.envs, dtype=int)
    t0 = time.time()
    for step in range(args.max_steps):
        with torch.no_grad():
            out = model({'obs': policy_obs(obs), 'is_train': False})
            act = out['logits'].argmax(-1) if args.deterministic else out['actions']
        if args.video and not finished[0]:
            state0 = jax.tree_util.tree_map(lambda x: x[0], env._state)
            frames.append(np.asarray(render_big(state0)).astype(np.uint8))
        obs, r, d, inf = env.step(act)
        rewards += r.cpu().numpy() * (~finished); lengths += (~finished)
        dm = d.cpu().numpy()
        for i in np.nonzero(dm & ~finished)[0]:
            results.append(inf['achievements'][i].cpu().numpy()); finished[i] = True
        if finished.all():
            break
    print(f'{finished.sum()} episodes in {time.time() - t0:.1f}s; reward mean {rewards[finished].mean():.2f}, '
          f'len mean {lengths[finished].mean():.0f}')
    if results:
        rates = np.mean(np.stack(results), axis=0)
        print(f'crafter score {crafter_score(rates):.1f}; achievements:',
              {a: round(float(v), 2) for a, v in zip(ACHIEVEMENTS, rates) if v > 0})
    print(f'env 0: reward {rewards[0]:.1f}, length {lengths[0]}')
    if args.video and frames:
        import imageio
        imageio.mimwrite(args.video, frames, fps=args.fps, quality=7)
        print('video', args.video, len(frames), 'frames', frames[0].shape)


if __name__ == '__main__':
    main()
