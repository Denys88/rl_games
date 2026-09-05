#!/usr/bin/env python
"""Round-robin evaluation of soccer checkpoints; keeps the best one.

    python scripts/soccer_eval.py -f <config.yaml> --checkpoints a.pth b.pth ... \
        --matches 32 --tag v8 --best-dir best_agents

Every checkpoint plays every other one (home vs away, so both orderings
appear) plus a random-action team, `--matches` matches per pairing, in the env
of the given config. Score = mean over opponents of win + 0.5 draw. The best
checkpoint is copied to <best-dir>/<tag>_ep<N>.pth and a table is appended to
<best-dir>/RESULTS.md.
"""
import argparse, os, re, shutil, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, yaml


def epoch_of(path):
    m = re.search(r'ep_?(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--matches', type=int, default=32, help='matches per pairing')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--best-dir', default='best_agents')
    ap.add_argument('--opponent-sigma-scale', type=float, nargs=2, default=(1.0, 1.0))
    ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--max-steps', type=int, default=0, help='0 = 1.1 * time_limit steps')
    args = ap.parse_args()

    from soccer_play import load_model
    from rl_games.envs.envpool_soccer import EnvpoolSoccerVecEnv, RANDOM_ID
    cfg = yaml.safe_load(open(args.file))['params']
    env_cfg = dict(cfg['config']['env_config'])
    env_cfg.update(opponent='pool', num_threads=8, seed=args.seed, opponent_deterministic=False,
                   opponent_sigma_scale=list(args.opponent_sigma_scale))
    ckpts = sorted(args.checkpoints, key=epoch_of)
    K = len(ckpts)
    opp_ids = list(range(1, K + 1)) + [RANDOM_ID]       # pool ids 1..K = ckpt index + 1
    N = len(opp_ids) * args.matches
    env = EnvpoolSoccerVecEnv('envpool_soccer', N, **env_cfg)
    models = [load_model(cfg, env.obs_dim, env.act_dim, c, env.device) for c in ckpts]
    env.set_opponent_template(models[0])
    pool = {i + 1: m.state_dict() for i, m in enumerate(models)}
    max_steps = args.max_steps or int(1.1 * float(env_cfg.get('time_limit', 30.0)) * 40)
    assignment = np.repeat(np.array(opp_ids, dtype=np.int64), args.matches)

    table = np.zeros((K, len(opp_ids)))       # win + 0.5 draw
    gdiff = np.zeros((K, len(opp_ids)))
    t0 = time.time()
    for i, model in enumerate(models):
        env.set_pool_assignment(assignment, pool)
        obs = env.reset()
        result = np.full(N, np.nan)
        for step in range(max_steps):
            with torch.no_grad():
                act = model({'obs': obs, 'is_train': False})['mus']
            obs, rew, done, info = env.step(act)
            if done.any():
                dm = done.view(N, env.team_size)[:, 0].cpu().numpy()
                fresh = dm & np.isnan(result)
                result[fresh] = info['goal_diff'].cpu().numpy()[fresh]
            if not np.isnan(result).any():
                break
        result = np.nan_to_num(result, nan=0.0)          # unfinished = draw
        for j, oid in enumerate(opp_ids):
            r = result[assignment == oid]
            table[i, j] = ((r > 0) + 0.5 * (r == 0)).mean()
            gdiff[i, j] = r.mean()
        print(f'[{i + 1}/{K}] ep {epoch_of(ckpts[i]):>6d}: ' +
              ' '.join(f'{table[i, j]:.2f}' for j in range(len(opp_ids))) +
              f'   ({time.time() - t0:.0f}s)', flush=True)
    env.close()

    # score vs other checkpoints only (diagonal = self-play, excluded); vs random reported separately
    mask = np.ones_like(table, dtype=bool)
    for i in range(K):
        mask[i, i] = False
    mask[:, -1] = False
    score = np.array([table[i, mask[i]].mean() if K > 1 else 0.0 for i in range(K)])
    best = int(np.argmax(score + 1e-3 * table[:, -1]))
    tag = args.tag or os.path.basename(args.file).replace('.yaml', '')
    os.makedirs(args.best_dir, exist_ok=True)
    dst = os.path.join(args.best_dir, f'{tag}_ep{epoch_of(ckpts[best])}.pth')
    shutil.copy(ckpts[best], dst)
    header = '| ckpt | ' + ' | '.join(f'vs ep{epoch_of(c)}' for c in ckpts) + ' | vs random | score |\n'
    header += '|' + '---|' * (K + 3) + '\n'
    rows = ''
    for i in range(K):
        rows += f'| ep{epoch_of(ckpts[i])} | ' + ' | '.join(
            ('-' if i == j else f'{table[i, j]:.2f}') for j in range(K)) + \
            f' | {table[i, -1]:.2f} | **{score[i]:.3f}**' + (' best' if i == best else '') + ' |\n'
    md = (f'\n## {tag} — {time.strftime("%Y-%m-%d %H:%M")}\n\n'
          f'config `{args.file}`, {args.matches} matches per pairing, home checkpoint in rows '
          f'(win + 0.5 draw), opponents stochastic x{args.opponent_sigma_scale[0]}-{args.opponent_sigma_scale[1]}.\n\n'
          + header + rows + f'\nbest: `{dst}`\n')
    with open(os.path.join(args.best_dir, 'RESULTS.md'), 'a') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
