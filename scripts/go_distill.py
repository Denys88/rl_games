#!/usr/bin/env python
"""Distill the big (b18c384nbt) Go net into a small student for the website.

Stages (all sharing the GPU with whatever else runs):
  1. gen    — self-play positions of the teacher against itself
              (torch teacher sampled vs flax teacher at temp 1, PgxGoVecEnv)
  2. label  — teacher outputs per position: policy logits, value,
              ownership map, score-distribution logits
  3. train  — student (b10c128) on KL(policy) + MSE(value) + MSE(ownership)
              + KL(score_dist), random dihedral augmentation
  4. eval   — student sampled vs the AlphaZero baseline
  5. export — fp16 base64 JSON + parity reference for the website

python scripts/go_distill.py --teacher <big ckpt> [--positions 500000]
"""

import argparse
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.15')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import base64
import json
import numpy as np
import torch
import torch.nn.functional as F

TEACHER_ARCH = dict(blocks=18, channels=384, gpool_every=3, value_units=256,
                    block_type='nbt', bottleneck_channels=192)
STUDENT_ARCH = dict(blocks=10, channels=128, gpool_every=2, value_units=128)
SCRATCH = os.environ.get('GO_DISTILL_OUT', 'runs/go9_distill')


def clean_sd(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    out = {}
    for k, v in sd.items():
        for pre in ('_orig_mod.', 'a2c_network.'):
            k = k.replace(pre, '')
        out[k] = v
    return out


def build_net(arch, aux):
    from rl_games.envs.go_network import GoResNetBuilder
    b = GoResNetBuilder()
    b.load(dict(arch, aux_heads=aux))
    return b.build('go_resnet', input_shape=(9, 9, 17), actions_num=82).cuda()


def teacher_heads(net, obs):
    """policy logits, value, ownership, score-dist logits for a batch."""
    x = obs.permute(0, 3, 1, 2).contiguous()
    x = net.stem(x)
    for blk in net.blocks:
        x = blk(x)
    trunk = F.relu(x)
    pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)
    pl = net.policy_conv(trunk).reshape(trunk.shape[0], -1)
    logits = torch.cat([pl, net.policy_pass(pooled)], dim=1)
    value = net.value_fc2(F.relu(net.value_fc1(pooled)))
    own = torch.tanh(net.own_conv(trunk).reshape(trunk.shape[0], -1))
    sdist = net.score_fc2(F.relu(net.score_fc1(pooled)))
    return logits, value.squeeze(-1), own, sdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--positions', type=int, default=500_000)
    ap.add_argument('--boards', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-games', type=int, default=256)
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)

    from rl_games.envs.go_flax import params_from_torch
    from rl_games.envs.go_network import GoResNetBuilder  # noqa
    from rl_games.envs.pgx_go import PgxGoVecEnv, _build_sym_tables
    from rl_games.envs.go_flax import make_flax_opponent

    sd = clean_sd(args.teacher)
    teacher = build_net(TEACHER_ARCH,
                        ['ownership', 'score_dist', 'opp_policy', 'plies_left'])
    teacher.load_state_dict(
        {k: v for k, v in sd.items() if k in teacher.state_dict()}, strict=False)
    teacher.eval()
    # fold value normalizer so labels are calibrated raw values
    if 'value_mean_std.running_mean' in sd:
        std = float(np.sqrt(float(sd['value_mean_std.running_var'].reshape(-1)[0]) + 1e-5))
        mean = float(sd['value_mean_std.running_mean'].reshape(-1)[0])
        with torch.no_grad():
            teacher.value_fc2.weight.mul_(std)
            teacher.value_fc2.bias.mul_(std).add_(mean)
        print(f'[distill] folded value stats (std {std:.3f} mean {mean:.3f})', flush=True)

    # ---- 1. generate positions -------------------------------------------
    t_flax = params_from_torch(sd, **TEACHER_ARCH)
    opp_fn, _ = make_flax_opponent(temperature=1.0, **TEACHER_ARCH)
    env = PgxGoVecEnv('pgx_go', args.boards, komi=7.0, seed=97,
                      opponent=opp_fn, opponent_params=t_flax)
    obs = env.reset()
    steps = args.positions // args.boards
    store = np.zeros((steps * args.boards, 173), dtype=np.uint8)
    k = 0
    for t in range(steps):
        with torch.no_grad():
            logits, _, _, _ = teacher_heads(teacher, obs.float())
        mask = env.get_action_masks()
        logits[~mask] = -torch.inf
        a = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        ob = obs.cpu().numpy().astype(np.uint8).reshape(args.boards, -1)
        store[k:k + args.boards] = np.packbits(ob, axis=1)[:, :173]
        k += args.boards
        obs, _, _, _ = env.step(a)
        if t % 50 == 0:
            print(f'[distill] gen {t}/{steps}', flush=True)
    np.save(f'{SCRATCH}/obs.npy', store)
    print(f'[distill] generated {k} positions', flush=True)

    # ---- 2. teacher labels ------------------------------------------------
    N = k
    lab_pol = np.zeros((N, 82), dtype=np.float16)
    lab_val = np.zeros(N, dtype=np.float32)
    lab_own = np.zeros((N, 81), dtype=np.float16)
    lab_sco = np.zeros((N, 41), dtype=np.float16)
    B = 2048
    for i in range(0, N, B):
        bits = store[i:i + B]
        ob = np.unpackbits(bits, axis=1)[:, :1377].astype(np.float32)
        t_obs = torch.from_numpy(ob).cuda().reshape(-1, 9, 9, 17)
        with torch.no_grad():
            lg, vl, ow, sc = teacher_heads(teacher, t_obs)
        lab_pol[i:i + B] = lg.cpu().numpy().astype(np.float16)
        lab_val[i:i + B] = vl.cpu().numpy()
        lab_own[i:i + B] = ow.cpu().numpy().astype(np.float16)
        lab_sco[i:i + B] = sc.cpu().numpy().astype(np.float16)
        if i % (B * 40) == 0:
            print(f'[distill] label {i}/{N}', flush=True)
    del teacher
    torch.cuda.empty_cache()

    # ---- 3. train student -------------------------------------------------
    student = build_net(STUDENT_ARCH, ['ownership', 'score_dist'])
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sym_tables = torch.from_numpy(_build_sym_tables(9).astype(np.int64)).cuda()
    n_steps = args.epochs * (N // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    step = 0
    for ep in range(args.epochs):
        perm = np.random.permutation(N)
        for i in range(0, N - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            ob = np.unpackbits(store[idx], axis=1)[:, :1377].astype(np.float32)
            obs_t = torch.from_numpy(ob).cuda().reshape(-1, 81, 17)
            tp = torch.from_numpy(lab_pol[idx].astype(np.float32)).cuda()
            tv = torch.from_numpy(lab_val[idx]).cuda()
            tw = torch.from_numpy(lab_own[idx].astype(np.float32)).cuda()
            ts = torch.from_numpy(lab_sco[idx].astype(np.float32)).cuda()
            sym = torch.randint(0, 8, (obs_t.shape[0],), device='cuda')
            idx81 = sym_tables[sym][:, :81]
            obs_t = torch.gather(obs_t, 1, idx81.unsqueeze(-1).expand(-1, -1, 17))
            tp = torch.gather(tp, 1, sym_tables[sym])
            tw = torch.gather(tw, 1, idx81)
            obs_t = obs_t.reshape(-1, 9, 9, 17)

            s_log, s_val, s_own, s_sco = teacher_heads(student, obs_t)
            pol_loss = F.kl_div(F.log_softmax(s_log, -1),
                                F.softmax(tp, -1), reduction='batchmean')
            val_loss = F.mse_loss(s_val, tv)
            own_loss = F.mse_loss(s_own, tw)
            sco_loss = F.kl_div(F.log_softmax(s_sco, -1),
                                F.softmax(ts, -1), reduction='batchmean')
            loss = pol_loss + val_loss + own_loss + 0.25 * sco_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f'[distill] step {step}/{n_steps} pol {pol_loss.item():.4f} '
                      f'val {val_loss.item():.4f} own {own_loss.item():.4f}', flush=True)
    torch.save({'model': student.state_dict()}, f'{SCRATCH}/student.pth')

    # ---- 4. eval vs baseline ---------------------------------------------
    from rl_games.envs.go_anchors import make_baseline_opponent
    student.eval()
    benv = PgxGoVecEnv('pgx_go', args.eval_games, komi=7.0, symmetry=False,
                       seed=1122, opponent=make_baseline_opponent('baselines'))
    obs = benv.reset()
    wins = n = 0
    while n < args.eval_games:
        mask = benv.get_action_masks()
        with torch.no_grad():
            lg, _, _, _ = teacher_heads(student, obs.float())
        lg[~mask] = -torch.inf
        a = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
        obs, _, d, info = benv.step(a)
        db = d.bool()
        if db.any():
            wins += int((info['win'][db] > 0).sum().item())
            n += int(db.sum().item())
    print(f'[distill] STUDENT vs baseline: {wins/n:.3f}', flush=True)

    # ---- 5. export fp16 json + parity ref --------------------------------
    def b64f16(t):
        a = t.detach().cpu().numpy().astype('<f2')
        return {'shape': list(a.shape), 'dtype': 'f16',
                'data': base64.b64encode(a.tobytes()).decode()}

    ssd = student.state_dict()
    out = {'arch': STUDENT_ARCH, 'tensors': {},
           'value_stats': {'mean': 0.0, 'var': 1.0 - 1e-5}}
    for k2, v in ssd.items():
        if k2.startswith(('stem', 'blocks.', 'policy_', 'value_fc',
                          'own_conv', 'score_fc')):
            out['tensors'][k2] = b64f16(v)
    with open(f'{SCRATCH}/go_model.json', 'w') as f:
        json.dump(out, f)

    os.environ['JAX_PLATFORMS'] = 'cpu'
    from pgx import go
    import jax, jax.numpy as jnp
    genv = go.Go(size=9, komi=7.0)
    state = genv.init(jax.random.PRNGKey(0))
    stepf = jax.jit(genv.step)
    moves = [40, 20, 60, 33, 47, 12, 68, 25, 55, 30, 41, 21]
    for m in moves:
        state = stepf(state, jnp.int32(m))
    obs_np = np.asarray(state.observation).astype(np.float32)
    with torch.no_grad():
        # reference must match fp16-rounded weights the page will load
        for p in student.parameters():
            p.copy_(p.half().float())
        lg, vl, ow, sc = teacher_heads(student, torch.from_numpy(obs_np[None]).cuda())
        sdist = torch.softmax(sc, -1)[0]
        score_exp = float((sdist * torch.arange(-20, 21).float().cuda()).sum())
    ref = {'moves': moves, 'obs': obs_np.reshape(-1).astype(int).tolist(),
           'legal': np.asarray(state.legal_action_mask).astype(int).tolist(),
           'logits': [round(float(x), 5) for x in lg[0].cpu()],
           'value': round(float(vl[0]), 5),
           'own': [round(float(v), 5) for v in ow[0].cpu()],
           'score_exp': round(score_exp, 4)}
    with open(f'{SCRATCH}/go_ref.json', 'w') as f:
        json.dump(ref, f)
    print(f'[distill] exported {len(out["tensors"])} tensors, '
          f'json {os.path.getsize(f"{SCRATCH}/go_model.json")/1e6:.1f}MB', flush=True)
    print('[distill] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
