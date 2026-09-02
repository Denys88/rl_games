#!/usr/bin/env python
"""azbig gen350 (latest from-scratch AZ) vs B3500 (published champion).

  1) raw vs raw   : azbig sampled@t0.35 vs B3500 flax@t1.0
  2) search@128   : both sides Gumbel MCTS on their own net;
                    B3500 value normalizer folded into value_fc2 (as shipped)
"""
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

AZBIG = 'runs/go9_azbig_27-11-23-04/nn/last_go9_azbig_gen_350.pth'
B3500 = 'runs/go9_big_b_26-17-52-12/nn/last_go9_big_b_ep_3500_rew_0.3246211.pth'
BIG_ARCH = dict(blocks=18, channels=384, gpool_every=3, value_units=256,
                block_type='nbt', bottleneck_channels=192)
SIMS = 128
GAMES = 256


def clean_sd(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    out = {}
    for k, v in sd.items():
        for pre in ('_orig_mod.', 'a2c_network.'):
            k = k.replace(pre, '')
        out[k] = v
    return out


def folded_sd(ckpt_path):
    sd = clean_sd(ckpt_path)
    vm, vv = sd.get('value_mean_std.running_mean'), sd.get('value_mean_std.running_var')
    if vm is not None and vv is not None:
        std = float(np.sqrt(float(vv.reshape(-1)[0]) + 1e-5))
        mean = float(vm.reshape(-1)[0])
        sd['value_fc2.weight'] = sd['value_fc2.weight'] * std
        sd['value_fc2.bias'] = sd['value_fc2.bias'] * std + mean
        print(f'folded value normalizer (std {std:.3f}, mean {mean:.3f})', flush=True)
    return sd


def torch_net(ckpt_path):
    from rl_games.envs.go_network import GoResNetBuilder
    b = GoResNetBuilder()
    b.load(dict(BIG_ARCH))
    net = b.build('go_resnet', input_shape=(9, 9, 17), actions_num=82).cuda().eval()
    sd = clean_sd(ckpt_path)
    net.load_state_dict({k: v for k, v in sd.items() if k in net.state_dict()},
                        strict=False)
    return net


def play(env, act_fn, games, tag):
    obs = env.reset()
    wins = n = 0
    cwins = [0, 0]
    cn = [0, 0]
    while n < games:
        mask = env.get_action_masks()
        a = act_fn(obs, mask, env)
        obs, _, d, info = env.step(a)
        db = d.bool()
        if db.any():
            w = info['win'][db] > 0
            col = info['game_color'][db]
            for c in (0, 1):
                cm = col == c
                cwins[c] += int((w & cm).sum().item())
                cn[c] += int(cm.sum().item())
            wins += int(w.sum().item())
            n += int(db.sum().item())
    wb = cwins[0] / max(cn[0], 1)
    ww = cwins[1] / max(cn[1], 1)
    print(f'RESULT {tag}: {wins / n:.3f} ({n}g)  '
          f'as black {wb:.3f} ({cn[0]}g)  as white {ww:.3f} ({cn[1]}g)', flush=True)


def main():
    import jax
    from rl_games.envs.pgx_go import PgxGoVecEnv
    from rl_games.envs.go_flax import params_from_torch, make_flax_opponent
    from rl_games.envs.go_search import make_search_policy, make_search_opponent
    from pgx import go

    az_flax = params_from_torch(clean_sd(AZBIG), **BIG_ARCH)
    pub_flax = params_from_torch(folded_sd(B3500), **BIG_ARCH)

    # 1) raw vs raw
    net = torch_net(AZBIG)

    def raw_sampled(obs, mask, env):
        with torch.no_grad():
            logits, _, _ = net({'obs': obs})
        logits = logits / 0.35
        logits[~mask] = -torch.inf
        return torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)

    opp_fn, _ = make_flax_opponent(temperature=1.0, **BIG_ARCH)
    env = PgxGoVecEnv('pgx_go', GAMES, komi=7.0, symmetry=False, seed=1357,
                      opponent=opp_fn, opponent_params=pub_flax)
    play(env, raw_sampled, GAMES, 'azbig350 raw(t0.35) vs B3500 raw(t1)')
    del env, net
    torch.cuda.empty_cache()

    # 2) search@128 vs search@128
    genv = go.Go(size=9, komi=7.0)
    az_search = make_search_policy(genv, num_simulations=SIMS, **BIG_ARCH)
    pub_opp = make_search_opponent(genv, num_simulations=SIMS, **BIG_ARCH)
    env = PgxGoVecEnv('pgx_go', GAMES, komi=7.0, symmetry=False, seed=4242,
                      opponent=pub_opp, opponent_params=pub_flax)
    rng = jax.random.PRNGKey(7)

    def az_search_act(obs, mask, env):
        nonlocal rng
        rng, k = jax.random.split(rng)
        a_env, _ = az_search(az_flax, env.get_jax_state(), k)
        return env.env_actions_to_sym(a_env).long()

    play(env, az_search_act, GAMES, f'azbig350@{SIMS} vs B3500@{SIMS}')
    print('EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
