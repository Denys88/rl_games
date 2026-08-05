"""Equivalence test for actor_critic_transformer with TrXL-style memory.

Simulates a long rollout (sliding window + zero-on-done + state snapshots at
chunk boundaries, exactly as play_steps_rnn does), then replays each chunk
through the training path with its stored prefix state. Outputs must match:
memory mode gives every training token the same effective context (sliding
span of max_steps) the rollout provides.
"""
import torch
from rl_games.algos_torch import transformer_network

torch.manual_seed(0)

B, OBS, ACT = 3, 11, 4
W = 8

def run(seq_length, use_dones):
    S = seq_length
    N = 2 * W + S  # several chunks worth of steps
    params = {
        'separate': False,
        'space': {'continuous': {
            'mu_activation': 'None', 'sigma_activation': 'None',
            'mu_init': {'name': 'default'}, 'sigma_init': {'name': 'const_initializer', 'val': 0.0},
            'fixed_sigma': True}},
        'mlp': {'units': [32, 16], 'activation': 'elu', 'initializer': {'name': 'default'}},
        'transformer': {'max_steps': W, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2,
                        'ffn_hidden': 48, 'memory': True},
    }
    net = transformer_network.TransformerBuilder.Network(
        params, actions_num=ACT, input_shape=(OBS,), num_seqs=B, value_size=1)
    net.eval()

    obs = torch.randn(B, N, OBS)
    dones = (torch.rand(B, N) < 0.2).float() if use_dones else torch.zeros(B, N)
    dones[:, 0] = 0

    # rollout: mirror play_steps_rnn ordering (zero-on-done from the previous
    # step's env result happens before the state snapshot at a chunk boundary)
    states = list(net.get_default_rnn_state())
    snapshots, roll_mu = [], []
    with torch.no_grad():
        for t in range(N):
            di = dones[:, t].nonzero(as_tuple=True)[0]
            for s in states:
                s[:, di, :] = 0
            if t % S == 0:
                snapshots.append(tuple(s.clone() for s in states))
            mu, sigma, value, states = net({
                'obs': obs[:, t], 'rnn_states': tuple(states), 'is_train': False})
            states = list(states)
            roll_mu.append(mu)
    roll_mu = torch.stack(roll_mu, dim=1)

    # training: each chunk with its stored prefix state
    max_err = 0.0
    with torch.no_grad():
        for c in range(N // S):
            t0 = c * S
            mu, sigma, value, _ = net({
                'obs': obs[:, t0:t0 + S].reshape(B * S, OBS),
                'dones': dones[:, t0:t0 + S].reshape(B * S),
                'seq_length': S, 'is_train': True,
                'rnn_states': snapshots[c]})
            err = (roll_mu[:, t0:t0 + S] - mu.view(B, S, ACT)).abs().max().item()
            max_err = max(max_err, err)
    return max_err

for S in (W, W // 2):
    for use_dones in (False, True):
        err = run(S, use_dones)
        tag = f'S={S} W={W} dones={use_dones}'
        status = 'OK' if err < 1e-4 else 'FAIL'
        print(f'{tag:<28} max err {err:.2e}  {status}')
        assert status == 'OK', tag
print('memory equivalence checks passed')
