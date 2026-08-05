"""Rollout-vs-training equivalence for actor_critic_transformer.

Feeds a trajectory one step at a time through the eval path (sliding window +
zero-on-done, exactly as the rnn-state plumbing does), then the same trajectory
through the batched training path (causal + done-segment mask), and checks the
outputs match. Guards the subtle invariants: causal masking inside the rollout
window recompute, RoPE relative-position consistency, and done-boundary resets.
"""
import torch

from rl_games.algos_torch import transformer_network

W = 8    # max_steps == seq_length
B = 3    # parallel envs
OBS = 11
ACT = 4

PARAMS = {
    'separate': False,
    'space': {'continuous': {
        'mu_activation': 'None', 'sigma_activation': 'None',
        'mu_init': {'name': 'default'}, 'sigma_init': {'name': 'const_initializer', 'val': 0.0},
        'fixed_sigma': True}},
    'mlp': {'units': [32, 16], 'activation': 'elu', 'initializer': {'name': 'default'}},
    'transformer': {'max_steps': W, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'ffn_hidden': 48},
}


def _run_scenario(net, obs_traj, dones_seq):
    # rollout path: one step at a time, zeroing state where done
    buf, valid = net.get_default_rnn_state()
    roll_mu, roll_val = [], []
    with torch.no_grad():
        for t in range(W):
            done_idx = dones_seq[:, t].nonzero(as_tuple=True)[0]
            buf[:, done_idx, :] = 0      # what zero_rnn_on_done does before step t
            valid[:, done_idx, :] = 0
            mu, sigma, value, (buf, valid) = net({
                'obs': obs_traj[:, t], 'rnn_states': (buf, valid), 'is_train': False})
            roll_mu.append(mu)
            roll_val.append(value)

        # training path: whole sequence at once
        mu, sigma, value, _ = net({
            'obs': obs_traj.reshape(B * W, OBS),
            'dones': dones_seq.reshape(B * W),
            'seq_length': W, 'is_train': True,
            'rnn_states': net.get_default_rnn_state()})

    err_mu = (torch.stack(roll_mu, dim=1) - mu.view(B, W, ACT)).abs().max().item()
    err_val = (torch.stack(roll_val, dim=1) - value.view(B, W, 1)).abs().max().item()
    return max(err_mu, err_val)


def test_rollout_matches_training():
    torch.manual_seed(0)
    net = transformer_network.TransformerBuilder.Network(
        PARAMS, actions_num=ACT, input_shape=(OBS,), num_seqs=B, value_size=1)
    net.eval()

    for dones_seq in (torch.zeros(B, W), (torch.rand(B, W) < 0.25).float()):
        dones_seq[:, 0] = 0  # step 0 follows a fresh reset (states start zeroed)
        err = _run_scenario(net, torch.randn(B, W, OBS), dones_seq)
        assert err < 1e-4, f'rollout/training mismatch: {err}'


if __name__ == '__main__':
    test_rollout_matches_training()
    print('ok')
