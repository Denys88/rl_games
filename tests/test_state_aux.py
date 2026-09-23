"""actor_critic_state_aux: pixel actor-critic with an auxiliary head that
regresses the privileged state from the actor trunk (no teacher)."""

import torch


def _cfg(coef=1.0):
    return {'model': {'name': 'continuous_a2c_logstd'},
            'network': {'name': 'actor_critic_state_aux', 'separate': False,
                        'state_aux': {'coef': coef, 'units': [32]},
                        'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                                                 'mu_init': {'name': 'default'},
                                                 'sigma_init': {'name': 'const_initializer', 'val': 0},
                                                 'fixed_sigma': True}},
                        'cnn': {'permute_input': True, 'type': 'conv2d', 'activation': 'elu',
                                'initializer': {'name': 'default'}, 'regularizer': {'name': 'None'},
                                'convs': [{'filters': 8, 'kernel_size': 5, 'strides': 2, 'padding': 0},
                                          {'filters': 8, 'kernel_size': 3, 'strides': 2, 'padding': 0}]},
                        'mlp': {'units': [32], 'activation': 'elu', 'initializer': {'name': 'default'}}}}


def _build(coef=1.0, state_dim=7):
    from rl_games.algos_torch.model_builder import ModelBuilder
    return ModelBuilder().load(_cfg(coef)).build(
        {'actions_num': 3, 'input_shape': (32, 32, 3), 'state_shape': (state_dim,), 'num_seqs': 1,
         'value_size': 1, 'normalize_value': False, 'normalize_input': False})


def test_state_aux_loss_present_in_train_and_grads_reach_cnn():
    model = _build()
    obs = torch.rand(6, 32, 32, 3)
    states = torch.randn(6, 7)
    model.train()
    out = model({'obs': obs, 'states': states, 'is_train': True, 'prev_actions': torch.zeros(6, 3)})
    aux = model.get_aux_loss()
    assert aux is not None and 'state_aux' in aux and torch.isfinite(aux['state_aux'])
    aux['state_aux'].backward()
    conv_w = next(p for n, p in model.named_parameters() if 'actor_cnn' in n and p.dim() == 4)
    assert conv_w.grad is not None and conv_w.grad.abs().sum() > 0
    assert model.get_aux_loss() is None                      # consumed; not reused for the next step
    assert 'mus' in out and out['mus'].shape == (6, 3)


def test_state_aux_absent_without_states_and_scaled_by_coef():
    model = _build(coef=1.0)
    model.train()
    model({'obs': torch.rand(4, 32, 32, 3), 'is_train': True, 'prev_actions': torch.zeros(4, 3)})
    assert model.get_aux_loss() is None
    torch.manual_seed(0)
    m1 = _build(coef=1.0)
    torch.manual_seed(0)
    m2 = _build(coef=0.5)
    obs = torch.rand(4, 32, 32, 3)
    states = torch.randn(4, 7)
    for m in (m1, m2):
        m.train()
        m({'obs': obs, 'states': states, 'is_train': True, 'prev_actions': torch.zeros(4, 3)})
    assert torch.allclose(m1.get_aux_loss()['state_aux'], 2 * m2.get_aux_loss()['state_aux'], atol=1e-6)


def test_state_aux_indices_subset():
    from rl_games.algos_torch.model_builder import ModelBuilder
    cfg = _cfg()
    cfg['network']['state_aux']['indices'] = [0, 2, 4]
    model = ModelBuilder().load(cfg).build(
        {'actions_num': 3, 'input_shape': (32, 32, 3), 'state_shape': (7,), 'num_seqs': 1,
         'value_size': 1, 'normalize_value': False, 'normalize_input': False})
    assert model.a2c_network.aux_head[-1].out_features == 3
