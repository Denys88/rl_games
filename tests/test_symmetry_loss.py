"""config.symmetry_loss: mirror-consistency loss on the continuous policy mean."""
import pytest
import torch

from tests.test_ppo_masking import make_ppo_agent, _rollout_batch


def mirror_maps(obs_dim=None, act_dim=None):
    # reversal is an involution: a valid mirror on any layout
    obs_dim = obs_dim or mirror_maps.obs_dim
    act_dim = act_dim or mirror_maps.act_dim
    return {'obs_perm': list(reversed(range(obs_dim))), 'obs_sign': [1.0] * obs_dim,
            'act_perm': list(reversed(range(act_dim))), 'act_sign': [-1.0] * act_dim}


def _dims():
    agent, _ = make_ppo_agent()
    return agent.obs_shape[0], agent.actions_num


def _agent(**config):
    obs_dim, act_dim = _dims()
    return make_ppo_agent(symmetry_loss={'coef': 0.5, 'maps': mirror_maps(obs_dim, act_dim)}, **config)[0]


def test_disabled_by_default():
    agent, _ = make_ppo_agent()
    assert agent.symmetry_loss_coef == 0.0 and agent._symmetry_maps is None


def test_maps_by_import_path():
    mirror_maps.obs_dim, mirror_maps.act_dim = _dims()
    agent, _ = make_ppo_agent(symmetry_loss={'coef': 0.25, 'maps': 'tests.test_symmetry_loss:mirror_maps'})
    assert agent.symmetry_loss_coef == 0.25
    assert agent._symmetry_maps[0].tolist() == list(reversed(range(mirror_maps.obs_dim)))


def test_rejects_bad_maps():
    obs_dim, act_dim = _dims()
    short = mirror_maps(obs_dim, act_dim)
    short['obs_perm'] = short['obs_perm'][:-1]
    with pytest.raises(ValueError, match='obs_perm'):
        make_ppo_agent(symmetry_loss={'maps': short})
    dup = mirror_maps(obs_dim, act_dim)
    dup['act_perm'] = [0] * act_dim
    with pytest.raises(ValueError, match='permutation'):
        make_ppo_agent(symmetry_loss={'maps': dup})


def test_rejects_recurrent_policies():
    obs_dim, act_dim = _dims()
    with pytest.raises(ValueError, match='feed-forward'):
        make_ppo_agent(rnn=True, symmetry_loss={'maps': mirror_maps(obs_dim, act_dim)})


def test_loss_matches_definition_and_leaves_obs_stats_alone():
    agent = _agent(normalize_input=True)
    obs = torch.randn(32, agent.obs_shape[0])
    agent.model.train()
    mu = agent.model.a2c_network({'obs': agent.model.norm_obs(obs)})[0]
    before = agent.model.running_mean_std.running_mean.clone()
    loss = agent.symmetry_loss(obs, mu)
    torch.testing.assert_close(agent.model.running_mean_std.running_mean, before)
    obs_perm, obs_sign, act_perm, act_sign = agent._symmetry_maps
    agent.model.running_mean_std.eval()
    mirrored_mu = agent.model.a2c_network({'obs': agent.model.norm_obs(obs[:, obs_perm] * obs_sign)})[0]
    expected = ((mirrored_mu - (mu[:, act_perm] * act_sign).detach()) ** 2).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.model.parameters())


def test_masked_rows_do_not_count():
    agent = _agent()
    obs = torch.randn(8, agent.obs_shape[0])
    mu = agent.model.a2c_network({'obs': agent.model.norm_obs(obs)})[0]
    masks = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    torch.testing.assert_close(agent.symmetry_loss(obs, mu, masks), agent.symmetry_loss(obs[:4], mu[:4]))


def test_train_epoch_logs_the_loss():
    agent = _agent()
    _rollout_batch(agent)
    agent.vec_env.set_train_info = lambda *args: None
    agent.train_epoch()
    assert 'symmetry_loss' in agent.aux_loss_dict
    assert torch.isfinite(agent.aux_loss_dict['symmetry_loss'][0])
