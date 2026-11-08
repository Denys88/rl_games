"""SAC FlashSAC options: the flash_sac network (shapes and parameter counts, weight projection,
gated per-group actor gradients, window encoder), categorical TD targets, n-step replay (stored and
memory-efficient), return-based reward scaling, held noise, the cosine LR schedule, and end-to-end
training on the point-mass tracking task with two action groups."""
import copy
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_sac_correctness import TEST_TRAIN_DIR, FakeNextStepVecEnv, SameStepFakeVecEnv, make_fake_env_sac_agent

from rl_games.algos_torch import sac_helper
from rl_games.algos_torch.flash_sac_network import FlashSACBuilder, ObsWindowEncoder
from rl_games.common import experience, schedulers
from rl_games.envs.test.point_mass_tracking import PointMassTracking


@pytest.fixture(autouse=True)
def few_threads():
    # many small CPU ops: dozens of intra-op threads on a busy box make them crawl
    before = torch.get_num_threads()
    torch.set_num_threads(min(4, before))
    yield
    torch.set_num_threads(before)


def flash_network(actor_units=32, critic_units=48, blocks=1, num_bins=51, window=None):
    net = {'name': 'flash_sac', 'actor': {'units': actor_units, 'blocks': blocks},
           'critic': {'units': critic_units, 'blocks': blocks, 'num_bins': num_bins, 'v_min': -5.0, 'v_max': 5.0},
           'log_std_bounds': [-5.0, 0.0]}
    if window:
        net['obs_window'] = window
    return net


FLASH_CONFIG = {
    'gamma': 0.98, 'n_step': 3, 'critic_tau': 0.01, 'policy_frequency': 2, 'critic_grad_clip': 0,
    'init_alpha': 0.01, 'min_alpha': 0.0, 'target_entropy_sigma': 0.15, 'learnable_temperature': True,
    'normalize_reward': True, 'held_noise': True, 'normalize_input': False,
}


def make_agent(env, network, seed=0, **config):
    """SAC agent on an injected env through the Runner (CPU)."""
    from rl_games.torch_runner import Runner
    cfg = {'name': 'pytest_flash_sac', 'env_name': 'point_mass_tracking', 'device': 'cpu',
           'num_actors': env.num_envs, 'env_info': env.get_env_info(), 'reward_shaper': {},
           'actor_lr': 1e-3, 'critic_lr': 1e-3, 'alpha_lr': 1e-3, 'batch_size': 64, 'num_steps_per_episode': 10,
           'num_updates_per_step': 2, 'num_warmup_frames': 1, 'replay_buffer_size': 4096, 'max_epochs': 2,
           'print_stats': False, 'save_best_after': 10 ** 6, 'save_frequency': 0,
           'train_dir': TEST_TRAIN_DIR}
    cfg.update(FLASH_CONFIG)
    cfg.update(config)
    params = {'seed': seed, 'algo': {'name': 'sac'}, 'model': {'name': 'soft_actor_critic'}, 'network': network,
              'config': cfg}
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    return runner.algo_factory.create('sac', base_name='pytest_flash_sac', params=runner.params)


@torch.no_grad()
def evaluate(agent, num_groups, steps=200, num_envs=64, seed=123):
    """Mean reward per step and group of the deterministic policy on fresh episodes."""
    env = PointMassTracking(num_envs, num_groups=num_groups, seed=seed, stagger=False)
    obs = env.reset()
    total = torch.zeros(num_groups)
    agent.set_eval()
    for _ in range(steps):
        obs, _, _, infos = env.step(agent.act(obs, None, sample=False))
        total += infos['reward_heads'].mean(dim=0)
    agent.set_train()
    return total / steps


# network


def by_type_groups(fingers):
    """Two hands with the action vector grouped by type: [wrist_trans R, L | wrist_rot R, L | fingers R, L]."""
    right = [0, 1, 2, 6, 7, 8, *range(12, 12 + fingers)]
    left = [3, 4, 5, 9, 10, 11, *range(12 + fingers, 12 + 2 * fingers)]
    return [right, left]


def tower_shapes(obs_dim, actions):
    """One actor tower: window encoder over the last 780 = 10 x 78 obs columns -> 128, trunk input
    obs - 780 + 128, unit linear to 128, 2 blocks 128 -> 512 -> 128, RMS norm, mean / log-std heads."""
    d = obs_dim - 780 + 128
    shapes = {
        'encoder.conv_in.weight': (128, 78, 1), 'encoder.conv_in.bias': (128,),
        'encoder.conv1.weight': (64, 128, 4), 'encoder.conv1.bias': (64,),   # k4 s2: 10 -> 4 steps
        'encoder.conv2.weight': (32, 64, 2), 'encoder.conv2.bias': (32,),    # k2 s1: 4 -> 3 steps
        'encoder.head.weight': (128, 96), 'encoder.head.bias': (128,),       # 32 x 3 -> latent 128
        'trunk.in_norm.weight': (d,), 'trunk.in_norm.bias': (d,), 'trunk.in_proj.weight': (128, d),
        'trunk.out_norm.weight': (128,),
        'head.mu.weight': (actions, 128), 'head.mu.bias': (actions,),
        'head.log_std.weight': (actions, 128), 'head.log_std.bias': (actions,),
    }
    for b in range(2):
        shapes.update({f'trunk.blocks.{b}.w1.weight': (512, 128), f'trunk.blocks.{b}.norm1.weight': (512,),
                       f'trunk.blocks.{b}.norm1.bias': (512,), f'trunk.blocks.{b}.w2.weight': (128, 512),
                       f'trunk.blocks.{b}.norm2.weight': (128,), f'trunk.blocks.{b}.norm2.bias': (128,)})
    return shapes


def critic_shapes(obs_dim, actions):
    """Critic: 4 members (2 twins x 2 groups), one shared window encoder, input obs - 780 + 128 + actions,
    unit linear to 1024, 2 blocks 1024 -> 4096 -> 1024, RMS norm, 101-bin head with bias."""
    d = obs_dim - 780 + 128 + actions
    shapes = {key: shape for key, shape in tower_shapes(obs_dim, 0).items() if key.startswith('encoder.')}
    shapes.update({'trunk.in_norm.weight': (4, d), 'trunk.in_norm.bias': (4, d), 'trunk.in_proj.weight': (4, 1024, d),
                   'trunk.out_norm.weight': (4, 1024), 'out_proj.weight': (4, 101, 1024), 'out_bias': (4, 101)})
    for b in range(2):
        shapes.update({f'trunk.blocks.{b}.w1.weight': (4, 4096, 1024), f'trunk.blocks.{b}.norm1.weight': (4, 4096),
                       f'trunk.blocks.{b}.norm1.bias': (4, 4096), f'trunk.blocks.{b}.w2.weight': (4, 1024, 4096),
                       f'trunk.blocks.{b}.norm2.weight': (4, 1024), f'trunk.blocks.{b}.norm2.bias': (4, 1024)})
    return shapes


# (obs dim, fingers per hand, params per actor tower, critic params). Tower = encoder 59,488 + in norm 2d
# + in proj 128d + blocks 2 x 132,352 + RMS 128 + heads 2 (128 a + a), d = obs - 652 (XHand 645,644);
# critic = encoder 59,488 + 4 x (2d' + 1024 d') + 2 x 33,595,392 + 4,096 + 4 x 101 x 1025 (XHand 77,813,556).
HANDS = {'xhand': (3088, 12, 645_644, 77_813_556), 'sharpa': (3360, 22, 683_584, 79_011_924)}


@pytest.mark.parametrize('hand', sorted(HANDS))
def test_reference_size_parameter_counts(hand):
    obs_dim, fingers, tower_params, critic_params = HANDS[hand]
    builder = FlashSACBuilder()
    builder.load({'actor': {'units': 128, 'blocks': 2}, 'critic': {'units': 1024, 'blocks': 2, 'num_bins': 101},
                  'obs_window': {'size': 780, 'steps': 10, 'hidden': 128, 'latent': 128}})
    actions = 2 * (6 + fingers)
    with torch.device('meta'):
        net = builder.build('sac', obs_dim=obs_dim, action_dim=actions, action_groups=by_type_groups(fingers))
    shapes = lambda m: {name: tuple(p.shape) for name, p in m.named_parameters()}
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert len(net.actor.towers) == 2 and net.critic.members == 4
    for tower in net.actor.towers:
        assert shapes(tower) == tower_shapes(obs_dim, 6 + fingers)
        assert count(tower) == tower_params
    assert shapes(net.critic) == critic_shapes(obs_dim, actions)
    assert count(net.critic) == count(net.critic_target) == critic_params


def _check_projected(module):
    for m in module.modules():
        if m is module or not hasattr(m, 'project'):
            continue
        name = type(m).__name__
        if name == 'UnitBatchNorm':
            sq = (m.weight ** 2 + m.bias ** 2).sum(-1)
            assert torch.allclose(sq, torch.full_like(sq, m.weight.shape[-1]), rtol=1e-4)
        elif name == 'UnitRMSNorm':
            sq = (m.weight ** 2).sum(-1)
            assert torch.allclose(sq, torch.full_like(sq, m.weight.shape[-1]), rtol=1e-4)
        else:
            norms = m.weight.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_weights_stay_projected_after_updates():
    env = PointMassTracking(16, num_groups=2)
    agent = make_agent(env, flash_network(window={'size': env.window, 'steps': 10, 'hidden': 16, 'latent': 8}),
                       actor_lr=1e-2, critic_lr=1e-2, max_epochs=3)
    net = agent.model.sac_network
    before = net.critic.trunk.in_proj.weight.detach().clone()
    agent.train()
    assert agent.update_counter > 0 and agent.actor_steps > 0
    assert not torch.equal(before, net.critic.trunk.in_proj.weight)  # it did move
    _check_projected(net.actor)
    _check_projected(net.critic)
    # policy head and window encoder are free
    head = net.actor.towers[0].head.mu.weight
    assert not torch.allclose(head.norm(dim=-1), torch.ones(head.shape[0]))


def test_policy_head_initial_std_and_bounds():
    torch.manual_seed(0)
    builder = FlashSACBuilder()
    builder.load(flash_network())
    net = builder.build('sac', obs_dim=20, action_dim=3)
    dist = net.actor(torch.randn(64, 20))
    # log-std weights start at zero: -5 + 0.5 * 5 * (tanh(0) + 1) = -2.5
    assert torch.allclose(dist.scale, torch.full_like(dist.scale, math.exp(-2.5)))
    assert dist.loc.abs().max() < 0.05  # mean weights ~ N(0, 1e-3)
    assert dist.rsample().abs().max() < 1.0


def test_gate_actions_route_each_group_gradient_to_its_dims():
    torch.manual_seed(0)
    builder = FlashSACBuilder()
    builder.load(flash_network())
    net = builder.build('sac', obs_dim=10, action_dim=5, action_groups=[[0, 2, 4], [1, 3]])
    actions = torch.rand(8, 5, requires_grad=True)
    q, _ = net.critic(torch.randn(8, 10), net.gate_actions(actions))
    q.view(2, 2, 8)[:, 0].sum().backward()  # both twins of group 0
    assert actions.grad[:, [0, 2, 4]].abs().sum() > 0
    assert torch.equal(actions.grad[:, [1, 3]], torch.zeros(8, 2))


def test_group_actor_returns_env_order():
    torch.manual_seed(0)
    builder = FlashSACBuilder()
    builder.load(flash_network())
    net = builder.build('sac', obs_dim=6, action_dim=4, action_groups=[[3, 0], [1, 2]])
    obs = torch.randn(5, 6)
    dist = net.actor(obs)
    mu0, _ = net.actor.towers[0](obs)
    mu1, _ = net.actor.towers[1](obs)
    assert torch.equal(dist.loc[:, [3, 0]], mu0) and torch.equal(dist.loc[:, [1, 2]], mu1)


def test_window_encoder_reads_steps_in_order():
    enc = ObsWindowEncoder(obs_dim=800, size=780, steps=10)
    obs = torch.randn(5, 800)
    out = enc(obs)
    assert out.shape == (5, 20 + 128) and torch.equal(out[:, :20], obs[:, :20])
    window = obs[:, 20:].reshape(5, 10, 78)[:, torch.randperm(10)].reshape(5, 780)
    assert not torch.allclose(enc(torch.cat([obs[:, :20], window], -1)), out)


def test_single_group_reduces_to_twin_q_sac():
    """One group: one tower, two critic members (the twins), one temperature, the plain SAC log-prob
    and an ungated action."""
    torch.manual_seed(0)
    env = PointMassTracking(16, num_groups=1)
    agent = make_agent(env, flash_network(num_bins=0), n_step=1, normalize_reward=False)
    net = agent.model.sac_network
    assert agent.num_heads == 1 and len(net.actor.towers) == 1 and net.critic.members == 2
    assert agent.log_alpha.shape == (1,)
    dist = net.actor(torch.randn(32, env.obs_dim))
    a = dist.rsample()
    assert torch.allclose(agent._group_log_prob(dist, a), agent._env_log_prob(dist, a))
    assert torch.equal(net.gate_actions(a)[0], a) and torch.equal(net.gate_actions(a)[1], a)


def test_critic_heads_loss_matches_plain_sac_formula():
    """Scalar head, one group: the per-group critic loss equals 0.5 (MSE(Q1, y) + MSE(Q2, y)) with
    y = r + gamma (1 - d) (min(Q1', Q2') - alpha log pi(a'))."""
    torch.manual_seed(0)
    env = PointMassTracking(16, num_groups=1)
    agent = make_agent(env, flash_network(num_bins=0), n_step=1, normalize_reward=False)
    net = agent.model.sac_network
    n = 32
    obs, next_obs = torch.randn(n, env.obs_dim), torch.randn(n, env.obs_dim)
    action, reward = torch.rand(n, 2) * 2 - 1, torch.randn(n, 1)
    done = torch.rand(n, 1) < 0.2
    online = copy.deepcopy(net.critic)
    target_net = copy.deepcopy(net.critic_target)
    rng = torch.get_rng_state()
    critic_loss, c1, c2 = agent.update_critic_heads(obs, action, reward, next_obs, done, None)

    torch.set_rng_state(rng)
    with torch.no_grad():
        net.actor.eval()
        dist = net.actor(next_obs)
        next_action = dist.rsample()
        log_pi = dist.log_prob(next_action).sum(-1)
        net.actor.train()
        obs_all, act_all = torch.cat([obs, next_obs]), torch.cat([action, next_action])
        q_t, _ = target_net(obs_all, act_all)
        y = reward[:, 0] + (~done[:, 0]) * agent.gamma * (torch.min(q_t[0, n:], q_t[1, n:]) - agent.alpha * log_pi)
        q, _ = online(obs_all, act_all)
        mse1, mse2 = ((q[0, :n] - y) ** 2).mean(), ((q[1, :n] - y) ** 2).mean()
    assert torch.allclose(c1, mse1, rtol=1e-5) and torch.allclose(c2, mse2, rtol=1e-5)
    assert torch.allclose(critic_loss, 0.5 * (mse1 + mse2), rtol=1e-5)


def test_actor_loss_trains_each_tower_from_its_own_group():
    torch.manual_seed(0)
    env = PointMassTracking(16, num_groups=2)
    agent = make_agent(env, flash_network())
    net = agent.model.sac_network
    obs, next_obs = torch.randn(32, env.obs_dim), torch.randn(32, env.obs_dim)
    for group in range(2):
        net.actor.zero_grad(set_to_none=True)
        net.critic.zero_grad(set_to_none=True)
        dist = net.actor(torch.cat([obs, next_obs]))
        a = dist.rsample()
        log_prob = agent._group_log_prob(dist, a)[:32]
        net.critic.eval()
        q, _ = net.critic(obs, net.gate_actions(a[:32]))
        net.critic.train()
        q = q.view(2, 2, 32).amin(dim=0)
        (agent.alpha[group] * log_prob[:, group] - q[group]).mean().backward()
        own, other = net.actor.towers[group], net.actor.towers[1 - group]
        assert all(p.grad is None or not p.grad.any() for p in other.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in own.parameters())


# categorical TD target

BINS = 101  # on [-5, 5]: bin k sits at -5 + 0.1 k


def _one_hot(index, n=1, bins=BINS):
    p = torch.zeros(n, bins)
    p[:, index] = 1.0
    return p


def test_projection_lands_on_expected_bins():
    probs = _one_hot(60, 4)  # all mass at z = 1.0
    out = sac_helper.project_categorical(probs, torch.tensor([0.25, 0.25, 0.25, 10.0]),
                                         torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.2, 0.0]),
                                         0.5, -5.0, 5.0)
    # 0.25 + 0.5 * 1.0 = 0.75 -> position 57.5: half on bins 57 and 58
    assert out[0, 57] == pytest.approx(0.5) and out[0, 58] == pytest.approx(0.5)
    # terminated: the atom collapses to the reward 0.25 -> position 52.5
    assert out[1, 52] == pytest.approx(0.5) and out[1, 53] == pytest.approx(0.5)
    # entropy bonus: 0.25 + 0.5 * (1.0 - 0.2) = 0.65 -> position 56.5
    assert out[2, 56] == pytest.approx(0.5) and out[2, 57] == pytest.approx(0.5)
    assert out[3, 100] == pytest.approx(1.0)  # clamped to the top bin
    assert torch.allclose(out.sum(-1), torch.ones(4))


def test_projection_keeps_the_mean_and_takes_per_row_discounts():
    torch.manual_seed(0)
    spread = torch.softmax(torch.randn(16, 41), -1)
    bins = torch.linspace(-2.0, 6.0, 41)  # asymmetric support
    r, disc = torch.full((16,), 0.1), torch.rand(16) * 0.5
    out = sac_helper.project_categorical(spread * 0 + _one_hot(20, 16, 41), r, torch.zeros(16), torch.zeros(16),
                                         disc, -2.0, 6.0)
    assert torch.allclose(out.sum(-1), torch.ones(16), atol=1e-5)
    out = sac_helper.project_categorical(spread, r, torch.zeros(16), torch.zeros(16), disc, -2.0, 6.0)
    # no clamping here: E[r + g z] = r + g E[z]
    assert torch.allclose((out * bins).sum(-1), r + disc * (spread * bins).sum(-1), atol=1e-4)


def test_min_twin_selection():
    q = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, 0.5]])
    log_p = torch.log_softmax(torch.randn(2, 3, BINS), -1)
    sel = sac_helper.min_twin_log_probs(q, log_p)
    assert torch.equal(sel[0], log_p[1, 0]) and torch.equal(sel[1], log_p[0, 1]) and torch.equal(sel[2], log_p[0, 2])


def test_target_entropy_per_group():
    # 0.5 n ln(2 pi e 0.15^2): 18 and 28 dims per hand
    assert sac_helper.gaussian_entropy(18, 0.15) == pytest.approx(-8.6073, abs=1e-4)
    assert sac_helper.gaussian_entropy(28, 0.15) == pytest.approx(-13.3891, abs=1e-4)
    agent = make_agent(PointMassTracking(8, num_groups=3), flash_network())
    assert agent.target_entropy.tolist() == pytest.approx([sac_helper.gaussian_entropy(2, 0.15)] * 3)
    assert agent.log_alpha.shape == (3,) and torch.allclose(agent.alpha, torch.full((3,), 0.01))


# replay


def _obs(step, num_envs, dim=3, final=False):
    """Obs that encode (step, env): column 0 = step (+0.5 for a pre-reset final obs), column 1 = env."""
    obs = torch.zeros(num_envs, dim)
    obs[:, 0] = step + (0.5 if final else 0.0)
    obs[:, 1] = torch.arange(num_envs)
    return obs


def _feed(buffer, steps, rewards, terminated=None, truncated=None, live=None):
    b = rewards[0].shape[0]
    for t in range(steps):
        term = torch.zeros(b, dtype=torch.bool) if terminated is None else terminated[t]
        trunc = torch.zeros(b, dtype=torch.bool) if truncated is None else truncated[t]
        done = (term | trunc).unsqueeze(-1)
        next_obs = torch.where(done, _obs(t, b, final=True), _obs(t + 1, b))
        kwargs = {} if live is None else {'live': live[t]}
        buffer.add(_obs(t, b), torch.full((b, 2), float(t)), rewards[t], next_obs, term.unsqueeze(1),
                   trunc.unsqueeze(1), **kwargs)


def _episode_ends(b, steps):
    term = [torch.zeros(b, dtype=torch.bool) for _ in range(steps)]
    trunc = [torch.zeros(b, dtype=torch.bool) for _ in range(steps)]
    term[1][1] = True    # env 1 fails at step 1
    trunc[1][2] = True   # env 2 times out at step 1
    term[2][2] = True    # ... and its next episode fails at step 2 (after the time-out: must not leak in)
    return term, trunc


@pytest.mark.parametrize('memory_efficient', [False, True])
def test_n_step_return_flags_discount_and_next_obs(memory_efficient):
    b, n, gamma, steps = 3, 3, 0.5, 6
    if memory_efficient:
        buffer = experience.MemoryEfficientReplayBuffer((3,), (2,), 64, 'cpu', num_envs=b, num_heads=2, n_step=n,
                                                        gamma=gamma)
    else:
        buffer = experience.VectorizedReplayBuffer((3,), (2,), 64, 'cpu', num_heads=2, n_step=n, gamma=gamma)
    rewards = [torch.tensor([[1.0, 10.0]] * b) * (t + 1) for t in range(steps)]
    term, trunc = _episode_ends(b, steps)
    _feed(buffer, steps, rewards, term, trunc)

    def anchored(step):
        if memory_efficient:
            return buffer.sample_batch(0, rows=torch.arange(step * b, (step + 1) * b))
        rows = torch.arange(step * b, (step + 1) * b)
        return {k: getattr(buffer, a)[rows] for k, a in
                [('reward', 'rewards'), ('next_obs', 'next_obses'), ('done', 'dones'), ('truncated', 'truncated'),
                 ('discount', 'discounts'), ('obs', 'obses')]}

    batch = anchored(0)
    # env 0: full window, r0 + g r1 + g^2 r2 = 1 + 0.5 * 2 + 0.25 * 3 = 2.75 (x10 for head 1), bootstrap g^3
    assert batch['reward'][0].tolist() == pytest.approx([2.75, 27.5])
    assert not batch['done'][0] and not batch['truncated'][0]
    assert batch['next_obs'][0, 0] == 3.0 and batch['discount'][0] == pytest.approx(0.125)
    # env 1: terminated at step 1 -> r0 + g r1, no bootstrap
    assert batch['reward'][1].tolist() == pytest.approx([2.0, 20.0]) and batch['done'][1]
    # env 2: timed out at step 1 -> r0 + g r1, bootstraps from the true final obs of step 1 with g^2
    assert batch['reward'][2].tolist() == pytest.approx([2.0, 20.0])
    assert batch['truncated'][2] and not batch['done'][2]
    assert batch['next_obs'][2, 0] == 1.5 and batch['next_obs'][2, 1] == 2
    assert batch['discount'][2] == pytest.approx(0.25)
    batch = anchored(1)
    assert batch['truncated'][2] and batch['next_obs'][2, 0] == 1.5 and batch['discount'][2] == pytest.approx(0.5)
    batch = anchored(2)  # env 2 terminates at once
    assert batch['done'][2] and batch['reward'][2, 0] == pytest.approx(3.0)


def test_stored_n_step_skips_rows_anchored_at_reset_steps():
    # next_step autoreset: the step after an episode end is a reset row, written nowhere
    b, n, steps = 2, 2, 5
    buffer = experience.VectorizedReplayBuffer((3,), (2,), 64, 'cpu', n_step=n, gamma=0.9)
    trunc = [torch.tensor([t == 1, False]) for t in range(steps)]
    live = [torch.tensor([t != 2, True]) for t in range(steps)]
    _feed(buffer, steps, [torch.ones(b) for _ in range(steps)], truncated=trunc, live=live)
    written = buffer.obses[:buffer.idx]
    assert buffer.idx == (steps - n + 1) * b - 1
    assert not ((written[:, 0] == 2) & (written[:, 1] == 0)).any()


def test_memory_efficient_next_obs_across_ring_wrap():
    b, n, cap = 4, 2, 23  # 23 is not a multiple of 4: batches straddle the wrap at varying offsets
    buffer = experience.MemoryEfficientReplayBuffer((3,), (2,), cap, 'cpu', num_envs=b, n_step=n, gamma=0.9)
    steps = 40
    _feed(buffer, steps, [torch.ones(b) for _ in range(steps)])
    assert buffer.size == cap
    rows = (torch.arange(cap - n * b) + buffer.ptr) % cap  # oldest first
    batch = buffer.sample_batch(0, rows=rows)
    assert torch.equal(batch['next_obs'][:, 0], batch['obs'][:, 0] + n)
    assert torch.equal(batch['next_obs'][:, 1], batch['obs'][:, 1])  # same env
    torch.manual_seed(0)
    batch = buffer.sample_batch(512)  # uniform sampling only returns rows whose next row exists
    assert torch.equal(batch['next_obs'][:, 0], batch['obs'][:, 0] + n)
    assert batch['obs'][:, 0].max() <= steps - 2 * n


def test_memory_efficient_final_obs_pool_survives_many_wraps():
    torch.manual_seed(0)
    b, n, cap, steps = 8, 3, 101, 400
    buffer = experience.MemoryEfficientReplayBuffer((3,), (2,), cap, 'cpu', num_envs=b, n_step=n, gamma=0.9)
    trunc = [torch.rand(b) < 0.15 for _ in range(steps)]
    term = [(torch.rand(b) < 0.05) & ~trunc[t] for t in range(steps)]
    _feed(buffer, steps, [torch.ones(b) for _ in range(steps)], term, trunc)
    batch = buffer.sample_batch(0, rows=(torch.arange(cap - n * b) + buffer.ptr) % cap)
    timeout = batch['truncated'][:, 0]
    assert timeout.any() and batch['done'].any()
    anchor, env = batch['obs'][:, 0], batch['obs'][:, 1].long()
    for i in torch.nonzero(timeout).squeeze(-1).tolist():
        t0 = int(anchor[i])
        first = next(t for t in range(t0, t0 + n) if trunc[t][env[i]] or term[t][env[i]])
        assert trunc[first][env[i]] and batch['next_obs'][i, 0] == first + 0.5
        assert batch['discount'][i, 0] == pytest.approx(0.9 ** (first - t0 + 1))
    plain = ~timeout & ~batch['done'][:, 0]
    assert torch.equal(batch['next_obs'][plain, 0], anchor[plain] + n)
    assert len(buffer._pool_birth) < int(torch.stack(trunc).sum())  # recycled, not grown per time-out


class _DropTail:
    """Compactor test double: the last two columns are aux * 2 and aux * 3 and are not stored."""
    compact_dim, aux_dim = 3, 1

    def compact(self, full):
        return full[:, :3]

    def expand(self, compact, aux):
        a = aux.float()
        return torch.cat([compact, 2 * a, 3 * a], dim=-1)


def test_n_step_buffer_loads_a_one_step_checkpoint():
    one = experience.VectorizedReplayBuffer((3,), (2,), 32, 'cpu', n_step=1)
    one.add(torch.randn(4, 3), torch.randn(4, 2), torch.randn(4, 1), torch.randn(4, 3),
            torch.zeros(4, 1, dtype=torch.bool))
    folded = experience.VectorizedReplayBuffer((3,), (2,), 32, 'cpu', n_step=3, gamma=0.9)
    folded.load_state_dict(one.state_dict())
    assert torch.allclose(folded.discounts, torch.full_like(folded.discounts, 0.9))
    assert torch.equal(folded.obses, one.obses)


def test_one_step_buffer_counts_terminated_and_truncated_as_terminated():
    buf = experience.VectorizedReplayBuffer((3,), (2,), 8, 'cpu', n_step=1)
    done = torch.tensor([[True], [False]])
    truncated = torch.tensor([[True], [True]])
    buf.add(torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 1), torch.randn(2, 3), done, truncated)
    assert buf.dones[:2].flatten().tolist() == [True, False]
    assert buf.truncated[:2].flatten().tolist() == [False, True]


def test_fp16_replay_reports_rounding_once(capsys):
    buf = experience.VectorizedReplayBuffer((2,), (1,), 8, 'cpu', obs_dtype=torch.float16)
    obs = torch.tensor([[0.01, 50.003], [0.02, 49.997]])
    buf.add(obs, torch.zeros(2, 1), torch.zeros(2, 1), obs, torch.zeros(2, 1, dtype=torch.bool))
    buf.add(obs, torch.zeros(2, 1), torch.zeros(2, 1), obs, torch.zeros(2, 1, dtype=torch.bool))
    out = capsys.readouterr().out
    assert out.count('rounds 1 column') == 1 and '1 (max |x| 50' in out


def test_compactor_rebuilds_dropped_columns():
    b, n = 4, 2
    buffer = experience.MemoryEfficientReplayBuffer((5,), (2,), 64, 'cpu', num_envs=b, n_step=n, gamma=0.9,
                                                    obs_dtype=torch.float16, compactor=_DropTail())
    assert buffer.obs.shape == (64, 3) and buffer.obs.dtype == torch.float16 and buffer.aux.dtype == torch.int32

    def full(step, final=False):
        aux = torch.arange(b).unsqueeze(-1) + 100 * step + (50 if final else 0)
        return torch.cat([_obs(step, b, final=final), 2 * aux, 3 * aux], -1), aux

    for t in range(6):
        trunc = torch.tensor([t == 1, False, False, False])
        obs, aux = full(t)
        nxt, nxt_aux = full(t + 1)
        fin, fin_aux = full(t, final=True)
        nxt = torch.where(trunc.unsqueeze(-1), fin, nxt)
        nxt_aux = torch.where(trunc.unsqueeze(-1), fin_aux, nxt_aux)
        buffer.add(obs, torch.zeros(b, 2), torch.ones(b), nxt, torch.zeros(b), trunc, obs_aux=aux,
                   next_obs_aux=nxt_aux)
    batch = buffer.sample_batch(0, rows=torch.arange(8))
    assert batch['obs'].dtype == torch.float32
    assert torch.allclose(batch['obs'][:4], full(0)[0])
    assert torch.allclose(batch['next_obs'][1:4], full(2)[0][1:4])
    assert torch.allclose(batch['next_obs'][0], full(1, final=True)[0][0])  # the pool keeps aux too


def test_stored_replay_fp16_obs_and_one_step_layout_unchanged():
    buffer = experience.VectorizedReplayBuffer((3,), (2,), 16, 'cpu', obs_dtype=torch.float16)
    assert buffer.obses.dtype == torch.float16 and buffer.discounts is None
    buffer.add(_obs(0, 2), torch.zeros(2, 2), torch.ones(2, 1), _obs(1, 2), torch.zeros(2, 1, dtype=torch.bool))
    batch = buffer.sample_batch(4)
    assert batch['obs'].dtype == torch.float32 and batch['discount'] is None
    assert len(buffer.sample(4)) == 6


class _StepCountTracking(PointMassTracking):
    """Obs end in the episode step, which the replay drops and rebuilds from the int aux."""

    def _full(self, obs):
        return {'obs': torch.cat([obs, self.t.float().unsqueeze(-1)], -1), 'obs_aux': self.t.int().unsqueeze(-1)}

    def reset(self):
        return self._full(super().reset())

    def step(self, actions):
        t_end = self.t + 1  # step count of the pre-reset final obs
        obs, rew, done, infos = super().step(actions)
        infos['final_observation'] = torch.cat([infos['final_observation'], t_end.float().unsqueeze(-1)], -1)
        infos['final_obs_aux'] = t_end.int().unsqueeze(-1)
        return self._full(obs), rew, done, infos

    def get_env_info(self):
        from gymnasium import spaces
        info = super().get_env_info()
        info['observation_space'] = spaces.Box(-math.inf, math.inf, (self.obs_dim + 1,))
        return info


class _StepCountCompactor:
    def __init__(self, env_info):
        self.compact_dim, self.aux_dim = env_info['observation_space'].shape[0] - 1, 1

    def compact(self, full):
        return full[:, :-1]

    def expand(self, compact, aux):
        return torch.cat([compact, aux.float()], dim=-1)


def test_agent_feeds_the_compactor_aux_and_final_aux():
    env = _StepCountTracking(16, num_groups=2, episode_length=12)
    agent = make_agent(env, flash_network(), n_step=3, max_epochs=4, replay_memory_efficient=True,
                       replay_obs_compactor=_StepCountCompactor(env.get_env_info()))
    agent.train()
    buffer = agent.replay_buffer
    assert buffer.obs.shape[1] == env.obs_dim and buffer.aux.shape[1] == 1
    batch = buffer.sample_batch(512)
    step, next_step = batch['obs'][:, -1], batch['next_obs'][:, -1]
    timeout, terminated = batch['truncated'][:, 0], batch['done'][:, 0]
    assert timeout.any()
    assert (next_step[timeout] == 12).all()  # the true final obs of a time-out, rebuilt from final_obs_aux
    plain = ~timeout & ~terminated
    assert torch.equal(next_step[plain], step[plain] + 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_host_replay_is_mapped_into_the_gpu():
    b, n = 16, 2
    buffer = experience.MemoryEfficientReplayBuffer((3,), (2,), 200, 'cuda:0', num_envs=b, num_heads=2, n_step=n,
                                                    gamma=0.9, obs_dtype=torch.float16, storage='host')
    assert buffer.storage.mapped and buffer.obs.is_cuda and next(iter(buffer.storage.host_tensors.values())).is_pinned()
    for t in range(30):
        trunc = torch.zeros(b, dtype=torch.bool)
        trunc[t % b] = True
        nxt = torch.where(trunc.unsqueeze(-1), _obs(t, b, final=True), _obs(t + 1, b))
        buffer.add(_obs(t, b).cuda(), torch.zeros(b, 2), torch.ones(b, 2), nxt.cuda(), torch.zeros(b), trunc)
    batch = buffer.sample_batch(256)
    timeout = batch['truncated'][:, 0]
    assert batch['obs'].is_cuda
    assert torch.equal(batch['next_obs'][~timeout, 0], batch['obs'][~timeout, 0] + n)
    assert (batch['next_obs'][timeout, 0] % 1 == 0.5).all()
    host = next(iter(buffer.storage.host_tensors.values()))  # rows written by GPU kernels are the host pages
    assert host.data_ptr() == buffer.obs.data_ptr() and torch.equal(host, buffer.obs.cpu())


# default network with the new options


@pytest.mark.parametrize('env_cls', [FakeNextStepVecEnv, SameStepFakeVecEnv])
def test_n_step_trains_with_the_default_network(env_cls):
    agent, _ = make_fake_env_sac_agent(env_cls=env_cls, n_step=3, max_epochs=3)
    agent.train()
    buf = agent.replay_buffer
    assert agent.update_counter > 0 and buf.discounts is not None
    written = buf.discounts[:buf.idx, 0]
    # full windows bootstrap with gamma^3; ends inside the window with gamma^k, k < 3
    assert torch.isclose(written, torch.tensor(agent.gamma ** 3)).any()
    assert (written <= agent.gamma + 1e-6).all()


def test_memory_efficient_replay_rejects_next_step_envs():
    with pytest.raises(ValueError, match='same_step'):
        make_fake_env_sac_agent(replay_memory_efficient=True)


def test_groups_need_an_ensemble_critic():
    env = PointMassTracking(8, num_groups=2)
    with pytest.raises(ValueError, match='ensemble critic'):
        make_agent(env, {'name': 'soft_actor_critic', 'separate': True,
                         'mlp': {'units': [16], 'activation': 'relu', 'initializer': {'name': 'default'}},
                         'log_std_bounds': [-5, 2]})


class _NoHeadsTracking(PointMassTracking):
    def step(self, actions):
        obs, rew, done, infos = super().step(actions)
        del infos['reward_heads']
        return obs, rew, done, infos


def test_groups_need_reward_heads():
    with pytest.raises(ValueError, match='reward_heads'):
        make_agent(_NoHeadsTracking(8, num_groups=2), flash_network()).train()


def test_scaled_rewards_reach_the_critic():
    env = PointMassTracking(16, num_groups=2)
    agent = make_agent(env, flash_network(), max_epochs=2)
    agent.train()
    seen = {}
    update_critic_heads = agent.update_critic_heads

    def spy(obs, action, reward, *args):
        seen['reward'] = reward
        return update_critic_heads(obs, action, reward, *args)

    agent.update_critic_heads = spy
    torch.manual_seed(0)
    agent.update(0)
    torch.manual_seed(0)
    raw = agent.replay_buffer.sample_batch(agent.batch_size)['reward']
    assert torch.allclose(seen['reward'], agent.reward_scaler.scale(raw))
    assert not torch.allclose(seen['reward'], raw)


def test_flash_min_alpha_is_one_temperature_and_edge_mass_is_logged():
    env = PointMassTracking(16, num_groups=2)
    agent = make_agent(env, flash_network(), max_epochs=2, init_alpha=0.01, min_alpha=0.05)
    agent.train()
    assert (agent.log_alpha.detach() >= math.log(0.05) - 1e-6).all()
    agent.update(0)
    for key in ('info/target_edge_low', 'info/target_edge_high'):
        mass = torch.stack(agent._head_stats[key])
        assert mass.shape[-1] == 2 and ((mass >= 0) & (mass <= 1)).all()


# reward scaling, held noise, LR schedule


def test_return_scaler_matches_direct_computation():
    torch.manual_seed(0)
    gamma, max_return = 0.9, 5.0
    scaler = sac_helper.ReturnScaler(num_heads=2, gamma=gamma, max_return=max_return, device='cpu')
    returns = torch.zeros(4, 2)
    prev_done = torch.zeros(4, dtype=torch.bool)
    history = []
    for _ in range(30):
        r = torch.randn(4, 2) * torch.tensor([1.0, 10.0])
        done = torch.rand(4) < 0.1
        scaler.update(r, done)
        returns = gamma * (~prev_done).float().unsqueeze(-1) * returns + r
        prev_done = done
        history.append(returns.clone())
    values = torch.stack(history).view(-1, 2)
    assert torch.allclose(scaler.abs_max, values.abs().amax(0))
    assert torch.allclose(scaler.mean, values.mean(0), atol=1e-5)
    assert torch.allclose(scaler.var, values.var(0, correction=0), rtol=1e-3)  # up to the 1e-4 prior count
    r = torch.ones(3, 2)
    denom = torch.maximum(torch.sqrt(scaler.var + 1e-8), scaler.abs_max / max_return)
    assert torch.allclose(scaler.scale(r), r / denom)


def test_return_scaler_closes_the_episode_on_its_terminal_step():
    scaler = sac_helper.ReturnScaler(num_heads=1, gamma=0.9, max_return=5.0, device='cpu')
    seen = []
    for done in (False, False, True, False):
        scaler.update(torch.ones(1, 1), torch.tensor([done]))
        seen.append(scaler.returns.item())
    assert seen == pytest.approx([1.0, 1.9, 2.71, 1.0])
    restored = sac_helper.ReturnScaler(num_heads=1, gamma=0.9, max_return=5.0, device='cpu')
    restored.load_state_dict(scaler.state_dict())
    restored.update(torch.ones(1, 1), torch.tensor([False]))
    assert restored.returns.item() == pytest.approx(1.9)


def test_held_noise_repeats_per_env_and_zeta_lengths():
    pmf = sac_helper.truncated_zeta_pmf(2.0, 16)
    assert pmf[0] == pytest.approx(1 / sum(1 / n ** 2 for n in range(1, 17)))
    torch.manual_seed(0)
    noise = sac_helper.HeldNoise(2.0, 16)
    num_envs = 64
    like = torch.zeros(num_envs, 3)
    last = noise(like).clone()
    run = torch.ones(num_envs, dtype=torch.long)
    lengths, changed_at = [], []
    for _ in range(600):
        eps = noise(like)
        same = (eps == last).all(dim=-1)
        lengths += run[~same].tolist()
        changed_at.append(~same)
        run = torch.where(same, run + 1, torch.ones_like(run))
        last = eps.clone()
    lengths = torch.tensor(lengths, dtype=torch.float64)
    assert lengths.min() >= 1 and lengths.max() <= 16
    assert lengths.mean().item() == pytest.approx(float((pmf * torch.arange(1, 17)).sum()), rel=0.05)  # 2.13
    assert (lengths == 1).double().mean().item() == pytest.approx(float(pmf[0]), abs=0.02)
    changed = torch.stack(changed_at)
    assert ((changed.any(dim=1)) & ~(changed.all(dim=1))).float().mean() > 0.9  # envs resample on their own steps


def test_held_noise_draws_fresh_noise_at_episode_starts():
    torch.manual_seed(0)
    noise = sac_helper.HeldNoise(2.0, 16)
    like = torch.zeros(8, 3)
    first = noise(like).clone()
    noise.left.fill_(10)  # every draw still held
    reset = torch.zeros(8, dtype=torch.bool)
    reset[0] = True
    second = noise(like, reset)
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1:], second[1:])


def test_cosine_lr_factor():
    assert schedulers.cosine_lr_factor(0, 0, 1000, 0.5) == pytest.approx(1.0)
    assert schedulers.cosine_lr_factor(500, 0, 1000, 0.5) == pytest.approx(0.75)
    assert schedulers.cosine_lr_factor(1005, 0, 1000, 0.5) == pytest.approx(0.5)
    assert schedulers.cosine_lr_factor(5, 10, 1000, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize('per_optimizer_step', [False, True])
def test_cosine_schedule_and_half_rate_quirk(per_optimizer_step):
    env = PointMassTracking(8, num_groups=2)
    agent = make_agent(env, flash_network(), lr_schedule='cosine', lr_per_optimizer_step=per_optimizer_step,
                       max_epochs=3, num_steps_per_episode=4, num_updates_per_step=1, num_warmup_frames=1)
    assert agent.lr_decay_steps == 3 * 4 * 1
    agent.train()
    # 8 updates (2 epochs of 4 steps after warm-up), the actor stepped on every 2nd
    assert agent.update_counter == 8 and agent.actor_steps == 4
    factor = lambda step: schedulers.cosine_lr_factor(step, 0, 12, 0.5)
    assert agent.critic_optimizer.param_groups[0]['lr'] == pytest.approx(1e-3 * factor(7))
    actor_step = 3 if per_optimizer_step else 7
    assert agent.actor_optimizer.param_groups[0]['lr'] == pytest.approx(1e-3 * factor(actor_step))
    assert agent.log_alpha_optimizer.param_groups[0]['lr'] == pytest.approx(1e-3 * factor(actor_step))


# end to end


def test_two_group_point_mass_learns():
    """64 envs, small networks, 300 env steps x 4 updates of 256 on CPU: the deterministic policy's
    reward per step, summed over both groups, rises from ~0.14 to 0.36-0.74 (seeds 0-6) and each
    group improves."""
    torch.manual_seed(3)
    env = PointMassTracking(64, num_groups=2, seed=3)
    window = {'size': env.window, 'steps': 10, 'hidden': 16, 'latent': 16}
    agent = make_agent(env, flash_network(actor_units=32, critic_units=64, blocks=2, num_bins=51, window=window),
                       seed=3, n_step=5, actor_lr=5e-4, critic_lr=5e-4, alpha_lr=5e-4, batch_size=256,
                       num_updates_per_step=4, num_warmup_frames=64 * 20, replay_buffer_size=64 * 200,
                       replay_memory_efficient=True, replay_obs_dtype='float16', max_epochs=30)
    before = evaluate(agent, 2)
    agent.train()
    after = evaluate(agent, 2)
    assert after.sum() > max(2.0 * before.sum(), before.sum() + 0.15), (before, after)
    assert (after > before).all(), (before, after)


@pytest.mark.parametrize('num_groups', [1, 3])
def test_any_number_of_groups_trains(num_groups):
    env = PointMassTracking(16, num_groups=num_groups)
    agent = make_agent(env, flash_network(), max_epochs=3, replay_memory_efficient=num_groups == 3)
    agent.train()
    assert agent.update_counter > 0 and torch.isfinite(agent.log_alpha).all()
    assert agent.act(env.reset(), None, sample=True).shape == (16, 2 * num_groups)
    if num_groups > 1:
        assert agent.replay_buffer.reward.shape[1] == num_groups


def test_checkpoint_round_trip_and_player(tmp_path):
    from rl_games.algos_torch.players import SACPlayer
    env = PointMassTracking(16, num_groups=2)
    agent = make_agent(env, flash_network(), max_epochs=3)
    agent.train()
    agent.save(str(tmp_path / 'flash'))
    path = str(tmp_path / 'flash.pth')
    clone = make_agent(env, flash_network(), seed=1)
    clone.restore(path)
    obs = env.reset()
    agent.set_eval()
    clone.set_eval()
    assert torch.equal(clone.act(obs, None), agent.act(obs, None))
    assert clone.update_counter == agent.update_counter and torch.equal(clone.log_alpha, agent.log_alpha)
    assert torch.equal(clone.reward_scaler.var, agent.reward_scaler.var)

    params = {'algo': {'name': 'sac'}, 'model': {'name': 'soft_actor_critic'}, 'network': flash_network(),
              'config': {'env_name': 'point_mass_tracking', 'env_info': env.get_env_info(), 'vec_env': env,
                         'device': 'cpu', 'device_name': 'cpu', 'player': {'games_num': 1}}}
    from rl_games.algos_torch import model_builder
    params['config']['network'] = model_builder.ModelBuilder().load(params)
    player = SACPlayer(params)
    player.restore(path)
    player.has_batch_dimension = True  # set by run() for batched envs
    assert torch.equal(player.get_action(obs, is_deterministic=True), agent.act(obs, None))
