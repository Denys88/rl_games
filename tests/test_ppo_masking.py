"""B1-PPO: next_step-autoreset garbage-row masking.

On next_step-autoreset envs the reset step's row (action ignored, filler
reward, obs = previous episode's terminal obs) must be excluded from PPO
losses, advantage normalization, and episode stats. same_step paths must be
untouched. Fixtures are shared with the SAC correctness suite.
"""

import copy

import numpy as np
import pytest
import torch

from tests.test_sac_correctness import (
    FakeNextStepVecEnv,
    StaggeredFakeNextStepVecEnv,
    SameStepFakeVecEnv,
    EP_LEN,
    OBS_DIM,
    ACT_DIM,
)

HORIZON = 3 * EP_LEN  # several episode boundaries per env per rollout
NUM_ENVS = 4


def _ppo_params(num_envs=NUM_ENVS, env_info=None, rnn=False, **config_overrides):
    network = {
        'name': 'actor_critic', 'separate': False,
        'space': {'continuous': {
            'mu_activation': 'None', 'sigma_activation': 'None',
            'mu_init': {'name': 'default'},
            'sigma_init': {'name': 'const_initializer', 'val': 0.0},
            'fixed_sigma': True}},
        'mlp': {'units': [16], 'activation': 'elu',
                'initializer': {'name': 'default'}},
    }
    if rnn:
        network['rnn'] = {'name': 'lstm', 'units': 8, 'layers': 1}
    config = {
        'name': 'pytest_ppo_masking', 'env_name': 'unused',
        'reward_shaper': {'scale_value': 1.0},
        'device': 'cpu', 'multi_gpu': False, 'mixed_precision': False,
        'normalize_input': False, 'normalize_value': False,
        'normalize_advantage': True, 'value_bootstrap': False,
        'num_actors': num_envs, 'horizon_length': HORIZON,
        'minibatch_size': num_envs * HORIZON, 'mini_epochs': 1,
        'learning_rate': 1e-4, 'lr_schedule': None, 'kl_threshold': 0.008,
        'gamma': 0.99, 'tau': 0.95, 'e_clip': 0.2, 'clip_value': False,
        'critic_coef': 1.0, 'entropy_coef': 0.0, 'truncate_grads': False,
        'grad_norm': 1.0, 'seq_length': 5 if rnn else 4,
        'max_epochs': 1, 'save_frequency': 0, 'save_best_after': 10_000,
        'print_stats': False, 'train_dir': '/tmp/pytest_ppo_masking_runs',
        'env_info': env_info,
    }
    for k, v in config_overrides.items():
        if v is None:
            config.pop(k, None)
        else:
            config[k] = v
    return {'algo': {'name': 'a2c_continuous'},
            'model': {'name': 'continuous_a2c_logstd'},
            'network': network,
            'config': config}


class _TorchEnvAdapter:
    """The fake envs speak numpy; A2C rollouts expect torch tensors."""

    def __init__(self, fake):
        self.fake = fake

    def reset(self):
        return torch.from_numpy(self.fake.reset())

    def step(self, actions):
        obs, rew, dones, infos = self.fake.step(actions.detach().cpu().numpy())
        return (torch.from_numpy(obs), torch.from_numpy(rew),
                torch.from_numpy(dones), infos)

    def get_env_info(self):
        return self.fake.get_env_info()

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass


def make_ppo_agent(env_cls=StaggeredFakeNextStepVecEnv, num_envs=NUM_ENVS,
                   rnn=False, seed=7, env_info=None, **config_overrides):
    from rl_games.torch_runner import Runner
    torch.manual_seed(seed)
    np.random.seed(seed)
    fake = env_cls(num_envs, EP_LEN)
    env = _TorchEnvAdapter(fake)
    params = _ppo_params(num_envs=num_envs,
                         env_info=env_info if env_info is not None else fake.get_env_info(),
                         rnn=rnn, **config_overrides)
    # top-level seed: without it Runner.load reseeds torch from time.time(),
    # so two agents built across a second boundary start from different weights
    params['seed'] = seed
    runner = Runner()
    runner.load({'params': params})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(runner.algo_name, base_name='test_ppo_masking',
                                       params=runner.params)
    return agent, fake


def _rollout_batch(agent):
    agent.init_tensors()
    agent.obs = agent.env_reset()
    with torch.no_grad():
        return agent.play_steps_rnn() if agent.is_rnn else agent.play_steps()


def _assert_mask_matches_obs_oracle(batch_dict, fake):
    # ground truth from observations alone: a garbage row's stored obs is the
    # previous episode's TERMINAL obs, i.e. step_in_episode == that env's ep_len
    obses = batch_dict['obses']
    masks = batch_dict['rnn_masks']
    env_ids = obses[:, 0].long()
    ep_len_of_row = torch.from_numpy(fake.ep_lens)[env_ids].float()
    expected_garbage = obses[:, 1] == ep_len_of_row
    assert masks.shape[0] == obses.shape[0]
    assert torch.equal(masks == 0.0, expected_garbage), (
        masks.tolist(), obses[:, 1].tolist())
    assert expected_garbage.any(), "fixture produced no garbage rows; test is vacuous"


def test_mask_flags_exactly_the_reset_rows():
    agent, fake = make_ppo_agent()
    batch = _rollout_batch(agent)
    assert 'rnn_masks' in batch
    _assert_mask_matches_obs_oracle(batch, fake)


def test_rnn_path_mask_matches_oracle():
    agent, fake = make_ppo_agent(rnn=True)
    assert agent.is_rnn
    batch = _rollout_batch(agent)
    assert 'rnn_masks' in batch
    _assert_mask_matches_obs_oracle(batch, fake)


def test_same_step_env_produces_no_mask_and_no_flag():
    agent, fake = make_ppo_agent(env_cls=SameStepFakeVecEnv)
    assert agent.mask_autoreset_rows is False
    batch = _rollout_batch(agent)
    assert 'rnn_masks' not in batch


def test_masked_rows_contribute_zero_gradient():
    # poison the garbage rows' returns/values: if masking works, one training
    # epoch from identical weights must produce identical parameters
    results = []
    initial = []
    for poison in (False, True):
        agent, fake = make_ppo_agent(seed=123)
        initial.append(copy.deepcopy(agent.model.state_dict()))
        batch = _rollout_batch(agent)
        if poison:
            garbage = batch['rnn_masks'] == 0.0
            batch['returns'] = batch['returns'].clone()
            batch['values'] = batch['values'].clone()
            batch['returns'][garbage] = 1e6
            batch['values'][garbage] = -1e6
        agent.prepare_dataset(batch)
        torch.manual_seed(0)  # identical permutation / any sampling noise
        for _ in range(agent.mini_epochs_num):
            for i in range(len(agent.dataset)):
                agent.train_actor_critic(agent.dataset[i])
        results.append(copy.deepcopy(agent.model.state_dict()))
    # same starting weights, or the comparison below measures the seed, not the mask
    for k in initial[0]:
        assert torch.equal(initial[0][k], initial[1][k]), (
            f"initial parameter {k} differs between the two agents: seed not pinned")
    clean, poisoned = results
    for k in clean:
        assert torch.allclose(clean[k], poisoned[k], atol=0, rtol=0), (
            f"parameter {k} differs: poisoned garbage rows leaked into the update")


def test_episode_stats_unpolluted():
    # uniform ep_len env: every logged episode must have length EXACTLY ep_len
    # (the old path counted the reset step too: ep_len + 1)
    agent, fake = make_ppo_agent(env_cls=FakeNextStepVecEnv)
    _rollout_batch(agent)
    assert agent.game_lengths.current_size > 0
    assert agent.game_lengths.get_mean() == pytest.approx(EP_LEN)
    # per-step reward is 1.0 on live rows, 0 filler on reset rows
    assert agent.game_rewards.get_mean() == pytest.approx(float(EP_LEN))


def test_multi_agent_next_step_guard_raises():
    fake = FakeNextStepVecEnv(NUM_ENVS, EP_LEN)
    env_info = fake.get_env_info()
    env_info['agents'] = 2
    with pytest.raises(ValueError, match="multi-agent"):
        make_ppo_agent(env_info=env_info)


class TestScalarSigmaParametrization:
    """sigma_parametrization: 'linear' (alias 'scalar') — head output IS the std (smooth floor)."""

    def test_scalar_aliases_to_linear(self):
        from rl_games.algos_torch import torch_ext  # noqa: F401  (import guard)
        import types
        from rl_games.algos_torch.models import apply_sigma_parametrization
        net_l = types.SimpleNamespace(sigma_parametrization='linear', min_sigma=0.05)
        net_s = types.SimpleNamespace(sigma_parametrization='scalar', min_sigma=0.05)
        raw = torch.linspace(-2.0, 3.0, 11)
        s_l, _ = apply_sigma_parametrization(raw, net_l)
        s_s, _ = apply_sigma_parametrization(raw, net_s)
        assert torch.equal(s_l, s_s)

    def _sigmas(self, min_sigma, init_val):
        from rl_games.algos_torch.model_builder import ModelBuilder
        params = {
            'algo': {'name': 'a2c_continuous'},
            'model': {'name': 'continuous_a2c_logstd'},
            'network': {'name': 'actor_critic', 'separate': False,
                'space': {'continuous': {
                    'mu_activation': 'None', 'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': init_val},
                    'fixed_sigma': False, 'min_sigma': min_sigma,
                    'sigma_parametrization': 'scalar'}},
                'mlp': {'units': [16], 'activation': 'elu',
                        'initializer': {'name': 'default'}}}}
        model = ModelBuilder().load(params).build(
            {'actions_num': 3, 'input_shape': (6,), 'num_seqs': 1, 'value_size': 1,
             'normalize_value': False, 'normalize_input': False})
        out = model({'is_train': False, 'obs': torch.randn(64, 6) * 3})
        return out['sigmas']

    def test_uniform_init_through_smooth_floor(self):
        import torch.nn.functional as F
        s = self._sigmas(min_sigma=0.05, init_val=1.0)
        expected = 0.05 + F.softplus(torch.tensor(1.0 - 0.05))
        assert torch.allclose(s, expected.expand_as(s), rtol=1e-4)

    def test_floor_is_smooth_positive_and_finite(self):
        s = self._sigmas(min_sigma=0.1, init_val=-5.0)
        assert (s > 0.1).all() and (s < 0.12).all()
        assert torch.isfinite(s).all()

    def test_gradient_alive_below_floor(self):
        # the smooth floor must keep a restoring gradient where a hard clamp dies
        raw = torch.tensor([-5.0, 0.05, 1.0], requires_grad=True)
        floor = 0.1
        sigma = floor + torch.nn.functional.softplus(raw - floor)
        sigma.sum().backward()
        assert (raw.grad > 0).all(), "zero gradient below the floor: dead zone is back"


class TestScheduleTypeAlias:

    def test_legacy_aliases_to_per_minibatch(self):
        agent, _ = make_ppo_agent(schedule_type='legacy')
        assert agent.schedule_type == 'per_minibatch'

    def test_default_is_per_minibatch(self):
        agent, _ = make_ppo_agent()
        assert agent.schedule_type == 'per_minibatch'

    def test_standard_untouched(self):
        agent, _ = make_ppo_agent(schedule_type='standard')
        assert agent.schedule_type == 'standard'

    def test_unknown_value_rejected(self):
        # train_epoch branches on the exact value; an unknown one used to
        # silently skip every scheduler step (standard_epoch was documented
        # without an implementation)
        with pytest.raises(ValueError, match="schedule_type"):
            make_ppo_agent(schedule_type='standard_epoch')


class TestYamlSchedulerBounds:
    # YAML 1.1 reads a bare exponent without a dot as a string: the agent
    # must cast every scheduler bound, or the first out-of-band KL raises
    # TypeError inside the scheduler's max()/min()

    def test_adaptive_bounds_from_yaml_exponent_strings(self):
        import yaml
        raw = yaml.safe_load('min_lr: 1e-5\nmax_lr: 1e-3\nkl_threshold: 8e-3\n'
                             'lr_multiplier: 15e-1')
        assert all(isinstance(v, str) for v in raw.values()), raw
        agent, _ = make_ppo_agent(lr_schedule='adaptive', **raw)
        assert agent.kl_threshold == pytest.approx(8e-3)
        # kl above the band: halve towards min_lr; below: raise up to max_lr
        lr, _ = agent.scheduler.update(1e-4, 0.0, 0, 0, kl_dist=0.05)
        assert lr == pytest.approx(1e-4 / 1.5)
        lr, _ = agent.scheduler.update(9e-4, 0.0, 0, 0, kl_dist=1e-4)
        assert lr == pytest.approx(1e-3)
        lr, _ = agent.scheduler.update(1.2e-5, 0.0, 0, 0, kl_dist=0.05)
        assert lr == pytest.approx(1e-5)

    def test_linear_min_lr_from_yaml_exponent_string(self):
        import yaml
        raw = yaml.safe_load('min_lr: 1e-5')
        assert isinstance(raw['min_lr'], str)
        agent, _ = make_ppo_agent(lr_schedule='linear', max_epochs=10, **raw)
        lr, _ = agent.scheduler.update(1e-4, 0.0, 10, 0, kl_dist=0.0)
        assert lr == pytest.approx(1e-5)


def test_running_stats_moment_merge_math():
    # the cross-rank merge must equal stats computed on the concatenated data
    torch.manual_seed(3)
    a = torch.randn(500, 4) * 2 + 1
    b = torch.randn(300, 4) * 5 - 2
    n1, n2 = float(len(a)), float(len(b))
    m1, v1 = a.mean(0), a.var(0, unbiased=False)
    m2, v2 = b.mean(0), b.var(0, unbiased=False)
    n = n1 + n2
    mean = (m1 * n1 + m2 * n2) / n
    var = ((v1 + m1 ** 2) * n1 + (v2 + m2 ** 2) * n2) / n - mean ** 2
    full = torch.cat([a, b])
    assert torch.allclose(mean, full.mean(0), atol=1e-5)
    assert torch.allclose(var, full.var(0, unbiased=False), atol=1e-4)


def test_running_stats_delta_merge_iterated():
    # simulate 2 ranks x 3 epochs: delta-based merge must equal stats over
    # ALL data seen, with no double-weighting of shared history
    torch.manual_seed(9)
    dim = 3
    all_data = []
    # global state each rank holds after merge: (count, wmean, wsq)
    base = [torch.zeros(1), torch.zeros(dim), torch.zeros(dim)]
    for epoch in range(3):
        deltas = []
        for rank in range(2):
            x = torch.randn(200 + 50 * rank, dim) * (rank + 1) + epoch
            all_data.append(x)
            n = torch.tensor([float(len(x))])
            deltas.append([n, x.sum(0), (x ** 2).sum(0)])
        summed = [d0 + d1 for d0, d1 in zip(*deltas)]
        base = [b + s for b, s in zip(base, summed)]
        mean = base[1] / base[0]
        var = base[2] / base[0] - mean ** 2
    full = torch.cat(all_data)
    assert torch.allclose(mean, full.mean(0), atol=1e-4)
    assert torch.allclose(var, full.var(0, unbiased=False), atol=1e-3)
    assert abs(base[0].item() - len(full)) < 1e-3   # counts exact, no doubling


def _allreduce_two_identical_ranks(t):
    # SUM all-reduce where both ranks hold identical tensors
    # (int multiplier: count tensors are integer dtype, like real NCCL SUM)
    t.mul_(2)


def test_merge_rank_stats_fresh_start_sums_local_histories():
    from rl_games.common.a2c_common import merge_rank_stats
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    torch.manual_seed(5)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(400, 3) * 2 + 1)
    c0, mean0, var0 = m.count.clone(), m.running_mean.clone(), m.running_var.clone()
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    # fresh start: each rank's history is genuinely local data -> counts add
    assert torch.allclose(m.count.float(), 2 * c0.float(), rtol=1e-6)
    assert torch.allclose(m.running_mean, mean0, atol=1e-5)
    assert torch.allclose(m.running_var, var0, atol=1e-4)


def test_seeded_snapshot_prevents_resume_count_inflation():
    from rl_games.common.a2c_common import merge_rank_stats, seed_stats_sync_snapshot
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    torch.manual_seed(6)
    m = RunningMeanStd((3,))
    m.train()
    m(torch.randn(400, 3) + 2)
    # emulate checkpoint restore: every rank loads IDENTICAL stats (shared
    # history) -- the restore path seeds the snapshot so the first sync
    # must not re-sum that history across ranks
    seed_stats_sync_snapshot(m)
    c0, mean0 = m.count.clone(), m.running_mean.clone()
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    assert torch.allclose(m.count.float(), c0.float(), rtol=1e-6)  # NOT x world_size
    assert torch.allclose(m.running_mean, mean0, atol=1e-6)
    # data collected AFTER the resume is fresh per-rank: only the delta sums
    m(torch.randn(100, 3) - 1)
    merge_rank_stats(m, _allreduce_two_identical_ranks)
    assert torch.allclose(m.count.float(), c0.float() + 200.0, rtol=1e-5)


def test_apply_masks_divides_by_valid_count():
    from rl_games.algos_torch import torch_ext
    losses = [torch.ones(8, 1)]
    mask = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.])
    res, sum_mask = torch_ext.apply_masks(losses, mask)
    assert float(sum_mask) == 4.0
    # mean over valid rows; the old numel() divisor returned 0.5 here,
    # silently scaling every loss (and the KL fed to adaptive LR) by the
    # valid fraction
    assert torch.allclose(res[0], torch.tensor(1.0))


def test_env_reset_clears_autoreset_tracker():
    # a fresh env_reset must invalidate stale prev-dones: the first row after
    # an explicit reset is a real step and must not be masked
    agent, fake = make_ppo_agent(seed=11)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    with torch.no_grad():
        agent.play_steps()
    assert agent._autoreset_prev_dones is not None  # tracker engaged mid-rollout
    agent.obs = agent.env_reset()                   # re-reset (eval interleave etc.)
    assert agent._autoreset_prev_dones is None
    with torch.no_grad():
        batch = agent.play_steps()
    first_row_masks = batch['rnn_masks'].view(agent.num_actors, agent.horizon_length)[:, 0]
    assert torch.all(first_row_masks == 1.0), "first row after explicit reset wrongly masked"


def test_rnn_state_isolated_from_garbage_obs():
    # the complete RNN invariant: poison the OBS of garbage rows — with the
    # rollout re-zero + the shifted rnn-dones channel, neither the losses nor
    # the recurrent state may transport the poison into any update
    # (normalize_input off: the input normalizer is a separate, unmasked
    # channel — garbage rows feed it real terminal observations, see PR notes)
    results = []
    for poison in (False, True):
        agent, fake = make_ppo_agent(seed=321, rnn=True, normalize_input=False,
                                     normalize_value=False)
        batch = _rollout_batch(agent)
        if poison:
            garbage = batch['rnn_masks'] == 0.0
            batch['obses'] = batch['obses'].clone()
            batch['obses'][garbage] = 1e6
        agent.prepare_dataset(batch)
        torch.manual_seed(0)
        for _ in range(agent.mini_epochs_num):
            for i in range(len(agent.dataset)):
                agent.train_actor_critic(agent.dataset[i])
        results.append(copy.deepcopy(agent.model.state_dict()))
    clean, poisoned = results
    for k in clean:
        assert torch.allclose(clean[k], poisoned[k], atol=0, rtol=0), (
            f"parameter {k} differs: garbage obs leaked through the RNN state")


def test_value_normalizer_ignores_garbage_returns():
    # normalize_value=True (the default): garbage-row returns must not move
    # the value normalizer's statistics, else they rescale live-row critic
    # targets and reach the weights through that side channel
    results = []
    for poison in (False, True):
        agent, fake = make_ppo_agent(seed=99, normalize_value=True)
        batch = _rollout_batch(agent)
        if poison:
            garbage = batch['rnn_masks'] == 0.0
            batch['returns'] = batch['returns'].clone()
            batch['returns'][garbage] = 1e6
        agent.prepare_dataset(batch)
        torch.manual_seed(0)
        for _ in range(agent.mini_epochs_num):
            for i in range(len(agent.dataset)):
                agent.train_actor_critic(agent.dataset[i])
        results.append((copy.deepcopy(agent.model.state_dict()),
                        agent.value_mean_std.running_mean.clone()))
    (clean, clean_rm), (poisoned, poisoned_rm) = results
    assert torch.allclose(clean_rm, poisoned_rm, atol=0, rtol=0), (
        "value normalizer statistics moved by garbage returns")
    for k in clean:
        assert torch.allclose(clean[k], poisoned[k], atol=0, rtol=0), (
            f"parameter {k} differs: garbage returns leaked via the value normalizer")


def test_recurrent_central_value_state_isolated_from_filler_rows(tmp_path):
    """Recurrent central critic must get the same post-filler re-zero as the
    actor: its get_value in get_action_values advances rnn_states on the
    filler observation, and without the mirrored reset the first real row of
    every new episode inherits filler-contaminated critic state (external
    review 2026-08-08, confirmed by repro: 6/6 boundaries contaminated)."""
    import numpy as np
    import torch
    from gymnasium import spaces
    from rl_games.torch_runner import Runner
    from tests.test_sac_correctness import (StaggeredFakeNextStepVecEnv,
                                            EP_LEN, OBS_DIM)

    NUM_ENVS = 4
    HORIZON = 3 * EP_LEN

    class DictObsNextStepEnv(StaggeredFakeNextStepVecEnv):
        def _wrap(self, o):
            return {'obs': o, 'states': (o * 10.0 + 1.0).astype(np.float32)}

        def reset(self):
            return self._wrap(super().reset())

        def step(self, actions):
            o, r, d, i = super().step(actions)
            return self._wrap(o), r, d, i

        def get_env_info(self):
            info = super().get_env_info()
            info['state_space'] = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
            return info

    class TorchDictEnvAdapter:
        def __init__(self, fake):
            self.fake = fake

        def _t(self, obs):
            return {k: torch.from_numpy(v) for k, v in obs.items()}

        def reset(self):
            return self._t(self.fake.reset())

        def step(self, actions):
            obs, rew, dones, infos = self.fake.step(actions.detach().cpu().numpy())
            return self._t(obs), torch.from_numpy(rew), torch.from_numpy(dones), infos

        def get_env_info(self):
            return self.fake.get_env_info()

        def get_env_state(self):
            return None

        def set_env_state(self, state):
            pass

    torch.manual_seed(7)
    np.random.seed(7)
    fake = DictObsNextStepEnv(NUM_ENVS, EP_LEN)
    env = TorchDictEnvAdapter(fake)
    rnn = {'name': 'lstm', 'units': 8, 'layers': 1}
    network = {
        'name': 'actor_critic', 'separate': False,
        'space': {'continuous': {
            'mu_activation': 'None', 'sigma_activation': 'None',
            'mu_init': {'name': 'default'},
            'sigma_init': {'name': 'const_initializer', 'val': 0.0},
            'fixed_sigma': True}},
        'mlp': {'units': [16], 'activation': 'elu', 'initializer': {'name': 'default'}},
        'rnn': rnn,
    }
    config = {
        'name': 'cv_rnn_filler', 'env_name': 'unused',
        'reward_shaper': {'scale_value': 1.0},
        'device': 'cpu', 'multi_gpu': False, 'mixed_precision': False,
        'normalize_input': False, 'normalize_value': False,
        'normalize_advantage': True, 'value_bootstrap': False,
        'num_actors': NUM_ENVS, 'horizon_length': HORIZON,
        'minibatch_size': NUM_ENVS * HORIZON, 'mini_epochs': 1,
        'learning_rate': 1e-4, 'lr_schedule': None, 'kl_threshold': 0.008,
        'gamma': 0.99, 'tau': 0.95, 'e_clip': 0.2, 'clip_value': False,
        'critic_coef': 1.0, 'entropy_coef': 0.0, 'truncate_grads': False,
        'grad_norm': 1.0, 'seq_length': 5,
        'max_epochs': 1, 'save_frequency': 0, 'save_best_after': 10_000,
        'print_stats': False, 'train_dir': str(tmp_path),
        'env_info': fake.get_env_info(),
        'central_value_config': {
            'minibatch_size': NUM_ENVS * HORIZON, 'mini_epochs': 1,
            'learning_rate': 1e-4, 'clip_value': False,
            'normalize_input': False, 'truncate_grads': False,
            'network': {'name': 'actor_critic', 'central_value': True,
                        'mlp': {'units': [16], 'activation': 'elu',
                                'initializer': {'name': 'default'}},
                        'rnn': rnn},
        },
    }
    runner = Runner()
    runner.load({'params': {'algo': {'name': 'a2c_continuous'},
                            'model': {'name': 'continuous_a2c_logstd'},
                            'network': network, 'config': config}})
    runner.params['config']['vec_env'] = env
    agent = runner.algo_factory.create(runner.algo_name, base_name='cv_rnn_filler',
                                       params=runner.params)
    assert agent.is_rnn and agent.has_central_value
    assert agent.central_value_net.is_rnn and agent.mask_autoreset_rows

    # spy on states ENTERING each row's forward
    log = []
    orig_gav = agent.get_action_values

    def spy_gav(obs):
        prev = agent._autoreset_prev_dones
        prev = prev.clone() if prev is not None else torch.zeros(NUM_ENVS)
        amag = torch.stack([s.abs().amax(dim=(0, 2)) for s in agent.rnn_states]).amax(0)
        cmag = torch.stack([s.abs().amax(dim=(0, 2))
                            for s in agent.central_value_net.rnn_states]).amax(0)
        log.append((prev, amag, cmag))
        return orig_gav(obs)

    agent.get_action_values = spy_gav
    agent.init_tensors()
    agent.obs = agent.env_reset()
    with torch.no_grad():
        agent.play_steps_rnn()

    checked = 0
    for n in range(len(log) - 1):
        prev, _, _ = log[n]
        _, amag2, cmag2 = log[n + 1]
        for e in range(NUM_ENVS):
            if prev[e] > 0:      # row n was a filler row for env e
                checked += 1
                assert amag2[e].item() == 0.0, f'actor state dirty at env {e} row {n+1}'
                assert cmag2[e].item() == 0.0, f'CV state dirty at env {e} row {n+1}'
    assert checked >= 4, f'staggered env produced too few boundaries ({checked})'


def test_masked_moments_finite_on_degenerate_masks():
    """0 or 1 valid rows must yield finite moments, not NaN (follow-up review
    2026-08-10; unreachable from autoreset masking, defensive for degenerate
    geometries)."""
    import torch
    from rl_games.algos_torch.torch_ext import get_mean_var_with_masks
    v = torch.randn(8)
    for n_valid in (0, 1, 2):
        m = torch.zeros(8)
        m[:n_valid] = 1.0
        mean, var = get_mean_var_with_masks(v, m)
        assert torch.isfinite(mean) and torch.isfinite(var), (n_valid, mean, var)
    # sanity: >=2 valid rows still produce the unbiased sample variance
    m = torch.ones(8)
    mean, var = get_mean_var_with_masks(v, m)
    assert torch.allclose(mean, v.mean(), atol=1e-6)
    assert torch.allclose(var, v.var(unbiased=True), atol=1e-5)
