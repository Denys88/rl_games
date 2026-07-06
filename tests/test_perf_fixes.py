"""Tests for the hot-loop performance batch: replay/experience buffer pinning
removal, disabled GradScaler under bf16 autocast, SAC submodule compilation,
and batched score-observer updates."""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.common.experience import VectorizedReplayBuffer


class TestReplayBufferNoPinning:

    def test_cpu_buffer_not_pinned_and_roundtrips(self):
        buf = VectorizedReplayBuffer((4,), (2,), capacity=16, device='cpu')
        assert not buf.obses.is_pinned()
        assert not buf.next_obses.is_pinned()
        obs = torch.randn(3, 4)
        act = torch.randn(3, 2)
        rew = torch.randn(3, 1)
        done = torch.zeros(3, 1, dtype=torch.bool)
        buf.add(obs, act, rew, obs + 1, done)
        s_obs, s_act, s_rew, s_next, s_done, s_trunc = buf.sample(2)
        assert s_obs.shape == (2, 4) and s_trunc.shape == (2, 1)


class TestScalerDisabledForBf16:

    def test_disabled_scaler_full_step_chain(self):
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        loss = model(torch.randn(8, 4)).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        assert scaler.state_dict() == {}

    def test_a2c_scaler_construction_disabled(self):
        # the constructor arg is what a2c_common now uses regardless of mixed_precision
        import inspect
        from rl_games.common import a2c_common
        src = inspect.getsource(a2c_common.A2CBase.__init__)
        assert "GradScaler('cuda', enabled=False)" in src

    def test_central_value_autocast_is_bf16(self):
        import inspect
        from rl_games.algos_torch import central_value
        src = inspect.getsource(central_value)
        assert src.count('dtype=torch.bfloat16') == 2
        assert 'GradScaler(enabled=False)' in src


class TestSacCompileInPlace:

    def test_module_compile_keeps_state_dict_keys_and_params(self):
        # in-place nn.Module.compile must not introduce _orig_mod prefixes or
        # replace parameter objects (optimizers hold references to them)
        actor = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
        params_before = [id(p) for p in actor.parameters()]
        keys_before = list(actor.state_dict().keys())
        actor.compile(mode='default')
        assert list(actor.state_dict().keys()) == keys_before
        assert [id(p) for p in actor.parameters()] == params_before


class FakeMeterAlgo:
    def __init__(self, num_agents=1):
        self.num_agents = num_agents
        self.games_to_track = 100
        self.ppo_device = 'cpu'
        self.writer = None


def make_observer(num_agents=1):
    obs = DefaultAlgoObserver()
    obs.after_init(FakeMeterAlgo(num_agents))
    return obs


class TestBatchedScoreObserver:

    def test_dict_infos_batched_mean_matches(self):
        obs = make_observer()
        infos = {'scores': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        obs.process_infos(infos, torch.tensor([0, 2, 3]))
        assert obs.game_scores.current_size == 3
        assert obs.game_scores.get_mean() == pytest.approx((1.0 + 3.0 + 4.0) / 3)

    def test_nan_fillers_skipped(self):
        obs = make_observer()
        infos = {'scores': np.array([1.0, np.nan, 5.0], dtype=np.float32)}
        obs.process_infos(infos, torch.tensor([0, 1, 2]))
        assert obs.game_scores.current_size == 2
        assert obs.game_scores.get_mean() == pytest.approx(3.0)

    def test_all_nan_no_update(self):
        obs = make_observer()
        infos = {'scores': np.array([np.nan, np.nan], dtype=np.float32)}
        obs.process_infos(infos, torch.tensor([0, 1]))
        assert obs.game_scores.current_size == 0

    def test_battle_won_bool_entries(self):
        obs = make_observer()
        infos = {'battle_won': np.array([True, False, True])}
        obs.process_infos(infos, torch.tensor([0, 1, 2]))
        assert obs.game_scores.get_mean() == pytest.approx(2.0 / 3.0)

    def test_lives_filter_envpool_path(self):
        obs = make_observer()
        infos = {'scores': np.array([10.0, 20.0, 30.0], dtype=np.float32),
                 'lives': np.array([0, 1, 0])}
        obs.process_infos(infos, torch.tensor([1]))  # overridden by lives==0 -> envs 0,2
        assert obs.game_scores.current_size == 2
        assert obs.game_scores.get_mean() == pytest.approx(20.0)

    def test_num_agents_division(self):
        obs = make_observer(num_agents=2)
        infos = {'scores': np.array([7.0], dtype=np.float32)}
        obs.process_infos(infos, torch.tensor([0, 1]))  # both agents of env 0
        assert obs.game_scores.current_size == 2
        assert obs.game_scores.get_mean() == pytest.approx(7.0)

    def test_out_of_range_indices_dropped(self):
        obs = make_observer()
        infos = {'scores': np.array([1.0], dtype=np.float32)}
        obs.process_infos(infos, torch.tensor([0, 5]))
        assert obs.game_scores.current_size == 1

    def test_no_scores_key_no_crash(self):
        obs = make_observer()
        obs.process_infos({'lives': np.array([0, 0])}, torch.tensor([0, 1]))
        assert obs.game_scores.current_size == 0

    def test_list_of_dicts_path_unchanged(self):
        obs = make_observer()
        infos = [{'scores': 3.0}, {'scores': 5.0}]
        obs.process_infos(infos, torch.tensor([0, 1]))
        assert obs.game_scores.current_size == 2
        assert obs.game_scores.get_mean() == pytest.approx(4.0)


class TestMinSigmaFloor:

    def _build_model(self, min_sigma):
        from rl_games.algos_torch.model_builder import ModelBuilder
        params = {
            'algo': {'name': 'a2c_continuous'},
            'model': {'name': 'continuous_a2c_logstd'},
            'network': {
                'name': 'actor_critic', 'separate': False,
                'space': {'continuous': {
                    'mu_activation': 'None', 'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': -3.0},
                    'fixed_sigma': False, 'min_sigma': min_sigma}},
                'mlp': {'units': [16], 'activation': 'elu',
                        'initializer': {'name': 'default'}}},
        }
        builder = ModelBuilder()
        network = builder.load(params)
        return network.build({'actions_num': 2, 'input_shape': (4,),
                              'num_seqs': 1, 'value_size': 1,
                              'normalize_value': False, 'normalize_input': False})

    def test_sigma_floored_and_logprob_consistent(self):
        model = self._build_model(min_sigma=0.2)
        obs = torch.randn(8, 4)
        out = model({'is_train': False, 'obs': obs})
        assert (out['sigmas'] >= 0.2).all()
        # log-prob consistency: neglogp computed from floored sigma matches Normal
        d = torch.distributions.Normal(out['mus'], out['sigmas'])
        expected = -d.log_prob(out['actions']).sum(dim=-1)
        assert torch.allclose(out['neglogpacs'], expected, atol=1e-5)

    def test_no_floor_by_default(self):
        model = self._build_model(min_sigma=0.0)
        obs = torch.randn(8, 4)
        out = model({'is_train': False, 'obs': obs})
        # sigma_init -3 => exp(-3) ~ 0.0498 < 0.2: without floor small sigmas survive
        assert (out['sigmas'] < 0.2).any()

    def test_softplus_parametrization(self):
        from rl_games.algos_torch.model_builder import ModelBuilder
        params = {
            'algo': {'name': 'a2c_continuous'},
            'model': {'name': 'continuous_a2c_logstd'},
            'network': {
                'name': 'actor_critic', 'separate': False,
                'space': {'continuous': {
                    'mu_activation': 'None', 'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': -1.05},
                    'fixed_sigma': False, 'min_sigma': 0.2,
                    'sigma_parametrization': 'softplus'}},
                'mlp': {'units': [16], 'activation': 'elu',
                        'initializer': {'name': 'default'}}},
        }
        network = ModelBuilder().load(params)
        model = network.build({'actions_num': 2, 'input_shape': (4,), 'num_seqs': 1,
                               'value_size': 1, 'normalize_value': False, 'normalize_input': False})
        out = model({'is_train': False, 'obs': torch.randn(8, 4)})
        assert (out['sigmas'] >= 0.2).all()
        d = torch.distributions.Normal(out['mus'], out['sigmas'])
        expected = -d.log_prob(out['actions']).sum(dim=-1)
        assert torch.allclose(out['neglogpacs'], expected, atol=1e-5)


class TestSchedulerTunables:

    def test_adaptive_custom_params(self):
        from rl_games.common.schedulers import AdaptiveScheduler
        s = AdaptiveScheduler(kl_threshold=0.01, min_lr=5e-4, max_lr=5e-3, lr_multiplier=2.0)
        lr, _ = s.update(1e-3, 0.0, 0, 0, kl_dist=0.021)  # kl > 2*0.01 -> /2
        assert lr == pytest.approx(5e-4)
        lr, _ = s.update(1e-3, 0.0, 0, 0, kl_dist=0.004)  # kl < 0.5*0.01 -> *2
        assert lr == pytest.approx(2e-3)
        lr, _ = s.update(6e-4, 0.0, 0, 0, kl_dist=0.03)   # floor respected
        assert lr == pytest.approx(5e-4)

    def test_adaptive_defaults_unchanged(self):
        from rl_games.common.schedulers import AdaptiveScheduler
        s = AdaptiveScheduler(kl_threshold=0.008)
        lr, _ = s.update(1e-3, 0.0, 0, 0, kl_dist=0.02)
        assert lr == pytest.approx(1e-3 / 1.5)
        lr, _ = s.update(1e-3, 0.0, 0, 0, kl_dist=0.003)
        assert lr == pytest.approx(1.5e-3)


class TestConfigDrivenEnvRegistration:

    def test_vecenv_type_and_observer_from_config(self):
        from rl_games.torch_runner import Runner
        from rl_games.common import env_configurations
        from rl_games.common.algo_observer import IsaacAlgoObserver
        r = Runner()
        r.load_config({'params': {}} and {
            'algo': {'name': 'a2c_continuous'},
            'config': {'env_name': 'pytest_fake_backend', 'vecenv_type': 'RAY',
                       'algo_observer': 'isaac', 'reward_shaper': {}}})
        assert 'pytest_fake_backend' in env_configurations.configurations
        assert env_configurations.configurations['pytest_fake_backend']['vecenv_type'] == 'RAY'
        assert isinstance(r.algo_observer, IsaacAlgoObserver)

    def test_injected_observer_wins(self):
        from rl_games.torch_runner import Runner
        from rl_games.common.algo_observer import DefaultAlgoObserver
        obs = DefaultAlgoObserver()
        r = Runner(algo_observer=obs)
        r.load_config({'algo': {'name': 'a2c_continuous'},
                       'config': {'algo_observer': 'isaac', 'reward_shaper': {}}})
        assert r.algo_observer is obs


class TestClassicReorientTask:

    def test_registers_with_classic_semantics(self):
        pytest.importorskip('wuji_mjlab')
        import rl_games.envs.mjlab_tasks.wuji_reorient_classic  # noqa: F401
        from mjlab.tasks.registry import list_tasks, load_env_cfg
        assert 'WujiHand_Reorient_Classic' in list_tasks()
        cfg = load_env_cfg('WujiHand_Reorient_Classic')
        assert set(cfg.rewards) == {'rot_reward', 'reach_goal_bonus', 'dist_penalty', 'action_penalty'}
        cmd = cfg.commands['reorient_command']
        assert (cmd.success_threshold, cmd.success_hold_steps, cmd.goal_switch_delay) == (0.2, 1, 0)
