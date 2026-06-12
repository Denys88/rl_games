"""Phase B SAC correctness tests (spec: docs/superpowers/specs/2026-06-11-phase-b-sac-correctness-design.md)."""
import functools
import os
import sys

import numpy as np
import pytest
import torch
import yaml

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from test_critical_fixes import _load_params, SAC_YAML, TEST_TRAIN_DIR


OBS_DIM = 4
ACT_DIM = 2
EP_LEN = 5


class FakeNextStepVecEnv:
    """Deterministic NEXT_STEP-autoreset vec env (envpool semantics).

    obs = [env_id, step_in_episode, episode_counter, 0]. At t == ep_len the env
    reports done (truncated by default) and returns the TRUE final obs; the NEXT
    step ignores the action and returns the post-reset obs (t=0) with reward 0,
    done=False — the 'garbage row' the agent must skip.
    """

    def __init__(self, num_envs, ep_len=EP_LEN, terminate_instead=False):
        self.num_envs = num_envs
        self.ep_len = ep_len
        self.terminate_instead = terminate_instead
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.episode = np.zeros(num_envs, dtype=np.int64)
        self.pending_reset = np.zeros(num_envs, dtype=bool)

    def _obs(self):
        o = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        o[:, 0] = np.arange(self.num_envs)
        o[:, 1] = self.t
        o[:, 2] = self.episode
        return o

    def reset(self):
        self.t[:] = 0
        self.episode[:] = 0
        self.pending_reset[:] = False
        return self._obs()

    def step(self, actions):
        rewards = np.ones(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for e in range(self.num_envs):
            if self.pending_reset[e]:
                self.t[e] = 0
                self.episode[e] += 1
                self.pending_reset[e] = False
                rewards[e] = 0.0
            else:
                self.t[e] += 1
                if self.t[e] >= self.ep_len:
                    dones[e] = True
                    truncated[e] = not self.terminate_instead
                    self.pending_reset[e] = True
        infos = {'time_outs': truncated.copy()}
        return self._obs(), rewards, dones.astype(np.float32), infos

    def get_env_info(self):
        return {
            'observation_space': spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32),
            'action_space': spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32),
            'agents': 1,
            'autoreset_mode': 'next_step',
        }

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass


def make_fake_env_sac_agent(num_envs=3, ep_len=EP_LEN, env_cls=FakeNextStepVecEnv, **config_overrides):
    from rl_games.torch_runner import Runner
    fake = env_cls(num_envs, ep_len)
    params = _load_params(SAC_YAML)
    cfg = params['config']
    cfg.pop('env_config', None)
    cfg.update({
        'device': 'cpu', 'multi_gpu': False, 'num_actors': num_envs,
        'print_stats': False, 'max_epochs': 2, 'save_frequency': 0,
        'save_best_after': 10_000, 'num_warmup_frames': 1,
        # one full episode per epoch: until finding 1.6 lands, train() can only
        # exit at max_epochs after at least one episode has completed
        'num_steps_per_episode': EP_LEN,
        'replay_buffer_size': 512, 'batch_size': 16,
        'train_dir': TEST_TRAIN_DIR, 'name': 'pytest_sac_fake',
        'env_info': fake.get_env_info(),
        'normalize_input': False,
    })
    for k, v in config_overrides.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    runner = Runner()
    runner.load({'params': params})
    # vec_env injected AFTER load: Runner.load deep-copies params and a live env
    # object must not be deep-copied
    runner.params['config']['vec_env'] = fake
    agent = runner.algo_factory.create(runner.algo_name, base_name='test_fake_sac', params=runner.params)
    return agent, fake


# reward_shaper.scale_value read straight from the helper's base yaml
# (sac_halfcheetah.yaml: 1.0) so the accounting test tracks the config the
# agent actually runs with rather than a hardcoded copy.
SHAPER_SCALE = float(_load_params(SAC_YAML)['config']['reward_shaper'].get('scale_value', 1.0))


def _completed_training_buffer(agent):
    agent.train()
    buf = agent.replay_buffer
    n = buf.capacity if buf.full else buf.idx
    return (buf.obses[:n], buf.actions[:n], buf.rewards[:n], buf.next_obses[:n],
            buf.dones[:n], buf.truncated[:n])


def test_no_cross_episode_rows_in_replay():
    # Finding 1.1: reset-step garbage rows must never be stored.
    agent, fake = make_fake_env_sac_agent(
        num_warmup_frames=None, num_warmup_steps=10_000,  # exploration still stores rows
        max_epochs=1, num_steps_per_episode=EP_LEN + 3)
    obses, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    assert obses.shape[0] > 0
    # a garbage row would be: obs at t==EP_LEN (terminal) -> next_obs at t==0
    bad = (obses[:, 1] == EP_LEN).nonzero(as_tuple=True)[0]
    assert len(bad) == 0, f"stored {len(bad)} transitions FROM the terminal obs (reset-step garbage)"
    # within-episode continuity
    assert torch.all(next_obses[:, 1] == obses[:, 1] + 1)
    # and same-episode pairing (episode counter must match across the row)
    assert torch.all(next_obses[:, 2] == obses[:, 2])


def test_truncation_stores_true_final_obs_and_flags():
    # Finding 1.4: at truncation, next_obs must be the TRUE final obs (t==EP_LEN),
    # done=False (bootstrap), truncated=True.
    agent, fake = make_fake_env_sac_agent(
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=EP_LEN + 3)
    _, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    rows = (next_obses[:, 1] == EP_LEN).nonzero(as_tuple=True)[0]
    assert len(rows) > 0, "no truncation-step rows captured"
    assert not dones[rows].any().item(), "truncated rows must bootstrap (done=False)"
    assert trunc[rows].all().item(), "truncated rows must carry truncated=True"


def test_termination_rows_do_not_bootstrap():
    agent, fake = make_fake_env_sac_agent(
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=EP_LEN + 3,
        env_cls=functools.partial(FakeNextStepVecEnv, terminate_instead=True))
    _, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    rows = (next_obses[:, 1] == EP_LEN).nonzero(as_tuple=True)[0]
    assert len(rows) > 0
    assert dones[rows].all().item(), "terminated rows must store done=True"
    assert not trunc[rows].any().item()


def test_reward_and_length_accounting_skips_reset_rows():
    agent, fake = make_fake_env_sac_agent(
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=2 * EP_LEN + 2)
    agent.train()
    assert agent.game_rewards.current_size > 0
    # per completed episode: EP_LEN steps of reward 1.0 (x shaper scale)
    expected = EP_LEN * 1.0 * SHAPER_SCALE
    assert agent.game_rewards.get_mean() == pytest.approx(expected, rel=1e-3)
    assert agent.game_lengths.get_mean() == pytest.approx(EP_LEN, rel=1e-3)


def test_vec_env_injection_via_env_info():
    # Finding 1.13: env_info-supplied path left self.vec_env unset.
    agent, fake = make_fake_env_sac_agent()
    assert agent.vec_env is fake
    obs = agent.env_reset()
    if isinstance(obs, dict):  # obs_to_tensors wraps plain obs into {'obs': ...}
        obs = obs['obs']
    assert obs.shape == (3, OBS_DIM)


def test_warmup_epoch_count_exact():
    # Finding 1.14: epoch_num increments BEFORE train_epoch, so `<` gives
    # num_warmup_steps - 1 warmup epochs; num_warmup_steps=1 disables warmup.
    agent, fake = make_fake_env_sac_agent(num_warmup_frames=None, num_warmup_steps=1, max_epochs=1)
    seen = []
    orig = agent.play_steps
    agent.play_steps = lambda random_exploration=False: (seen.append(random_exploration), orig(random_exploration))[1]
    agent.train()
    assert seen == [True], f"num_warmup_steps=1 must give exactly one warmup epoch, got {seen}"


def test_gamma_tensor_stays_fp32_under_mixed_precision():
    # Finding 1.9: gamma_tensor recast to bf16 turns 0.99 into 0.98828125.
    agent, _ = make_fake_env_sac_agent(mixed_precision=True)
    assert agent.gamma_tensor.dtype == torch.float32
    # abs=1e-7: fp32 representation of 0.99 is off by ~9.5e-9; the bf16 bug was off by 1.7e-3
    assert agent.gamma_tensor.item() == pytest.approx(0.99, abs=1e-7)


def test_actor_stats_skip_non_update_steps():
    # Finding 1.8: zero placeholders diluted a_loss/entropy means; alpha_losses
    # got None placeholders (alpha_loss never logged for even num_updates_per_step,
    # mean_list crash for odd > 1).
    agent, _ = make_fake_env_sac_agent()
    a_losses, entropies, alphas, alpha_losses = [], [], [], []
    real = (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.2), torch.tensor(0.1))
    agent.extract_actor_stats(a_losses, entropies, alphas, alpha_losses, real)
    agent.extract_actor_stats(a_losses, entropies, alphas, alpha_losses, None)
    assert len(a_losses) == 1 and len(entropies) == 1 and len(alpha_losses) == 1
    # approx: fp32 tensor(0.1).item() == 0.10000000149011612, not the Python double 0.1
    assert a_losses[0].item() == 1.0 and alpha_losses[0].item() == pytest.approx(0.1)


def test_max_epochs_exit_without_any_completed_episode():
    # Finding 1.6: exits/saves were nested under game_rewards.current_size > 0 —
    # training could not stop before the first episode ended (pre-fix: this HANGS).
    agent, fake = make_fake_env_sac_agent(ep_len=10_000, max_epochs=3)
    agent.train()
    assert agent.epoch_num == 3, f"train() must exit at max_epochs=3, ran {agent.epoch_num}"


def test_replay_buffer_truncated_roundtrip():
    from rl_games.common.experience import VectorizedReplayBuffer
    buf = VectorizedReplayBuffer((OBS_DIM,), (ACT_DIM,), 8, 'cpu')
    o = torch.randn(3, OBS_DIM); a = torch.randn(3, ACT_DIM)
    r = torch.randn(3, 1); d = torch.tensor([[True], [False], [True]])
    tr = torch.tensor([[False], [False], [True]])
    buf.add(o, a, r, o, d, tr)
    obs, act, rew, nobs, done, trunc = buf.sample(3)
    assert trunc.shape == (3, 1) and trunc.dtype == torch.bool
    sd = buf.state_dict()
    assert 'truncated' in sd
    # backward compat: pre-upgrade state dicts lack the key
    sd.pop('truncated')
    buf2 = VectorizedReplayBuffer((OBS_DIM,), (ACT_DIM,), 8, 'cpu')
    buf2.load_state_dict(sd)
    assert not buf2.truncated.any()


def test_replay_buffer_add_defaults_truncated_false():
    from rl_games.common.experience import VectorizedReplayBuffer
    buf = VectorizedReplayBuffer((OBS_DIM,), (ACT_DIM,), 8, 'cpu')
    o = torch.randn(2, OBS_DIM); a = torch.randn(2, ACT_DIM)
    buf.add(o, a, torch.randn(2, 1), o, torch.zeros(2, 1, dtype=torch.bool))
    *_, trunc = buf.sample(2)
    assert not trunc.any()


def test_replay_buffer_truncated_overflow_wrap():
    from rl_games.common.experience import VectorizedReplayBuffer
    buf = VectorizedReplayBuffer((OBS_DIM,), (ACT_DIM,), 4, 'cpu')
    o = torch.randn(3, OBS_DIM); a = torch.randn(3, ACT_DIM)
    r = torch.randn(3, 1); d = torch.zeros(3, 1, dtype=torch.bool)
    tr1 = torch.tensor([[True], [False], [True]])
    buf.add(o, a, r, o, d, tr1)
    buf.add(o, a, r, o, d, tr1)  # wraps: capacity 4, 6 rows total
    assert buf.full
    # the wrap path (overflow copy block) must carry truncated too:
    # rows now at [0,1] are the overflow of the second add (tr1[-2:] = [False, True])
    assert buf.truncated[0].item() == False and buf.truncated[1].item() == True


def test_fake_env_declares_next_step_mode():
    agent, fake = make_fake_env_sac_agent()
    assert agent.env_info.get('autoreset_mode') == 'next_step'


def test_wrappers_declare_next_step_mode():
    import pathlib
    for f in ['rl_games/envs/envpool.py', 'rl_games/common/gymnasium_vecenv.py']:
        src = pathlib.Path(os.path.join(REPO, f)).read_text()
        assert 'autoreset_mode' in src and 'next_step' in src, f"{f} must declare autoreset_mode"


def test_can_concat_infos_sees_wrapper_attr():
    # Finding 55: only env.unwrapped was checked, so wrapper-level
    # concat_infos declarations (e.g. wrappers.TimeLimit) were invisible.
    from rl_games.common import vecenv as vecenv_mod

    class Inner:
        pass

    class Wrapper:
        concat_infos = True
        def __init__(self):
            self.unwrapped = Inner()

    class Bare:  # no concat_infos, no .unwrapped
        pass

    class UnwrappedOnly:
        def __init__(self):
            self.unwrapped = type('U', (), {'concat_infos': True})()

    assert vecenv_mod.can_concat_infos(Wrapper()) is True
    assert vecenv_mod.can_concat_infos(UnwrappedOnly()) is True
    assert vecenv_mod.can_concat_infos(Bare()) is False


def test_ray_merge_infos_without_concat_infos():
    # Finding 49: RayVecEnv dropped per-worker time_outs unless the env opted
    # into concat_infos; _merge_ray_infos is the extracted merge helper.
    from rl_games.common.vecenv import _merge_ray_infos

    merged = _merge_ray_infos([
        {'time_outs': True},                          # scalar (single-agent worker)
        {'time_outs': False},
        {'time_outs': np.array([True, False])},       # per-agent array
        {},                                           # worker without the key
    ])
    # no scores/battle_won in any worker -> no spurious keys
    assert set(merged) == {'time_outs'}
    np.testing.assert_array_equal(merged['time_outs'],
                                  np.array([True, False, True, False, False]))

    # battle_won merges per-worker, aligned with worker index
    merged = _merge_ray_infos([
        {'time_outs': False, 'battle_won': True},
        {'time_outs': False},
        {'time_outs': True, 'battle_won': False},
    ])
    assert set(merged) == {'time_outs', 'battle_won'}
    assert len(merged['battle_won']) == 3
    assert merged['battle_won'][0] == 1.0
    assert np.isnan(merged['battle_won'][1])  # interior gap: filler, never read
    assert merged['battle_won'][2] == 0.0


class _FakeObserverAlgo:
    """Minimal algo stub for DefaultAlgoObserver.after_init/process_infos."""
    games_to_track = 100
    ppo_device = 'cpu'
    writer = None
    num_agents = 1


def test_ray_merge_preserves_scores_for_observer():
    # Review fix for 18eb01b: the merged dict must keep carrying 'scores' so
    # DefaultAlgoObserver's dict path tracks the same results the old
    # per-worker-list path did (atari InfoWrapper, TestRNNEnv, CarRacing).
    from rl_games.common.algo_observer import DefaultAlgoObserver
    from rl_games.common.vecenv import _merge_ray_infos

    # workers 0/1 finished an episode (scores set on done), worker 2 timed out
    # without emitting scores
    worker_infos = [
        {'scores': 1.0, 'time_outs': False},
        {'scores': 0.0, 'time_outs': False},
        {'time_outs': True},
    ]
    merged = _merge_ray_infos(worker_infos)
    np.testing.assert_array_equal(merged['time_outs'],
                                  np.array([False, False, True]))
    # dict-path-consumable form: per-worker array truncated at the trailing
    # worker without the key, so the observer's length guard skips worker 2
    np.testing.assert_array_equal(merged['scores'], np.array([1.0, 0.0]))

    done_indices = torch.tensor([0, 1, 2])

    # old list path (pre-18eb01b behavior) as reference
    obs_list = DefaultAlgoObserver()
    obs_list.after_init(_FakeObserverAlgo())
    obs_list.process_infos(worker_infos, done_indices)

    # new merged dict through the observer's dict path
    obs_dict = DefaultAlgoObserver()
    obs_dict.after_init(_FakeObserverAlgo())
    obs_dict.process_infos(merged, done_indices)

    # both paths tracked exactly workers 0 and 1, skipped scoreless worker 2
    assert obs_list.game_scores.current_size == 2
    assert obs_dict.game_scores.current_size == obs_list.game_scores.current_size
    assert obs_dict.game_scores.get_mean() == pytest.approx(0.5)
    assert obs_dict.game_scores.get_mean() == pytest.approx(obs_list.game_scores.get_mean())


def test_ray_merge_nan_filler_does_not_poison_observer():
    # Episodic-life atari on ray: EpisodicLifeEnv wraps OUTSIDE InfoWrapper
    # (wrappers.py), so a worker can be done WITHOUT scores in the same step
    # where a later worker carries scores. _merge_ray_infos pads that interior
    # gap with NaN; the observer's dict path must skip the filler (old list
    # path semantics: missing key = skip). Feeding the NaN into AverageMeter
    # would poison its running mean permanently.
    from rl_games.common.algo_observer import DefaultAlgoObserver
    from rl_games.common.vecenv import _merge_ray_infos

    # worker 0 done without scores (life lost), worker 1 done with scores
    worker_infos = [
        {'time_outs': False},
        {'scores': 21.0, 'time_outs': False},
    ]
    merged = _merge_ray_infos(worker_infos)
    assert len(merged['scores']) == 2
    assert np.isnan(merged['scores'][0])  # interior gap filler
    assert merged['scores'][1] == 21.0

    done_indices = torch.tensor([0, 1])

    # old list path (reference semantics): worker 0 has no key -> skipped
    obs_list = DefaultAlgoObserver()
    obs_list.after_init(_FakeObserverAlgo())
    obs_list.process_infos(worker_infos, done_indices)

    # new merged dict through the observer's dict path
    obs_dict = DefaultAlgoObserver()
    obs_dict.after_init(_FakeObserverAlgo())
    obs_dict.process_infos(merged, done_indices)

    # both paths track exactly one game: worker 1's score
    assert obs_list.game_scores.current_size == 1
    assert obs_dict.game_scores.current_size == 1
    assert obs_list.game_scores.get_mean() == pytest.approx(21.0)
    assert obs_dict.game_scores.get_mean() == pytest.approx(21.0)

    # the meter must stay finite after a subsequent good update
    later = _merge_ray_infos([{'scores': 7.0, 'time_outs': False}])
    obs_dict.process_infos(later, torch.tensor([0]))
    assert obs_dict.game_scores.current_size == 2
    assert np.isfinite(obs_dict.game_scores.get_mean())
    assert obs_dict.game_scores.get_mean() == pytest.approx(14.0)
