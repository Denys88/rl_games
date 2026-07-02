"""Correctness tests for SAC: autoreset/truncation handling, replay-buffer writes,
normalizer statistics, warmup accounting, exit paths, and the benchmark harness."""
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
        # per-env episode lengths (uniform here; staggered in subclass)
        self.ep_lens = np.full(num_envs, ep_len, dtype=np.int64)
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
                if self.t[e] >= self.ep_lens[e]:
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


class StaggeredFakeNextStepVecEnv(FakeNextStepVecEnv):
    """Per-env episode length = ep_len + env_id: envs hit their reset-step
    garbage rows on DIFFERENT steps, so play_steps sees mixed live masks
    (some rows garbage while others live within the same vec step)."""

    def __init__(self, num_envs, ep_len=EP_LEN, terminate_instead=False):
        super().__init__(num_envs, ep_len, terminate_instead)
        self.ep_lens = ep_len + np.arange(num_envs, dtype=np.int64)


class SameStepFakeVecEnv(FakeNextStepVecEnv):
    """Deterministic SAME_STEP-autoreset vec env (classic gym semantics).

    Same obs encoding as FakeNextStepVecEnv, but when an episode ends at
    t == ep_len the env resets WITHIN the step: it returns the POST-RESET obs
    (t=0, episode+1) with done=True (truncated unless terminate_instead) and a
    reward that is valid for the final step. The TRUE final obs (t == ep_len)
    is only exposed via info['final_observation'] (np array [num_envs, OBS_DIM])
    when provide_final_obs=True; without it the agent must fall back to the
    documented s_t proxy.
    """

    def __init__(self, num_envs, ep_len=EP_LEN, terminate_instead=False,
                 provide_final_obs=False):
        super().__init__(num_envs, ep_len, terminate_instead)
        self.provide_final_obs = provide_final_obs

    def step(self, actions):
        rewards = np.ones(self.num_envs, dtype=np.float32)  # final-step reward is VALID
        self.t += 1
        done = self.t >= self.ep_lens
        truncated = done if not self.terminate_instead else np.zeros_like(done)
        final_obs = self._obs()  # TRUE final obs (t == ep_len on done envs), pre-reset
        self.t[done] = 0
        self.episode[done] += 1
        infos = {'time_outs': truncated.copy()}
        if self.provide_final_obs:
            infos['final_observation'] = final_obs
        return self._obs(), rewards, done.astype(np.float32), infos

    def get_env_info(self):
        info = super().get_env_info()
        info['autoreset_mode'] = 'same_step'
        return info


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
        # one full episode per epoch keeps the reward/length accounting
        # assertions deterministic
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
    # Regression: next_step autoreset reset-step garbage rows must never be stored.
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
    # At truncation, next_obs must be the TRUE final obs (t==EP_LEN),
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
    # Regression: the env_info-supplied construction path left self.vec_env unset.
    agent, fake = make_fake_env_sac_agent()
    assert agent.vec_env is fake
    obs = agent.env_reset()
    if isinstance(obs, dict):  # obs_to_tensors wraps plain obs into {'obs': ...}
        obs = obs['obs']
    assert obs.shape == (3, OBS_DIM)


def test_warmup_epoch_count_exact():
    # epoch_num increments BEFORE train_epoch, so `<` gives
    # num_warmup_steps - 1 warmup epochs; num_warmup_steps=1 disables warmup.
    agent, fake = make_fake_env_sac_agent(num_warmup_frames=None, num_warmup_steps=1, max_epochs=1)
    seen = []
    orig = agent.play_steps
    agent.play_steps = lambda random_exploration=False: (seen.append(random_exploration), orig(random_exploration))[1]
    agent.train()
    assert seen == [True], f"num_warmup_steps=1 must give exactly one warmup epoch, got {seen}"


def test_gamma_tensor_stays_fp32_under_mixed_precision():
    # Regression: gamma_tensor recast to bf16 turns 0.99 into 0.98828125.
    agent, _ = make_fake_env_sac_agent(mixed_precision=True)
    assert agent.gamma_tensor.dtype == torch.float32
    # abs=1e-7: fp32 representation of 0.99 is off by ~9.5e-9; the bf16 bug was off by 1.7e-3
    assert agent.gamma_tensor.item() == pytest.approx(0.99, abs=1e-7)


def test_actor_stats_skip_non_update_steps():
    # Regression: zero placeholders diluted a_loss/entropy means; alpha_losses
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
    # Regression: exits/saves were nested under game_rewards.current_size > 0 —
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


def test_replay_buffer_zero_row_add_does_not_set_full():
    # Regression: `full = full or idx == 0` marked a FRESH buffer full on a
    # zero-row add (idx stays 0). The old `if live.any()` guard in play_steps
    # masked this; the guard is gone, the buffer must be correct on its own.
    from rl_games.common.experience import VectorizedReplayBuffer
    buf = VectorizedReplayBuffer((OBS_DIM,), (ACT_DIM,), 8, 'cpu')
    z_o = torch.empty(0, OBS_DIM); z_a = torch.empty(0, ACT_DIM)
    z_r = torch.empty(0, 1); z_d = torch.empty(0, 1, dtype=torch.bool)
    buf.add(z_o, z_a, z_r, z_o, z_d, z_d)
    assert buf.full is False and buf.idx == 0
    # still empty -> sampling must fail loudly, not return garbage rows
    with pytest.raises(RuntimeError):
        buf.sample(2)
    # subsequent real adds behave normally and sample only from idx valid rows
    buf.add(torch.randn(2, OBS_DIM), torch.randn(2, ACT_DIM), torch.randn(2, 1),
            torch.randn(2, OBS_DIM), torch.zeros(2, 1, dtype=torch.bool))
    assert buf.full is False and buf.idx == 2
    obs, *_ = buf.sample(4)
    assert obs.shape == (4, OBS_DIM)


def test_staggered_episodes_mixed_live_mask():
    # Exercise a MIXED live mask — ep_lens [5, 6, 7] put each env's
    # reset-step garbage row on a different step (6/12, 7/14, 8/16), so some
    # envs are on garbage rows while others are live in the same vec step.
    N = 16  # steps; garbage rows per env e: N // (ep_lens[e] + 1) == 2 each
    agent, fake = make_fake_env_sac_agent(
        env_cls=StaggeredFakeNextStepVecEnv,
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=N)
    obses, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    # buffer row count == total live steps from the staggered schedule
    expected_rows = sum(N - N // (l + 1) for l in fake.ep_lens)
    assert obses.shape[0] == expected_rows  # 42
    for e in range(fake.num_envs):
        rows_e = int((obses[:, 0] == e).sum())
        assert rows_e == N - N // (int(fake.ep_lens[e]) + 1)
    # no cross-episode rows: stored transitions are contiguous within episodes
    assert torch.all(next_obses[:, 1] == obses[:, 1] + 1)
    assert torch.all(next_obses[:, 2] == obses[:, 2])
    # accounting under mixed masks: two completed episodes per env -> mean of
    # (5, 5, 6, 6, 7, 7) == 6.0
    assert agent.game_lengths.get_mean() == pytest.approx(6.0, rel=1e-3)


def test_same_step_without_final_obs_stores_proxy():
    # same_step runtime, no info['final_observation']: the documented fallback
    # stores the s_t proxy as next_obs on truncation rows.
    steps = EP_LEN + 3
    agent, fake = make_fake_env_sac_agent(
        env_cls=SameStepFakeVecEnv,
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=steps)
    obses, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    # same_step: every row is live — nothing skipped
    assert obses.shape[0] == steps * fake.num_envs
    rows = trunc[:, 0].nonzero(as_tuple=True)[0]
    assert len(rows) == fake.num_envs  # one truncation per env in 8 steps
    # proxy fallback: next_obs == obs (s_t), NOT the post-reset obs the env returned
    assert torch.equal(next_obses[rows], obses[rows])
    assert not dones[rows].any().item(), "truncated rows must bootstrap (done=False)"
    assert trunc[rows].all().item()


def test_same_step_with_final_observation_stores_true_final_obs():
    # same_step runtime WITH info['final_observation'] (np array
    # [num_envs, OBS_DIM]): truncation rows must carry the TRUE final obs
    # (t == ep_len, same env, same episode), not the proxy or post-reset obs.
    steps = EP_LEN + 3
    agent, fake = make_fake_env_sac_agent(
        env_cls=functools.partial(SameStepFakeVecEnv, provide_final_obs=True),
        num_warmup_frames=None, num_warmup_steps=10_000,
        max_epochs=1, num_steps_per_episode=steps)
    obses, _, _, next_obses, dones, trunc = _completed_training_buffer(agent)
    assert obses.shape[0] == steps * fake.num_envs
    rows = trunc[:, 0].nonzero(as_tuple=True)[0]
    assert len(rows) == fake.num_envs
    assert torch.all(next_obses[rows][:, 1] == EP_LEN), "must store the TRUE final obs (t == ep_len)"
    assert torch.all(next_obses[rows][:, 0] == obses[rows][:, 0])  # same env
    assert torch.all(next_obses[rows][:, 2] == obses[rows][:, 2])  # same episode
    assert not dones[rows].any().item()
    assert trunc[rows].all().item()


def test_gymnasium_manual_path_declares_same_step():
    # Regression: _step_manual resets within the same step (same_step
    # semantics), but get_env_info unconditionally declared 'next_step'.
    # env_creator bypasses gymnasium.make, so the manual path is constructible
    # with a stub env — no env registration needed.
    from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv

    class _StubEnv:
        observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    venv = GymnasiumVecEnv('stub', 2, env_creator=lambda: _StubEnv())
    assert venv._use_native_vec is False
    assert venv.get_env_info()['autoreset_mode'] == 'same_step'


def test_fake_env_declares_next_step_mode():
    agent, fake = make_fake_env_sac_agent()
    assert agent.env_info.get('autoreset_mode') == 'next_step'


class MultiAgentFakeNextStepVecEnv(FakeNextStepVecEnv):
    def get_env_info(self):
        info = super().get_env_info()
        info['agents'] = 2
        return info


def test_multi_agent_next_step_rejected():
    # Per-env done tracking (_prev_dones) and live-row striding
    # in the next_step path assume one row per env — multi-agent rows would
    # silently misalign. The agent must refuse loudly at construction.
    with pytest.raises(ValueError, match="multi-agent"):
        make_fake_env_sac_agent(env_cls=MultiAgentFakeNextStepVecEnv)


def test_wrappers_declare_next_step_mode():
    import pathlib
    for f in ['rl_games/envs/envpool.py', 'rl_games/common/gymnasium_vecenv.py']:
        src = pathlib.Path(os.path.join(REPO, f)).read_text()
        assert 'autoreset_mode' in src and 'next_step' in src, f"{f} must declare autoreset_mode"


def test_can_concat_infos_sees_wrapper_attr():
    # Regression: only env.unwrapped was checked, so wrapper-level
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
    # Regression: RayVecEnv dropped per-worker time_outs unless the env opted
    # into concat_infos; _merge_ray_infos is the extracted merge helper.
    from rl_games.common.vecenv import _merge_ray_infos

    infos_list = [
        {'time_outs': True},                          # scalar (single-agent worker)
        {'time_outs': False},
        {'time_outs': np.array([True, False])},       # per-agent array
        {},                                           # worker without the key
    ]
    merged = _merge_ray_infos(infos_list)
    # no scores/battle_won in any worker -> no spurious keys
    assert set(merged) == {'time_outs', 'worker_infos'}
    np.testing.assert_array_equal(merged['time_outs'],
                                  np.array([True, False, True, False, False]))
    # escape hatch: custom observers reading keys outside the merged set get
    # the ORIGINAL per-worker list (identity, not a copy)
    assert merged['worker_infos'] is infos_list

    # battle_won merges per-worker, aligned with worker index
    merged = _merge_ray_infos([
        {'time_outs': False, 'battle_won': True},
        {'time_outs': False},
        {'time_outs': True, 'battle_won': False},
    ])
    assert set(merged) == {'time_outs', 'battle_won', 'worker_infos'}
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


def test_utd_ratio_sets_updates_per_step():
    # utd_ratio = gradient updates per env FRAME, so the
    # update count scales with num_actors instead of silently diluting.
    # num_updates_per_step=None: independent of whatever key the yaml carries.
    agent, _ = make_fake_env_sac_agent(num_envs=8, utd_ratio=0.5, num_updates_per_step=None)
    assert agent.num_updates_per_step == 4  # round(0.5 * 8)


def test_legacy_num_updates_per_step_still_honored():
    agent, _ = make_fake_env_sac_agent(num_envs=8, num_updates_per_step=3, utd_ratio=None)
    assert agent.num_updates_per_step == 3


class ScaledActionsFakeEnv(FakeNextStepVecEnv):
    """Action space with non-unit bounds: scale = 0.4 per dim, bias = 0."""

    def get_env_info(self):
        info = super().get_env_info()
        info['action_space'] = spaces.Box(-0.4, 0.4, (ACT_DIM,), dtype=np.float32)
        return info


def test_log_prob_includes_action_scale_jacobian():
    # The entropy term must refer to the ENV action space.
    agent, _ = make_fake_env_sac_agent()
    assert agent.log_action_scale_sum.item() == pytest.approx(0.0, abs=1e-6)  # unit bounds

    agent2, _ = make_fake_env_sac_agent(env_cls=ScaledActionsFakeEnv)
    expected = ACT_DIM * float(np.log(0.4))
    assert agent2.log_action_scale_sum.item() == pytest.approx(expected, rel=1e-5)


def test_update_paths_use_env_space_log_prob():
    import inspect
    from rl_games.algos_torch import sac_agent
    src = inspect.getsource(sac_agent.SACAgent.update_critic) + \
          inspect.getsource(sac_agent.SACAgent.update_actor_and_alpha)
    # both update sites must route log-probs through the env-space helper...
    assert src.count('_env_log_prob') >= 2
    # ...and the helper itself applies the action-scale Jacobian correction
    helper = inspect.getsource(sac_agent.SACAgent._env_log_prob)
    assert 'log_action_scale_sum' in helper


def test_env_log_prob_matches_analytic_jacobian():
    # Numeric end-to-end: the value both update sites consume must equal
    # log pi_norm(a) - sum(log action_scale), with the analytic constant.
    agent, _ = make_fake_env_sac_agent(env_cls=ScaledActionsFakeEnv)
    torch.manual_seed(0)
    obs = torch.randn(5, OBS_DIM)
    with torch.no_grad():
        dist = agent.model.actor(obs)
        a = dist.sample()
        lp_norm = dist.log_prob(a).sum(-1, keepdim=True)
        got = agent._env_log_prob(dist, a)
    expected = lp_norm - ACT_DIM * float(np.log(0.4))
    assert got.shape == (5, 1)
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_sac_player_rescales_actions_per_dim():
    # Regression: SACPlayer emitted raw [-1, 1] actions clamped to the GLOBAL
    # env min/max — wrong for non-unit and per-dim-asymmetric action boxes.
    # Constructible without an env: env_info in config skips env creation.
    from rl_games.algos_torch.players import SACPlayer
    params = _load_params(SAC_YAML)
    cfg = params['config']
    cfg.pop('env_config', None)
    info = FakeNextStepVecEnv(2).get_env_info()
    low = np.array([-0.4, 0.0], dtype=np.float32)   # per-dim scale [0.4, 1.0], bias [0, 1.0]
    high = np.array([0.4, 2.0], dtype=np.float32)
    info['action_space'] = spaces.Box(low, high, dtype=np.float32)
    cfg.update({'env_info': info, 'device_name': 'cpu', 'normalize_input': False})
    player = SACPlayer(params)
    assert player.env is None  # constructed without an env

    a = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-2.0, 2.0]])  # last row: fp-error clamp
    out = player.rescale_actions(a)
    expected = torch.tensor([[0.4, 0.0], [0.2, 1.25], [-0.4, 2.0]])
    assert torch.allclose(out, expected, atol=1e-6)

    # end-to-end through get_action: deterministic action == rescale(dist.mean)
    obs = torch.zeros(1, OBS_DIM)
    player.has_batch_dimension = True
    with torch.no_grad():
        got = player.get_action(obs, is_deterministic=True)
        dist = player.model.actor(player.model.norm_obs(obs))
        expected = player.rescale_actions(dist.mean)
    assert torch.allclose(got, expected)
    assert torch.all(got >= torch.from_numpy(low)) and torch.all(got <= torch.from_numpy(high))


# RunningMeanStd.__init__ (running_mean_std.py) registers the count buffer as
# torch.ones((), dtype=torch.int64) with running_mean=0 — i.e. one phantom
# zero-valued sample baked in before any real data.
RMS_COUNT_INIT = 1


def test_normalizer_counts_each_frame_exactly_once():
    # Normalizer stats must update once per env frame from rollout data,
    # never from replayed minibatches.
    agent, fake = make_fake_env_sac_agent(
        normalize_input=True, max_epochs=2, num_steps_per_episode=4,
        num_warmup_frames=None, num_warmup_steps=1)
    agent.train()
    rms = agent.model.running_mean_std
    # frames: num_envs * (1 reset obs + max_epochs * steps_per_epoch)
    frames_seen = 3 * (1 + 2 * 4)  # 27
    # + RMS_COUNT_INIT: the count buffer starts at 1 (phantom sample), so after
    # ingesting exactly the 27 streamed frames it must read 28. Pre-fix red
    # evidence: rollout frames are never counted, while epoch 2's 4 updates
    # each push obs+next_obs of a batch-16 minibatch -> 1 + 4 * 2 * 16 = 129.
    assert int(rms.count.item()) == frames_seen + RMS_COUNT_INIT, \
        f"normalizer count {int(rms.count.item())} != {frames_seen} + init {RMS_COUNT_INIT}"


def test_normalizer_frozen_during_update():
    agent, fake = make_fake_env_sac_agent(
        normalize_input=True, max_epochs=1, num_steps_per_episode=4,
        num_warmup_frames=None, num_warmup_steps=10_000)  # warmup fills buffer, no grad updates
    agent.train()
    count_before = int(agent.model.running_mean_std.count.item())
    agent.set_train()
    agent.update(agent.epoch_num)
    assert int(agent.model.running_mean_std.count.item()) == count_before, \
        "update() mutated normalizer statistics from replayed minibatches"


def test_normalizer_stats_match_streamed_frames():
    # the mean must reflect ALL streamed frames (incl. post-reset obs), not replay
    agent, fake = make_fake_env_sac_agent(
        normalize_input=True, max_epochs=2, num_steps_per_episode=4,
        num_warmup_frames=None, num_warmup_steps=10_000)
    agent.train()
    rms = agent.model.running_mean_std
    # Streamed-frame schedule (col 1 of the fake obs is step-in-episode t):
    # 3 lock-stepped envs, ep_len=5, 8 env steps after the initial reset obs.
    #   reset obs:        t=0
    #   steps 1..5:       t=1,2,3,4,5   (done+truncated at t=5, pending_reset)
    #   step 6:           t=0           (post-reset garbage-step obs — still streamed)
    #   steps 7..8:       t=1,2
    # Per-env sequence [0,1,2,3,4,5,0,1,2]: 9 frames, sum 18.
    # All 3 envs identical -> 27 frames, total sum 54.
    # RMS starts with a phantom zero sample (count=1, mean=0), so the merged
    # mean is 54 / (27 + 1) = 27/14, not 54/27.
    expected_mean_t = 54.0 / (27 + RMS_COUNT_INIT)  # = 27/14 ~= 1.9285714
    # rel=1e-5: inputs are small exact integers; running buffers are float64
    # (float32 on MPS-only machines), so only division round-off remains.
    assert rms.running_mean[1].item() == pytest.approx(expected_mean_t, rel=1e-5)


# ---------------------------------------------------------------------------
# Benchmark harness smoke tests (benchmarks/sac_benchmark.py)
# ---------------------------------------------------------------------------

BENCHMARK_HARNESS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmarks', 'sac_benchmark.py')


def test_benchmark_report_handles_missing_out_dir(tmp_path):
    import ast
    import importlib.util
    import subprocess

    # no module-level envpool import (envpool is an optional extra)
    tree = ast.parse(open(BENCHMARK_HARNESS).read())
    top_imports = {alias.name.split('.')[0]
                   for node in tree.body if isinstance(node, ast.Import)
                   for alias in node.names}
    top_imports |= {node.module.split('.')[0]
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module}
    assert 'envpool' not in top_imports, 'harness must not import envpool at module level'

    # module imports clean without envpool installed
    spec = importlib.util.spec_from_file_location('sac_benchmark', BENCHMARK_HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # --report on a missing OUT dir: exit 0, one n/a row per battery entry
    env = {**os.environ, 'SAC_BENCH_OUT': str(tmp_path / 'does-not-exist')}
    res = subprocess.run([sys.executable, BENCHMARK_HARNESS, '--report'],
                         env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.count('n/a') >= len(mod.RUNS)
    for tag, _cfg, _seed, _frames in mod.RUNS:
        assert tag in res.stdout


def test_benchmark_probe_exits_with_hint_when_envpool_missing():
    import importlib.util
    import subprocess

    # only assert the failure branch when envpool is genuinely absent;
    # with envpool installed --probe would launch a 600 s training run.
    if importlib.util.find_spec('envpool') is not None:
        pytest.skip('envpool installed; --probe would start the throughput probe')
    res = subprocess.run([sys.executable, BENCHMARK_HARNESS, '--probe'],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 1
    assert 'uv sync --extra envpool --extra mujoco' in res.stderr
