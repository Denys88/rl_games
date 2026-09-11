"""Gymnasium-API contract tests for the legacy wrapper family.

``rl_games/common/wrappers.py`` used to implement the pre-gymnasium API
(``reset()`` -> obs, ``step()`` -> 4-tuple) while every vecenv that consumes
those wrappers (``RayWorker``, the gymnasium vecenv) speaks the gymnasium
contract (``reset()`` -> ``(obs, info)``, ``step()`` ->
``(obs, reward, terminated, truncated, info)``).  Every route that composed
these wrappers -- ``atari_gym``, ``smac``/``smac_v2``/``smac_cnn``,
``BipedalWalkerCnn-v3``, ``CarRacing-v2`` -- was therefore dead on arrival.

These tests pin the gymnasium contract on the whole family.  They need no
ale_py / ray / smac / pettingzoo: the Atari-like and multi-agent envs below
are synthetic.
"""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces

from rl_games.common import wrappers
from rl_games.common import env_configurations
from rl_games.common.vecenv import RayWorker


# --------------------------------------------------------------------------
# contract helpers
# --------------------------------------------------------------------------

def _is_flag(x):
    """terminated/truncated may be a scalar bool or a per-agent bool array."""
    if isinstance(x, (bool, np.bool_)):
        return True
    return isinstance(x, np.ndarray) and x.dtype == np.bool_


def check_reset(env, **kwargs):
    result = env.reset(**kwargs)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"{type(env).__name__}.reset must return (obs, info), got {type(result)}")
    obs, info = result
    assert isinstance(info, dict), (
        f"{type(env).__name__}.reset info must be a dict, got {type(info)}")
    return obs, info


def check_step(env, action):
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5, (
        f"{type(env).__name__}.step must return a 5-tuple, got "
        f"{len(result) if isinstance(result, tuple) else type(result)}")
    obs, reward, terminated, truncated, info = result
    assert _is_flag(terminated), f"terminated must be bool/ndarray, got {terminated!r}"
    assert _is_flag(truncated), f"truncated must be bool/ndarray, got {truncated!r}"
    assert isinstance(info, dict)
    return result


# --------------------------------------------------------------------------
# synthetic envs
# --------------------------------------------------------------------------

class FakeSignalEnv(gym.Env):
    """Flat-obs gymnasium env with scripted termination / truncation."""

    def __init__(self, terminate_at=None, truncate_at=None, obs_dim=3, reward=1.0):
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.reward = reward
        self.t = 0
        self.reset_count = 0
        self.step_count = 0
        self.last_seed = 'unset'

    def _obs(self):
        return np.full(self.observation_space.shape, float(self.t), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_seed = seed
        self.reset_count += 1
        self.t = 0
        return self._obs(), {'reset_count': self.reset_count}

    def step(self, action):
        self.step_count += 1
        self.t += 1
        terminated = self.terminate_at is not None and self.t >= self.terminate_at
        truncated = self.truncate_at is not None and self.t >= self.truncate_at
        return self._obs(), self.reward, terminated, truncated, {'t': self.t}


class _FakeALE:
    def __init__(self, env):
        self._env = env

    def lives(self):
        return self._env.lives

    def getRAM(self):
        return np.zeros(128, dtype=np.uint8)


class FakeAtariEnv(gym.Env):
    """Minimal ALE-like gymnasium env: lives, action meanings, uint8 image obs."""

    def __init__(self, n_lives=3, steps_per_life=5, shape=(84, 84, 3)):
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.n_lives = n_lives
        self.steps_per_life = steps_per_life
        self.lives = n_lives
        self.ale = _FakeALE(self)
        self.t = 0
        self.reset_count = 0
        self.step_count = 0
        self.actions_taken = []

    def get_action_meanings(self):
        return ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

    def _obs(self):
        return np.full(self.observation_space.shape, self.t % 256, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self.lives = self.n_lives
        self.t = 0
        return self._obs(), {'reset_count': self.reset_count}

    def step(self, action):
        self.step_count += 1
        self.actions_taken.append(action)
        self.t += 1
        if self.t % self.steps_per_life == 0 and self.lives > 0:
            self.lives -= 1
        terminated = self.lives <= 0
        return self._obs(), 1.0, terminated, False, {'t': self.t}


class FakeMultiAgentEnv(gym.Env):
    """SMAC-shaped env: per-agent spaces, obs batched over agents, array flags."""

    def __init__(self, n_agents=3, obs_dim=5, state_dim=7, with_states=False, episode_len=6):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.with_states = with_states
        self.episode_len = episode_len
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.state_space = spaces.Box(low=0, high=1, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.concat_infos = True
        self.t = 0

    def get_number_of_agents(self):
        return self.n_agents

    def _obs(self):
        obs = np.full((self.n_agents, self.obs_dim), float(self.t), dtype=np.float32)
        if not self.with_states:
            return obs
        state = np.full((self.state_dim,), float(self.t), dtype=np.float32)
        return {'obs': obs, 'state': state}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        terminated = np.repeat(done, self.n_agents)
        truncated = np.repeat(False, self.n_agents)
        info = {'time_outs': [False] * self.n_agents}
        rewards = np.ones(self.n_agents, dtype=np.float32)
        return self._obs(), rewards, terminated, truncated, info


# --------------------------------------------------------------------------
# 1. per-wrapper gymnasium contract
# --------------------------------------------------------------------------

def test_info_wrapper_contract():
    base = FakeSignalEnv(terminate_at=3)
    env = wrappers.InfoWrapper(base)
    check_reset(env)
    for _ in range(2):
        _, _, terminated, truncated, info = check_step(env, 0)
        assert not terminated and not truncated
        assert 'scores' not in info
    _, _, terminated, truncated, info = check_step(env, 0)
    assert terminated and not truncated
    assert info['scores'] == pytest.approx(3.0)


def test_info_wrapper_reports_scores_on_truncation():
    env = wrappers.InfoWrapper(FakeSignalEnv(truncate_at=2))
    check_reset(env)
    check_step(env, 0)
    _, _, terminated, truncated, info = check_step(env, 0)
    assert truncated and not terminated
    assert info['scores'] == pytest.approx(2.0)


def test_noop_reset_env_contract():
    base = FakeAtariEnv()
    env = wrappers.NoopResetEnv(base, noop_max=5)
    env.override_num_noops = 3
    obs, _ = check_reset(env)
    assert obs.shape == base.observation_space.shape
    assert base.step_count == 3
    assert base.reset_count == 1
    check_step(env, 0)


def test_noop_reset_env_rereset_on_episode_end():
    base = FakeAtariEnv(n_lives=1, steps_per_life=2)
    env = wrappers.NoopResetEnv(base, noop_max=5)
    env.override_num_noops = 5
    obs, _ = check_reset(env)
    # env terminates every 2 no-ops -> the loop must reset and keep going
    assert base.reset_count > 1
    assert obs.shape == base.observation_space.shape


def test_fire_reset_env_contract():
    base = FakeAtariEnv()
    env = wrappers.FireResetEnv(base)
    obs, _ = check_reset(env)
    assert obs.shape == base.observation_space.shape
    assert base.actions_taken == [1, 2]
    check_step(env, 0)


def test_fire_reset_env_rereset_on_episode_end():
    base = FakeAtariEnv(n_lives=1, steps_per_life=1)
    env = wrappers.FireResetEnv(base)
    check_reset(env)
    # every FIRE/UP step ends the episode -> two re-resets on top of the first
    assert base.reset_count == 3


def test_episodic_life_env_contract():
    base = FakeAtariEnv(n_lives=2, steps_per_life=2)
    env = wrappers.EpisodicLifeEnv(base)
    check_reset(env)
    assert base.reset_count == 1
    check_step(env, 0)
    _, _, terminated, truncated, _ = check_step(env, 0)
    # life lost (2 -> 1) but the game is not over
    assert terminated is True
    assert truncated is False
    assert env.was_real_done is False

    # reset after a life loss must be a no-op step, not a real reset
    resets_before, steps_before = base.reset_count, base.step_count
    check_reset(env)
    assert base.reset_count == resets_before
    assert base.step_count == steps_before + 1

    # play to the real game over
    for _ in range(5):
        _, _, terminated, truncated, _ = check_step(env, 0)
        if env.was_real_done:
            break
    assert env.was_real_done is True
    assert terminated is True
    check_reset(env)
    assert base.reset_count == resets_before + 1


def test_episodic_life_env_treats_truncation_as_real_done():
    base = FakeSignalEnv(truncate_at=1)
    base.ale = _FakeALE(base)
    base.lives = 1
    env = wrappers.EpisodicLifeEnv(base)
    check_reset(env)
    _, _, _, truncated, _ = check_step(env, 0)
    assert truncated is True
    assert env.was_real_done is True


def test_episode_stacked_env_contract():
    base = FakeSignalEnv(reward=0.0)
    env = wrappers.EpisodeStackedEnv(base)
    env.max_stacked_steps = 3
    check_reset(env)
    for _ in range(2):
        _, _, terminated, truncated, _ = check_step(env, 0)
        assert not terminated and not truncated
    _, reward, terminated, truncated, _ = check_step(env, 0)
    assert terminated is True
    assert reward == -1


def test_max_and_skip_env_contract():
    base = FakeSignalEnv(terminate_at=10)
    env = wrappers.MaxAndSkipEnv(base, skip=4, use_max=False)
    obs, _ = check_reset(env)
    assert obs.shape == base.observation_space.shape
    obs, reward, terminated, truncated, _ = check_step(env, 0)
    assert base.step_count == 4
    assert reward == pytest.approx(4.0)
    assert not terminated and not truncated


def test_max_and_skip_env_breaks_on_terminated():
    base = FakeSignalEnv(terminate_at=2)
    env = wrappers.MaxAndSkipEnv(base, skip=4, use_max=False)
    check_reset(env)
    _, reward, terminated, truncated, _ = check_step(env, 0)
    assert terminated is True
    assert truncated is False
    assert base.step_count == 2
    assert reward == pytest.approx(2.0)


def test_max_and_skip_env_breaks_on_truncated():
    base = FakeSignalEnv(truncate_at=2)
    env = wrappers.MaxAndSkipEnv(base, skip=4, use_max=False)
    check_reset(env)
    _, _, terminated, truncated, _ = check_step(env, 0)
    assert truncated is True
    assert terminated is False
    assert base.step_count == 2


def test_max_and_skip_env_use_max_uint8():
    base = FakeAtariEnv()
    env = wrappers.MaxAndSkipEnv(base, skip=4)
    obs, _ = check_reset(env)
    assert obs.shape == base.observation_space.shape
    obs, _, _, _, _ = check_step(env, 0)
    assert obs.shape == base.observation_space.shape


def test_frame_stack_contract_flat_obs():
    base = FakeSignalEnv(terminate_at=10, obs_dim=3)
    env = wrappers.FrameStack(base, 2, False)
    obs, _ = check_reset(env)
    assert obs.shape == (2, 3)
    assert env.observation_space.shape == (2, 3)
    obs, _, terminated, truncated, _ = check_step(env, 0)
    assert obs.shape == (2, 3)
    assert not terminated and not truncated


def test_frame_stack_contract_image_obs():
    base = FakeAtariEnv()
    env = wrappers.FrameStack(base, 4)
    obs, _ = check_reset(env)
    assert obs.shape == (84, 84, 12)
    obs, _, _, _, _ = check_step(env, 0)
    assert obs.shape == (84, 84, 12)


def test_frame_stack_forwards_reset_kwargs():
    base = FakeSignalEnv()
    env = wrappers.FrameStack(base, 2, False)
    check_reset(env, seed=123)
    assert base.last_seed == 123


def test_sticky_action_env_contract():
    base = FakeSignalEnv(terminate_at=10)
    env = wrappers.StickyActionEnv(base, p=1.0)
    check_reset(env)
    assert env.last_action == 0
    check_step(env, 1)
    check_step(env, 1)


def test_montezuma_info_wrapper_contract():
    base = FakeAtariEnv(n_lives=1, steps_per_life=2)
    env = wrappers.MontezumaInfoWrapper(base, room_address=3)
    check_reset(env)
    check_step(env, 0)
    _, _, terminated, _, info = check_step(env, 0)
    assert terminated is True
    assert 'visited_rooms' in info['scores']


def test_really_done_wrapper_contract():
    base = FakeAtariEnv(n_lives=3, steps_per_life=2)
    env = wrappers.ReallyDoneWrapper(base)
    check_reset(env)
    for _ in range(3):
        _, _, terminated, truncated, _ = check_step(env, 0)
        assert not terminated
        assert truncated is False


def test_allow_backtracking_contract():
    base = FakeSignalEnv(terminate_at=5, reward=2.0)
    env = wrappers.AllowBacktracking(base)
    check_reset(env)
    _, reward, _, _, _ = check_step(env, 0)
    assert reward == pytest.approx(2.0)


def test_impala_env_wrapper_contract():
    base = FakeAtariEnv()
    env = wrappers.ImpalaEnvWrapper(base)
    obs, _ = check_reset(env)
    assert set(obs.keys()) == {'observation', 'reward', 'last_action'}
    obs, _, _, _, _ = check_step(env, 0)
    assert set(obs.keys()) == {'observation', 'reward', 'last_action'}


# --------------------------------------------------------------------------
# 2. RayWorker over FrameStack
# --------------------------------------------------------------------------

@pytest.fixture
def framestack_cartpole_route():
    name = '_test_legacy_framestack_cartpole'
    env_configurations.register(name, {
        'env_creator': lambda **kwargs: wrappers.FrameStack(gym.make('CartPole-v1'), 4, False),
        'vecenv_type': 'RAY',
    })
    yield name
    env_configurations.configurations.pop(name, None)


def test_ray_worker_with_frame_stack(framestack_cartpole_route):
    worker = RayWorker(framestack_cartpole_route, {})
    obs = worker.reset()
    assert obs.shape == (4, 4)
    dones = 0
    for _ in range(50):
        # always push left: CartPole terminates within ~10 steps, exercising auto-reset
        obs, reward, done, info = worker.step(0)
        assert obs.shape == (4, 4)
        assert 'time_outs' in info
        assert np.ndim(reward) == 0
        if done:
            dones += 1
            # auto-reset: the returned obs is the fresh episode's stacked obs
            assert np.allclose(obs[0], obs[-1])
    assert dones > 0
    worker.close()


@pytest.fixture
def batched_stack_route():
    """SMAC-shaped route: BatchedFrameStackWithStates over a central-value env.

    Stands in for `create_smac`/`create_smac_cnn`, which cannot run here (no
    StarCraft II). Same wrapper, same per-agent array flags.
    """
    name = '_test_legacy_batched_stack'
    env_configurations.register(name, {
        'env_creator': lambda **kwargs: wrappers.BatchedFrameStackWithStates(
            FakeMultiAgentEnv(n_agents=3, obs_dim=5, state_dim=7, with_states=True,
                              episode_len=4),
            4, transpose=False, flatten=True),
        'vecenv_type': 'RAY',
    })
    yield name
    env_configurations.configurations.pop(name, None)


def test_ray_worker_with_batched_frame_stack(batched_stack_route):
    worker = RayWorker(batched_stack_route, {})
    assert worker.get_number_of_agents() == 3
    obs = worker.reset()
    assert obs['obs'].shape == (3, 20)
    dones = 0
    for _ in range(10):
        obs, reward, done, info = worker.step(np.zeros(3, dtype=np.int64))
        assert obs['obs'].shape == (3, 20)
        assert np.shape(done) == (3,)
        assert 'time_outs' in info
        if np.all(done):
            dones += 1
    assert dones > 0
    worker.close()


# --------------------------------------------------------------------------
# 3. nested wrappers
# --------------------------------------------------------------------------

def test_nested_max_and_skip_over_info_wrapper_cartpole():
    env = wrappers.MaxAndSkipEnv(wrappers.InfoWrapper(gym.make('CartPole-v1')), skip=4, use_max=False)
    obs, _ = check_reset(env, seed=0)
    assert obs.shape == (4,)
    for _ in range(10):
        obs, _, terminated, truncated, _ = check_step(env, 0)
        assert obs.shape == (4,)
        if terminated or truncated:
            check_reset(env)


# --------------------------------------------------------------------------
# 4. TimeLimit
# --------------------------------------------------------------------------

def test_time_limit_truncates_never_terminates():
    env = wrappers.TimeLimit(FakeSignalEnv(), 5)
    check_reset(env)
    for _ in range(4):
        _, _, terminated, truncated, info = check_step(env, 0)
        assert terminated is False
        assert truncated is False
        assert info['time_outs'] is False
    _, _, terminated, truncated, info = check_step(env, 0)
    assert terminated is False
    assert truncated is True
    assert info['time_outs'] is True


def test_time_limit_preserves_inner_termination():
    env = wrappers.TimeLimit(FakeSignalEnv(terminate_at=2), 5)
    check_reset(env)
    check_step(env, 0)
    _, _, terminated, truncated, info = check_step(env, 0)
    assert terminated is True
    assert truncated is False
    assert info['time_outs'] is False


def test_time_limit_does_not_clobber_inner_truncation():
    env = wrappers.TimeLimit(FakeSignalEnv(truncate_at=2), 5)
    check_reset(env)
    check_step(env, 0)
    _, _, terminated, truncated, info = check_step(env, 0)
    assert terminated is False
    assert truncated is True
    assert info['time_outs'] is True


def test_time_limit_resets_counter():
    env = wrappers.TimeLimit(FakeSignalEnv(), 2)
    check_reset(env)
    check_step(env, 0)
    _, _, _, truncated, _ = check_step(env, 0)
    assert truncated is True
    check_reset(env)
    _, _, _, truncated, _ = check_step(env, 0)
    assert truncated is False


# --------------------------------------------------------------------------
# 5/6. deepmind atari chain
# --------------------------------------------------------------------------

def test_wrap_deepmind_chain():
    base = FakeAtariEnv(n_lives=3, steps_per_life=5)
    env = wrappers.wrap_deepmind(base, episode_life=True, clip_rewards=False, frame_stack=True)
    obs, _ = check_reset(env)
    assert obs.shape == (84, 84, 4)
    for _ in range(20):
        obs, _, terminated, truncated, _ = check_step(env, 0)
        assert obs.shape == (84, 84, 4)
        if terminated or truncated:
            obs, _ = check_reset(env)
            assert obs.shape == (84, 84, 4)


def test_make_atari_deepmind_chain(monkeypatch):
    def fake_make(env_id, **kwargs):
        return FakeAtariEnv(n_lives=3, steps_per_life=5)

    monkeypatch.setattr(wrappers.gym, 'make', fake_make)
    env = wrappers.make_atari_deepmind('FakeNoFrameskip-v4', noop_max=4, skip=4)
    obs, _ = check_reset(env)
    assert obs.shape == (84, 84, 4)
    for _ in range(20):
        obs, _, terminated, truncated, _ = check_step(env, 0)
        assert obs.shape == (84, 84, 4)
        if terminated or truncated:
            check_reset(env)


def test_make_car_racing_chain(monkeypatch):
    def fake_make(env_id, **kwargs):
        return FakeAtariEnv(n_lives=3, steps_per_life=5)

    monkeypatch.setattr(wrappers.gym, 'make', fake_make)
    env = wrappers.make_car_racing('FakeCarRacing-v2', skip=4)
    obs, _ = check_reset(env)
    assert obs.shape == (84, 84, 4)
    check_step(env, 0)


# --------------------------------------------------------------------------
# 7. batched frame stacks (SMAC routes)
# --------------------------------------------------------------------------

def test_batched_frame_stack_flatten():
    base = FakeMultiAgentEnv(n_agents=3, obs_dim=5)
    env = wrappers.BatchedFrameStack(base, 4, transpose=False, flatten=True)
    obs, _ = check_reset(env)
    assert obs.shape == (3, 20)
    assert env.observation_space.shape == (20,)
    obs, reward, terminated, truncated, info = check_step(env, np.zeros(3, dtype=np.int64))
    assert obs.shape == (3, 20)
    assert terminated.shape == (3,)
    assert truncated.shape == (3,)


def test_batched_frame_stack_stacked():
    base = FakeMultiAgentEnv(n_agents=3, obs_dim=5)
    env = wrappers.BatchedFrameStack(base, 4, transpose=False, flatten=False)
    obs, _ = check_reset(env)
    assert obs.shape == (3, 4, 5)
    obs, _, _, _, _ = check_step(env, np.zeros(3, dtype=np.int64))
    assert obs.shape == (3, 4, 5)


def test_batched_frame_stack_transpose():
    base = FakeMultiAgentEnv(n_agents=3, obs_dim=5)
    env = wrappers.BatchedFrameStack(base, 4, transpose=True)
    obs, _ = check_reset(env)
    assert obs.shape == (3, 5, 4)
    obs, _, _, _, _ = check_step(env, np.zeros(3, dtype=np.int64))
    assert obs.shape == (3, 5, 4)


def test_batched_frame_stack_with_states():
    base = FakeMultiAgentEnv(n_agents=3, obs_dim=5, state_dim=7, with_states=True)
    env = wrappers.BatchedFrameStackWithStates(base, 4, transpose=False, flatten=True)
    obs, _ = check_reset(env)
    assert set(obs.keys()) == {'obs', 'state'}
    assert obs['obs'].shape == (3, 20)
    assert obs['state'].shape == (4, 7)
    obs, _, terminated, truncated, _ = check_step(env, np.zeros(3, dtype=np.int64))
    assert obs['obs'].shape == (3, 20)
    assert obs['state'].shape == (4, 7)
    assert terminated.shape == (3,)
    assert truncated.shape == (3,)


def test_batched_frame_stack_episode_end_flags():
    base = FakeMultiAgentEnv(n_agents=2, obs_dim=4, episode_len=2)
    env = wrappers.BatchedFrameStack(base, 2, flatten=True)
    check_reset(env)
    _, _, terminated, _, _ = check_step(env, np.zeros(2, dtype=np.int64))
    assert not terminated.any()
    _, _, terminated, _, _ = check_step(env, np.zeros(2, dtype=np.int64))
    assert terminated.all()


# --------------------------------------------------------------------------
# 8. multiwalker (optional dep) and dm_control retirement
# --------------------------------------------------------------------------

def test_multiwalker_env_contract():
    pytest.importorskip('pettingzoo')
    env = env_configurations.create_multiwalker_env()
    check_reset(env)
    check_step(env, np.zeros((3,) + env.action_space.shape, dtype=np.float32))


def test_dm_control_route_retired():
    assert 'dm_control' not in env_configurations.configurations
    assert not hasattr(env_configurations, 'create_dm_control_env')
    # the shipped dm_control configs go through the gymnasium/envpool routes
    assert 'gymnasium' in env_configurations.configurations


# --------------------------------------------------------------------------
# 9. no old-API signature may remain in wrappers.py
# --------------------------------------------------------------------------

def test_no_legacy_reset_signature_left():
    import inspect
    import re
    source = inspect.getsource(wrappers)
    bad = re.findall(r'def reset\(self\)\s*:', source)
    assert not bad, f'old-API reset() signatures left in wrappers.py: {bad}'
