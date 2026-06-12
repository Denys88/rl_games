"""Phase B SAC correctness tests (spec: docs/superpowers/specs/2026-06-11-phase-b-sac-correctness-design.md)."""
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
