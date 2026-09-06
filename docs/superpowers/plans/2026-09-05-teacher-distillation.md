# Teacher Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A frozen-teacher distillation loss inside rl_games PPO with weight schedules (DAgger / DAgger→PPO / teacher-guided RL as config variants), applied to a Craftax-Classic pixel student with a symbolic-state teacher.

**Architecture:** `TeacherDistillation` (pure torch helper) owns the frozen teacher, the three schedules, the KL/MSE loss, DAgger action mixing and the central-value warm start; the PPO agents call it from `get_action_values` and `calc_gradients`. `CraftaxVecEnv` steps Craftax-Classic in JAX and returns `{'obs': pixels, 'states': symbolic}` torch tensors via dlpack.

**Tech Stack:** PyTorch, rl_games `a2c_discrete`/`a2c_continuous`, JAX + craftax 1.6 (`venv/`, python 3.11, `jax[cuda12]`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-teacher-distillation-design.md`

## Global Constraints

- Craftax lives in the 3.11 venv: run its tests and training with `venv/bin/python`. Distillation unit tests are pure torch and run in either venv.
- Observation dict convention: `{'obs': student input, 'states': teacher / central-value input}`; `get_env_info` returns `observation_space` (obs) and `state_space` (states).
- Schedules are `[[epoch, value], ...]` lists, piecewise-linear, constant outside the range.
- Commit trailer as used on this branch.

---

### Task 1: `TeacherDistillation` core (schedules, losses, mixing, warm start)

**Files:**
- Create: `rl_games/common/distillation.py`
- Test: `tests/test_distillation.py`

**Interfaces:**
- Produces: `piecewise_linear(spec, epoch) -> float`; class `TeacherDistillation(config, teacher_model, is_discrete, device)` with `coef(epoch)`, `ppo_coef(epoch)`, `beta(epoch)`, `loss(res_dict, states, rnn_masks=None) -> tensor`, `mix_actions(res_dict, states, epoch) -> res_dict`, `warm_start_central_value(cv_model) -> (copied, skipped)`, classmethod `from_config(config, state_shape, actions_num, device)` building the teacher from `teacher_config` + `teacher_checkpoint`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distillation.py
import numpy as np
import pytest
import torch


def test_piecewise_linear_schedule():
    from rl_games.common.distillation import piecewise_linear
    spec = [[0, 1.0], [10, 1.0], [20, 0.0]]
    assert piecewise_linear(spec, -5) == 1.0
    assert piecewise_linear(spec, 5) == 1.0
    assert abs(piecewise_linear(spec, 15) - 0.5) < 1e-9
    assert piecewise_linear(spec, 30) == 0.0
    assert piecewise_linear(0.3, 7) == 0.3            # scalar spec = constant
    assert piecewise_linear(None, 7) == 0.0


class _Teacher(torch.nn.Module):
    """Stand-in teacher: logits = W states (discrete) or mus = W states (continuous)."""

    def __init__(self, in_dim, out_dim, discrete=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.discrete = discrete

    def forward(self, d):
        h = self.lin(d['obs'])
        if self.discrete:
            return {'logits': h}
        return {'mus': h, 'sigmas': torch.full_like(h, 0.5)}


def _distill(teacher, discrete, **cfg):
    from rl_games.common.distillation import TeacherDistillation
    base = {'coef': 1.0, 'ppo_coef': 1.0, 'beta': 0.0, 'loss': 'kl'}
    base.update(cfg)
    return TeacherDistillation(base, teacher, is_discrete=discrete, device='cpu')


def test_discrete_kl_matches_manual_and_is_zero_for_identical():
    t = _Teacher(4, 5)
    d = _distill(t, True)
    states = torch.randn(16, 4)
    student_logits = torch.randn(16, 5)
    with torch.no_grad():
        p = torch.softmax(t.lin(states), -1)
    manual = (p * (torch.log(p) - torch.log_softmax(student_logits, -1))).sum(-1).mean()
    got = d.loss({'logits': student_logits}, states)
    assert torch.allclose(got, manual, atol=1e-6)
    with torch.no_grad():
        same = d.loss({'logits': t.lin(states)}, states)
    assert same.abs() < 1e-6
    masks = torch.zeros(16, 1); masks[:4] = 1
    masked = d.loss({'logits': student_logits}, states, rnn_masks=masks)
    manual_m = (p * (torch.log(p) - torch.log_softmax(student_logits, -1))).sum(-1)[:4].mean()
    assert torch.allclose(masked, manual_m, atol=1e-6)


def test_gaussian_kl_and_mse():
    t = _Teacher(4, 3, discrete=False)
    states = torch.randn(8, 4)
    mus = torch.randn(8, 3); sig = torch.full((8, 3), 0.8)
    with torch.no_grad():
        tm = t.lin(states); ts = torch.full_like(tm, 0.5)
    kl = (torch.log(sig / ts) + (ts ** 2 + (tm - mus) ** 2) / (2 * sig ** 2) - 0.5).sum(-1).mean()
    assert torch.allclose(_distill(t, False).loss({'mus': mus, 'sigmas': sig}, states), kl, atol=1e-6)
    mse = ((tm - mus) ** 2).sum(-1).mean()
    assert torch.allclose(_distill(t, False, loss='mse').loss({'mus': mus, 'sigmas': sig}, states), mse, atol=1e-6)


def test_mix_actions_substitutes_beta_rows_with_valid_neglogp():
    torch.manual_seed(1)
    t = _Teacher(4, 5)
    d = _distill(t, True, beta=[[0, 1.0], [10, 0.0]])
    N = 2000
    states = torch.randn(N, 4)
    logits = torch.randn(N, 5)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    res = {'logits': logits, 'actions': actions.clone(), 'neglogpacs': -dist.log_prob(actions)}
    out = d.mix_actions(res, states, epoch=5)                       # beta 0.5
    changed = (out['actions'] != actions).float().mean().item()
    assert 0.25 < changed < 0.55                                     # ~half the rows re-drawn from the teacher
    assert torch.allclose(out['neglogpacs'], -dist.log_prob(out['actions']), atol=1e-5)
    untouched = d.mix_actions(dict(res), states, epoch=20)           # beta 0
    assert torch.equal(untouched['actions'], actions)
    # continuous: substituted rows carry the student's Gaussian neglogp
    tc = _Teacher(4, 3, discrete=False)
    dc = _distill(tc, False, beta=1.0)
    mus = torch.randn(N, 3); sig = torch.full((N, 3), 0.7)
    resc = {'mus': mus, 'sigmas': sig, 'actions': mus.clone(), 'neglogpacs': torch.zeros(N)}
    outc = dc.mix_actions(resc, states, epoch=0)
    ref = -torch.distributions.Normal(mus, sig).log_prob(outc['actions']).sum(-1)
    assert torch.allclose(outc['neglogpacs'], ref, atol=1e-4)
    assert not torch.allclose(outc['actions'], mus)


def test_warm_start_copies_matching_tensors_only():
    teacher = torch.nn.ModuleDict({'trunk': torch.nn.Linear(4, 8), 'value': torch.nn.Linear(8, 1),
                                   'logits': torch.nn.Linear(8, 5)})
    cv = torch.nn.ModuleDict({'trunk': torch.nn.Linear(4, 8), 'value': torch.nn.Linear(8, 1),
                              'extra': torch.nn.Linear(2, 2)})
    d = _distill(teacher, True)
    copied, skipped = d.warm_start_central_value(cv)
    assert copied == 4 and skipped == 2                           # extra.* skipped
    assert torch.equal(cv['trunk'].weight, teacher['trunk'].weight)
    assert torch.equal(cv['value'].bias, teacher['value'].bias)
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv312/bin/python -m pytest tests/test_distillation.py -q`
Expected: FAIL, `ModuleNotFoundError: rl_games.common.distillation`

- [ ] **Step 3: Implement**

```python
# rl_games/common/distillation.py
"""Frozen-teacher distillation inside PPO (DAgger as a loss term).

The teacher is a frozen rl_games policy that consumes privileged `states`;
the student consumes `obs`. Three piecewise-linear schedules over epochs:

  coef      weight of the distillation loss on student-visited states
  ppo_coef  weight of the PPO loss (0 -> pure DAgger phase)
  beta      P(env action taken from the teacher) — classic DAgger mixing;
            the student's neglogp is recomputed for substituted rows so PPO
            importance ratios stay valid.

Config (algo `config.distillation`):

    distillation:
      teacher_config: <yaml with the teacher's params>
      teacher_checkpoint: <rl_games .pth>
      loss: kl                 # discrete: KL(teacher||student); continuous: kl | mse
      coef: [[0, 1.0], [400, 1.0], [800, 0.0]]
      ppo_coef: [[0, 0.0], [400, 0.0], [401, 1.0]]
      beta: [[0, 0.5], [200, 0.0]]
      warm_start_value: True   # copy matching teacher tensors into the central value net
"""

import copy

import numpy as np
import torch
import yaml


def piecewise_linear(spec, epoch):
    """[[epoch, value], ...] -> value at `epoch` (linear between knots,
    constant outside). A scalar spec is a constant; None is 0."""
    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)):
        return float(spec)
    knots = sorted((float(e), float(v)) for e, v in spec)
    if epoch <= knots[0][0]:
        return knots[0][1]
    for (e0, v0), (e1, v1) in zip(knots[:-1], knots[1:]):
        if e0 <= epoch <= e1:
            if e1 == e0:
                return v1
            return v0 + (v1 - v0) * (epoch - e0) / (e1 - e0)
    return knots[-1][1]


def _strip_prefix(sd, prefix='_orig_mod.'):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


class TeacherDistillation:
    def __init__(self, config, teacher_model, is_discrete, device):
        self.config = config
        self.teacher = teacher_model.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.is_discrete = is_discrete
        self.device = device
        self.loss_type = config.get('loss', 'kl')
        if self.loss_type not in ('kl', 'mse'):
            raise ValueError("distillation loss must be 'kl' or 'mse'")
        self._coef = config.get('coef', 1.0)
        self._ppo_coef = config.get('ppo_coef', 1.0)
        self._beta = config.get('beta', 0.0)
        self.warm_start_value = bool(config.get('warm_start_value', True))
        self.last_loss = None

    # ----------------------------------------------------------- schedules

    def coef(self, epoch):
        return piecewise_linear(self._coef, epoch)

    def ppo_coef(self, epoch):
        return piecewise_linear(self._ppo_coef, epoch)

    def beta(self, epoch):
        return piecewise_linear(self._beta, epoch)

    # ------------------------------------------------------------- teacher

    @torch.no_grad()
    def teacher_forward(self, states):
        return self.teacher({'obs': states, 'is_train': False})

    @classmethod
    def from_config(cls, config, state_shape, actions_num, device):
        """Build the frozen teacher from its own yaml + checkpoint."""
        from rl_games.algos_torch.model_builder import ModelBuilder
        params = yaml.safe_load(open(config['teacher_config']))['params']
        net = ModelBuilder().load(params)
        tcfg = params['config']
        model = net.build({'actions_num': actions_num, 'input_shape': tuple(state_shape), 'num_seqs': 1,
                           'value_size': tcfg.get('value_size', 1),
                           'normalize_value': tcfg.get('normalize_value', False),
                           'normalize_input': tcfg.get('normalize_input', False)})
        ckpt = torch.load(config['teacher_checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(_strip_prefix(ckpt['model']))
        is_discrete = params['algo']['name'] == 'a2c_discrete'
        print(f"[Distillation] teacher {config['teacher_checkpoint']} (epoch {ckpt.get('epoch', '?')}), "
              f"{'discrete' if is_discrete else 'continuous'}, coef {config.get('coef')}, "
              f"ppo_coef {config.get('ppo_coef')}, beta {config.get('beta')}")
        return cls(config, model, is_discrete, device)

    # ---------------------------------------------------------------- loss

    def loss(self, res_dict, states, rnn_masks=None):
        """Mean distillation loss over rows (masked when rnn_masks given)."""
        t = self.teacher_forward(states)
        if self.is_discrete:
            logp_t = torch.log_softmax(t['logits'].float(), dim=-1)
            logp_s = torch.log_softmax(res_dict['logits'].float(), dim=-1)
            per_row = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)
        else:
            mu_s, sig_s = res_dict['mus'].float(), res_dict['sigmas'].float()
            mu_t, sig_t = t['mus'].float(), t['sigmas'].float()
            if self.loss_type == 'mse':
                per_row = ((mu_t - mu_s) ** 2).sum(dim=-1)
            else:
                per_row = (torch.log(sig_s / sig_t) + (sig_t ** 2 + (mu_t - mu_s) ** 2) / (2 * sig_s ** 2) - 0.5).sum(dim=-1)
        if rnn_masks is not None:
            m = rnn_masks.reshape(-1).float()
            out = (per_row * m).sum() / m.sum().clamp(min=1.0)
        else:
            out = per_row.mean()
        self.last_loss = out.detach()
        return out

    # ------------------------------------------------------------- mixing

    def mix_actions(self, res_dict, states, epoch):
        """DAgger beta-mixing on the rollout batch: substitute the teacher's
        sampled action for a Bernoulli(beta) subset of rows and recompute the
        student's neglogp for those rows."""
        beta = self.beta(epoch)
        if beta <= 0.0:
            return res_dict
        n = states.shape[0]
        sel = torch.rand(n, device=states.device) < beta
        if not sel.any():
            return res_dict
        t = self.teacher_forward(states[sel])
        actions = res_dict['actions'].clone()
        if self.is_discrete:
            ta = torch.distributions.Categorical(logits=t['logits'].float()).sample()
            actions[sel] = ta.to(actions.dtype)
            neglogp = -torch.log_softmax(res_dict['logits'].float(), dim=-1).gather(1, actions.view(-1, 1).long()).squeeze(1)
        else:
            ta = torch.distributions.Normal(t['mus'].float(), t['sigmas'].float()).sample()
            actions[sel] = ta.to(actions.dtype)
            mus, sig = res_dict['mus'].float(), res_dict['sigmas'].float()
            neglogp = 0.5 * (((actions.float() - mus) / sig) ** 2).sum(-1) \
                + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] + torch.log(sig).sum(-1)
        res_dict['actions'] = actions
        res_dict['neglogpacs'] = neglogp.to(res_dict['neglogpacs'].dtype)
        return res_dict

    # ---------------------------------------------------------- warm start

    def warm_start_central_value(self, cv_model):
        """Copy every teacher tensor whose name and shape match into the
        central-value model (trunk, value head, input/value normalisers)."""
        tsd = _strip_prefix(self.teacher.state_dict())
        csd = cv_model.state_dict()
        new, copied, skipped = {}, 0, 0
        for k, v in csd.items():
            if k in tsd and tuple(tsd[k].shape) == tuple(v.shape):
                new[k] = tsd[k].to(v.device, v.dtype)
                copied += 1
            else:
                skipped += 1
        cv_model.load_state_dict(new, strict=False)
        print(f'[Distillation] warm-started central value: {copied} tensors copied, {skipped} skipped')
        return copied, skipped

    # ------------------------------------------------------------- logging

    def log(self, writer, epoch, frame):
        if writer is None:
            return
        writer.add_scalar('info/distill_coef', self.coef(epoch), frame)
        writer.add_scalar('info/ppo_coef', self.ppo_coef(epoch), frame)
        writer.add_scalar('info/teacher_beta', self.beta(epoch), frame)
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv312/bin/python -m pytest tests/test_distillation.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rl_games/common/distillation.py tests/test_distillation.py
git commit -m "TeacherDistillation: schedules, KL/MSE loss, DAgger mixing, central-value warm start"
```

---

### Task 2: Craftax vec env + observer

**Files:**
- Create: `rl_games/envs/craftax_vecenv.py`
- Modify: `rl_games/common/vecenv.py` (register `CRAFTAX`), `rl_games/common/env_configurations.py` (env `craftax`)
- Test: `tests/test_craftax_env.py`

**Interfaces:**
- Produces: `CraftaxVecEnv(config_name, num_actors, game='Craftax-Classic-v1', obs='both', seed=0, device='cuda')`; `reset() -> obs`, `step(actions) -> (obs, rewards, dones, infos)` with torch tensors on `device`; `obs` is a dict `{'obs', 'states'}` for `obs='both'`, a tensor otherwise; `get_env_info()` with `observation_space`, `state_space` (both modes), `action_space` Discrete(17), `agents` 1; infos on any done: `achievements` (N, 22) float, `done_mask` (N,) bool. `CraftaxObserver` (same file) logs `craftax/score` and `craftax/ach_<name>`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_craftax_env.py
import numpy as np
import pytest
import torch


def _has_craftax():
    try:
        import craftax, jax  # noqa
        return True
    except Exception:
        return False


needs_craftax = pytest.mark.skipif(not _has_craftax(), reason='craftax/jax unavailable')


def _make(n=8, **kw):
    from rl_games.envs.craftax_vecenv import CraftaxVecEnv
    cfg = dict(game='Craftax-Classic-v1', obs='both', seed=1, device='cpu')
    cfg.update(kw)
    return CraftaxVecEnv('craftax', n, **cfg)


@needs_craftax
def test_both_mode_shapes_and_env_info():
    env = _make(8)
    info = env.get_env_info()
    assert info['observation_space'].shape == (63, 63, 3) and info['observation_space'].dtype == np.uint8
    assert info['state_space'].shape == (1345,)
    assert info['action_space'].n == 17 and info['agents'] == 1
    obs = env.reset()
    assert obs['obs'].shape == (8, 63, 63, 3) and obs['obs'].dtype == torch.uint8
    assert obs['states'].shape == (8, 1345) and obs['states'].dtype == torch.float32
    assert int(obs['obs'].max()) > 0
    o, r, d, inf = env.step(torch.randint(0, 17, (8,)))
    assert o['obs'].shape == (8, 63, 63, 3) and r.shape == (8,) and d.shape == (8,) and d.dtype == torch.bool
    assert r.dtype == torch.float32


@needs_craftax
def test_single_modes():
    env = _make(4, obs='symbolic')
    assert env.get_env_info()['observation_space'].shape == (1345,)
    o = env.reset()
    assert torch.is_tensor(o) and o.shape == (4, 1345)
    env = _make(4, obs='pixels')
    assert env.get_env_info()['observation_space'].shape == (63, 63, 3)
    o = env.reset()
    assert torch.is_tensor(o) and o.shape == (4, 63, 63, 3) and o.dtype == torch.uint8


@needs_craftax
def test_autoreset_and_achievements_info():
    env = _make(4, max_timesteps=30)          # short episodes -> dones within 30 steps
    env.reset()
    seen = False
    for t in range(40):
        o, r, d, info = env.step(torch.randint(0, 17, (4,)))
        assert o['obs'].shape == (4, 63, 63, 3)
        if d.any():
            seen = True
            assert info['achievements'].shape == (4, 22)
            assert torch.equal(info['done_mask'], d)
    assert seen


def test_craftax_observer_score():
    from rl_games.envs.craftax_vecenv import CraftaxObserver, crafter_score
    rates = np.zeros(22); rates[:2] = 1.0            # two achievements always, rest never
    expected = np.exp(np.mean(np.log(1 + rates * 100))) - 1
    assert abs(crafter_score(rates) - expected) < 1e-9

    class _Algo:
        writer = None
        num_agents = 1
    obs = CraftaxObserver()
    obs.after_init(_Algo())
    ach = torch.zeros(4, 22); ach[0, 3] = 1; ach[1, 3] = 1
    obs.process_infos({'achievements': ach, 'done_mask': torch.tensor([True, True, False, False])}, torch.tensor([[0], [1]]))
    assert len(obs.episodes) == 2 and abs(obs.rates()[3] - 1.0) < 1e-9 and obs.rates()[0] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_craftax_env.py -q`
Expected: 4 failed, `ModuleNotFoundError: rl_games.envs.craftax_vecenv`

- [ ] **Step 3: Implement**

```python
# rl_games/envs/craftax_vecenv.py
"""Craftax (Crafter in JAX) for rl_games, with privileged symbolic states.

Craftax renders one game state two ways: a 1345-d symbolic vector and a
63x63x3 pixel image (Craftax-Classic). This env returns

  obs: symbolic  -> float32 (N, 1345)
  obs: pixels    -> uint8   (N, 63, 63, 3)
  obs: both      -> {'obs': uint8 pixels, 'states': float32 symbolic}

so a symbolic teacher and a pixel student see the same states (see
rl_games/common/distillation.py). Steps run under jax.jit with craftax's
auto-reset (same-step: the obs returned on done is the new episode's first
obs); tensors reach torch via dlpack. ~90k env-steps/s for 1024 envs on GPU.

Infos on any done: `achievements` (N, 22) float32 of the finished episodes
(rows of unfinished envs are stale and must be masked with `done_mask`).
CraftaxObserver logs achievement rates and the Crafter score.
"""

import gymnasium as gym
import numpy as np
import torch

from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.ivecenv import IVecEnv

ACHIEVEMENTS = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron', 'collect_sapling', 'collect_stone',
    'collect_wood', 'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword', 'make_wood_pickaxe', 'make_wood_sword',
    'place_furnace', 'place_plant', 'place_stone', 'place_table', 'wake_up',
]


def crafter_score(rates):
    """Crafter score: geometric mean of achievement success rates (in %)."""
    rates = np.asarray(rates, dtype=np.float64)
    return float(np.exp(np.mean(np.log(1.0 + rates * 100.0))) - 1.0)


class CraftaxVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import jax
        import jax.numpy as jnp
        from craftax.craftax_env import make_craftax_env_from_name

        self.num_envs = int(num_actors)
        self.game = kwargs.pop('game', 'Craftax-Classic-v1')
        self.obs_mode = kwargs.pop('obs', 'both')
        if self.obs_mode not in ('symbolic', 'pixels', 'both'):
            raise ValueError(f'obs must be symbolic | pixels | both, got {self.obs_mode!r}')
        self.device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        seed = int(kwargs.pop('seed', 0))
        if self.game != 'Craftax-Classic-v1':
            raise NotImplementedError('only Craftax-Classic-v1 is wired up (pixel renderer + 1345-d symbolic)')
        from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer, render_craftax_symbolic

        self._jax, self._jnp = jax, jnp
        env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=True)
        params = env.default_params
        if 'max_timesteps' in kwargs:
            params = params.replace(max_timesteps=int(kwargs.pop('max_timesteps')))
        self.params = params
        self.env = env
        self.actions_num = env.action_space(params).n
        render_pixels = make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)
        want_pix = self.obs_mode in ('pixels', 'both')
        want_sym = self.obs_mode in ('symbolic', 'both')
        ach_keys = ['Achievements/' + a for a in ACHIEVEMENTS]

        def _outputs(state):
            pix = render_pixels(state).astype(jnp.uint8) if want_pix else None
            sym = render_craftax_symbolic(state) if want_sym else None
            return pix, sym

        def _reset(key):
            keys = jax.random.split(key, self.num_envs)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
            pix, sym = jax.vmap(_outputs)(state)
            return state, pix, sym

        def _step(key, state, actions):
            keys = jax.random.split(key, self.num_envs)
            _, state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, actions, params)
            pix, sym = jax.vmap(_outputs)(state)
            ach = jnp.stack([info[k] for k in ach_keys], axis=1).astype(jnp.float32)
            return state, pix, sym, reward.astype(jnp.float32), done, ach

        self._reset_fn = jax.jit(_reset)
        self._step_fn = jax.jit(_step)
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self._dlpack_ok = True

        self.pixel_space = gym.spaces.Box(0, 255, shape=(63, 63, 3), dtype=np.uint8)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(1345,), dtype=np.float32)
        self.observation_space = self.pixel_space if want_pix else self.state_space
        self.action_space = gym.spaces.Discrete(self.actions_num)

    # ------------------------------------------------------------ bridging

    def _to_torch(self, x):
        if x is None:
            return None
        if self._dlpack_ok:
            try:
                return torch.from_dlpack(x).to(self.device)
            except Exception:
                self._dlpack_ok = False
        return torch.as_tensor(np.asarray(x)).to(self.device)

    def _to_jax(self, t):
        t = t.detach().to(torch.int32).contiguous()
        try:
            return self._jax.dlpack.from_dlpack(t)
        except Exception:
            return self._jnp.asarray(t.cpu().numpy())

    def _pack(self, pix, sym):
        if self.obs_mode == 'both':
            return {'obs': self._to_torch(pix), 'states': self._to_torch(sym)}
        return self._to_torch(pix if self.obs_mode == 'pixels' else sym)

    def _next_key(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    # ------------------------------------------------------------- IVecEnv

    def reset(self):
        self._state, pix, sym = self._reset_fn(self._next_key())
        return self._pack(pix, sym)

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        acts = self._to_jax(actions.reshape(self.num_envs))
        self._state, pix, sym, reward, done, ach = self._step_fn(self._next_key(), self._state, acts)
        rewards = self._to_torch(reward)
        dones = self._to_torch(done).bool()
        infos = {}
        if bool(dones.any()):
            infos['achievements'] = self._to_torch(ach)
            infos['done_mask'] = dones
        return self._pack(pix, sym), rewards, dones, infos

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {'observation_space': self.observation_space, 'state_space': self.state_space,
                'action_space': self.action_space, 'agents': 1, 'value_size': 1}


class CraftaxObserver(AlgoObserver):
    """Logs craftax/score (Crafter score) and craftax/ach_<name> success rates
    over the last `window` finished episodes."""

    def __init__(self, window=1000):
        super().__init__()
        self.window = window
        self.episodes = []
        self.writer = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = getattr(algo, 'writer', None)

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or 'achievements' not in infos:
            return
        ach = infos['achievements'][infos['done_mask']].cpu().numpy()
        self.episodes.extend(list(ach))
        self.episodes = self.episodes[-self.window:]

    def rates(self):
        return np.mean(np.stack(self.episodes), axis=0) if self.episodes else np.zeros(len(ACHIEVEMENTS))

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None or not self.episodes:
            return
        r = self.rates()
        self.writer.add_scalar('craftax/score', crafter_score(r), frame)
        self.writer.add_scalar('craftax/episodes_in_window', len(self.episodes), frame)
        for name, v in zip(ACHIEVEMENTS, r):
            self.writer.add_scalar(f'craftax/ach_{name}', float(v), frame)


def create_craftax(**kwargs):
    return CraftaxVecEnv(kwargs.pop('config_name', 'craftax'), kwargs.pop('num_actors', 1024), **kwargs)
```

`rl_games/common/vecenv.py` (append):

```python
def _create_craftax(config_name, num_actors, **kwargs):
    from rl_games.envs.craftax_vecenv import CraftaxVecEnv
    return CraftaxVecEnv(config_name, num_actors, **kwargs)
register('CRAFTAX', _create_craftax)
```

`rl_games/common/env_configurations.py` (in `configurations`):

```python
    'craftax' : {
        'vecenv_type': 'CRAFTAX'
    },
```

- [ ] **Step 4: Run to verify**

Run: `venv/bin/python -m pytest tests/test_craftax_env.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rl_games/envs/craftax_vecenv.py rl_games/common/vecenv.py rl_games/common/env_configurations.py tests/test_craftax_env.py
git commit -m "CraftaxVecEnv: Craftax-Classic with symbolic states + pixel obs, CraftaxObserver"
```

---

### Task 3: Agent hooks (states plumbing, mixing, loss) + integration test

**Files:**
- Modify: `rl_games/common/a2c_common.py`, `rl_games/common/experience.py`, `rl_games/algos_torch/a2c_discrete.py`, `rl_games/algos_torch/a2c_continuous.py`
- Test: `tests/test_craftax_env.py` (append integration test)

**Interfaces:**
- Consumes: `TeacherDistillation.from_config`, `mix_actions`, `loss`, `coef`, `ppo_coef`, `warm_start_central_value`, `log`.
- Produces: agent attributes `self.distill`, `self.store_states`; dataset key `states`; tensorboard `losses/distill`, `info/distill_coef`, `info/ppo_coef`, `info/teacher_beta`.

- [ ] **Step 1: Write the failing integration test**

```python
# append to tests/test_craftax_env.py

@needs_craftax
def test_distillation_end_to_end_two_epochs(tmp_path):
    """Symbolic teacher (untrained, saved to disk) -> pixel student with the
    distillation block; two PPO epochs must run and log losses/distill."""
    import yaml
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch.model_builder import ModelBuilder
    tcfg = yaml.safe_load(open('rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml'))
    net = ModelBuilder().load(tcfg['params'])
    c = tcfg['params']['config']
    teacher = net.build({'actions_num': 17, 'input_shape': (1345,), 'num_seqs': 1, 'value_size': 1,
                         'normalize_value': c['normalize_value'], 'normalize_input': c['normalize_input']})
    ck = tmp_path / 'teacher.pth'
    torch.save({'model': teacher.state_dict(), 'epoch': 0}, ck)
    scfg = yaml.safe_load(open('rl_games/configs/craftax/ppo_craftax_classic_pixels_distill.yaml'))
    sc = scfg['params']['config']
    sc.update(num_actors=16, horizon_length=8, minibatch_size=64, max_epochs=2, save_frequency=0,
              train_dir=str(tmp_path), name='distill_smoke', device='cpu')
    sc['central_value_config']['minibatch_size'] = 64
    sc['distillation'].update(teacher_checkpoint=str(ck), beta=0.5)
    sc['env_config'].update(device='cpu')
    runner = Runner()
    runner.load(scfg)
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    assert agent.distill is not None and agent.store_states
    agent.train()
    assert 'distill' in agent.aux_loss_dict
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_craftax_env.py -q -k end_to_end`
Expected: FAIL (config file missing or `AttributeError: distill`)

- [ ] **Step 3: Implement the hooks**

`rl_games/common/experience.py` line ~340: `self.has_central_value = algo_info.get('store_states', algo_info['has_central_value'])` (the flag only gates the `states` tensor allocation).

`rl_games/common/a2c_common.py`:

```python
        # (after self.has_central_value / self.state_space block, ~line 312)
        self.distill_config = self.config.get('distillation', None)
        self.distill = None                     # built by the agent after the model exists
        self.store_states = self.has_central_value or self.distill_config is not None
        if self.store_states and not hasattr(self, 'state_space'):
            self.state_space = self.env_info.get('state_space', None) or self.observation_space
```

`init_tensors` algo_info: add `'store_states': self.store_states`.

`play_steps` / `play_steps_rnn`: the two `if self.has_central_value: self.experience_buffer.update_data('states', ...)` become `if self.store_states:`.

`get_action_values`: after `res_dict = self.inference_model()(input_dict)` (before the central-value block):

```python
            if self.distill is not None:
                res_dict = self.distill.mix_actions(res_dict, obs['states'], self.epoch_num)
```

`prepare_dataset` (main dataset): after the `rollout_target_keys` loop:

```python
        if self.distill is not None:
            dataset_dict['states'] = batch_dict['states']
```

Epoch logging (next to `info/last_lr`, line ~643):

```python
        if self.distill is not None:
            self.distill.log(self.writer, self.epoch_num, frame)
```

Both agents, in `__init__` after the central value net is created (end of the `if self.has_central_value:` block):

```python
        if self.distill_config is not None:
            from rl_games.common.distillation import TeacherDistillation
            self.distill = TeacherDistillation.from_config(
                self.distill_config, self.state_space.shape, self.actions_num, self.ppo_device)
            if self.has_central_value and self.distill.warm_start_value:
                self.distill.warm_start_central_value(self.central_value_net.model)
```

Both agents, in `calc_gradients`, right after `loss` is computed (both fused and regular branches end before the `aux_loss` block):

```python
            if self.distill is not None:
                d_loss = self.distill.loss(res_dict, input_dict['states'], rnn_masks)
                loss = self.distill.ppo_coef(self.epoch_num) * loss + self.distill.coef(self.epoch_num) * d_loss
            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if self.distill is not None:
                self.aux_loss_dict['distill'] = [d_loss.detach()]
```

(`a2c_discrete.py` has the same structure; `actions_num` is `self.actions_num` in both agents.)

- [ ] **Step 4: Configs needed by the test** — create the three configs from Task 4 now (they are plain data), then run:

Run: `venv/bin/python -m pytest tests/test_craftax_env.py tests/test_distillation.py -q`
Expected: 10 passed

Run the soccer + league regression to prove the plumbing change is inert elsewhere:
`venv312/bin/python -m pytest tests/test_envpool_soccer.py tests/test_population_network.py -q` → 24 passed.

- [ ] **Step 5: Commit**

```bash
git add rl_games/common/a2c_common.py rl_games/common/experience.py rl_games/algos_torch/a2c_discrete.py rl_games/algos_torch/a2c_continuous.py rl_games/configs/craftax tests/test_craftax_env.py
git commit -m "PPO: frozen-teacher distillation hooks (states in dataset, DAgger mixing, scheduled loss, CV warm start)"
```

---

### Task 4: Configs and teacher training

**Files:**
- Create: `rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml`, `ppo_craftax_classic_pixels.yaml`, `ppo_craftax_classic_pixels_distill.yaml`
- Create: `scripts/craftax_train.py` (attaches `CraftaxObserver`)

Teacher config:

```yaml
params:
  seed: 7
  algo: {name: a2c_discrete}
  model: {name: discrete_a2c}
  network:
    name: actor_critic
    separate: False
    space: {discrete: {}}
    mlp: {units: [512, 512, 256], activation: elu, initializer: {name: default}}
  config:
    name: craftax_classic_symbolic
    env_name: craftax
    device: cuda
    reward_shaper: {scale_value: 1.0}
    normalize_advantage: True
    normalize_input: True
    normalize_value: True
    value_bootstrap: False
    gamma: 0.99
    tau: 0.95
    learning_rate: 3.0e-4
    lr_schedule: adaptive
    kl_threshold: 0.01
    grad_norm: 1.0
    truncate_grads: True
    e_clip: 0.2
    entropy_coef: 0.01
    critic_coef: 1.0
    clip_value: True
    num_actors: 1024
    horizon_length: 64
    minibatch_size: 16384
    mini_epochs: 4
    max_epochs: 1500          # ~100M frames
    save_best_after: 50
    save_frequency: 250
    score_to_win: 100000
    games_to_track: 500
    env_config: {game: Craftax-Classic-v1, obs: symbolic, seed: 7}
```

Student scratch config: same but `network` = cnn (32 k8 s4, 64 k4 s2, 64 k3 s1, `permute_input: True`, `normalize_input: False`) + mlp [512], `env_config.obs: pixels`, `minibatch_size: 8192`, `name: craftax_classic_pixels`.

Distill config: student scratch + `env_config.obs: both`, `central_value_config` (network = teacher's MLP with `central_value: True`, `normalize_input: True`, minibatch 16384, mini_epochs 4, lr 3e-4), and

```yaml
    distillation:
      teacher_config: rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml
      teacher_checkpoint: runs/craftax_teacher/nn/craftax_classic_symbolic.pth
      loss: kl
      coef: [[0, 1.0], [300, 1.0], [600, 0.0]]
      ppo_coef: [[0, 0.0], [300, 0.0], [301, 1.0]]
      beta: [[0, 0.5], [150, 0.0]]
      warm_start_value: True
```

`scripts/craftax_train.py`: like `soccer_train.py` — args `-f`, `-c`, `--max-epochs`, `--name`, `--train-dir`; `Runner(CraftaxObserver())`.

Then: `venv/bin/python -u scripts/craftax_train.py -f rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml > craftax_teacher.log 2>&1` in the background; check fps and `craftax/score` after ~100 epochs.

- [ ] Commit configs + script.

### Task 5: Student runs and write-up

- Student scratch and student distill, same `max_epochs`, launched after the teacher finishes; compare `craftax/score`, achievement rates and episode reward at equal frames; write `docs/DISTILLATION.md` with the mechanism, the configs, and the numbers.
