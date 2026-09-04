# Soccer Population League Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train N=8 independent MLP soccer policies in one rl_games process, every match pairing two of them, both teams learning.

**Architecture:** A `population_actor_critic` network keeps all N policies as stacked `(N, ...)` parameters and routes each row to its slot (read from a one-hot in the observation) with padded batched matmuls, so the standard PPO agent trains the whole population in one update. `EnvpoolSoccerVecEnv` gains a `population` mode where all four robots of a match are learner rows and each match carries a (home slot, away slot) pair. A `SoccerPopulationObserver` keeps the N×N payoff matrix, resamples pairings and writes per-slot standalone checkpoints.

**Tech Stack:** PyTorch (`torch.baddbmm`), rl_games `a2c_continuous`, envpool `DmcSoccerBoxhead-v1` (py3.12 `venv312`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-soccer-population-league-design.md`

## Global Constraints

- Run every test with `venv312/bin/python -m pytest ...` (envpool soccer only exists in the 3.12 venv). Tests that touch the env are marked `@needs_soccer`.
- Observation layout in population mode: `[base envpool obs (111)] [player-id one-hot (team_size)] [slot one-hot (N)]`; the slot one-hot is always the LAST N dims.
- Player rows are env-major, home team first: `[home0, home1, away0, away1]`.
- Standalone slot checkpoints must load into the standard `actor_critic` model of `ppo_soccer_boxhead_league_v9.yaml` (obs 113 = 111 + 2) via `scripts/soccer_play.py`'s `load_model` (`torch.load(path)['model']`).
- Commit messages end with the Co-Authored-By / Claude-Session trailer used in this branch.

---

### Task 1: Population network — forward parity with per-slot MLPs

**Files:**
- Create: `rl_games/algos_torch/population_network.py`
- Test: `tests/test_population_network.py`

**Interfaces:**
- Produces: `PopulationBuilder` (rl_games `NetworkBuilder`), `PopulationBuilder.Network(params, actions_num=, input_shape=, value_size=, num_seqs=)` whose `forward(obs_dict) -> (mu (B,A), logstd (B,A), value (B,1), None)`; parameters `a2c_network.weights.{i}` `(N, in, out)`, `a2c_network.biases.{i}` `(N, out)`, `a2c_network.mu_w (N, h, A)`, `a2c_network.mu_b (N, A)`, `a2c_network.value_w (N, h, V)`, `a2c_network.value_b (N, V)`, `a2c_network.sigma (N, A)`; helper `slot_of(obs, N) -> LongTensor (B,)`.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_population_network.py
import numpy as np
import pytest
import torch


def _params(N=3, units=(16, 8), seed=5):
    return {'population_size': N, 'seed': seed,
            'mlp': {'units': list(units), 'activation': 'elu', 'initializer': {'name': 'default'}},
            'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None',
                                     'mu_init': {'name': 'default'},
                                     'sigma_init': {'name': 'const_initializer', 'val': 0},
                                     'fixed_sigma': True}}}


def _build(N=3, D=6, A=2, units=(16, 8), seed=5):
    from rl_games.algos_torch.population_network import PopulationBuilder
    b = PopulationBuilder()
    b.load(_params(N, units, seed))
    return b.build('pop', actions_num=A, input_shape=(D + N,), value_size=1, num_seqs=1)


def _obs(B, D, N, slots):
    obs = torch.randn(B, D)
    onehot = torch.nn.functional.one_hot(slots, N).float()
    return torch.cat([obs, onehot], dim=1)


def _slot_mlp(net, k):
    """Standard nn.Sequential equal to slot k of the population net."""
    layers = []
    for W, b in zip(net.weights, net.biases):
        lin = torch.nn.Linear(W.shape[1], W.shape[2])
        lin.weight.data = W[k].t().clone(); lin.bias.data = b[k].clone()
        layers += [lin, torch.nn.ELU()]
    trunk = torch.nn.Sequential(*layers)
    mu = torch.nn.Linear(net.mu_w.shape[1], net.mu_w.shape[2]); mu.weight.data = net.mu_w[k].t().clone(); mu.bias.data = net.mu_b[k].clone()
    val = torch.nn.Linear(net.value_w.shape[1], net.value_w.shape[2]); val.weight.data = net.value_w[k].t().clone(); val.bias.data = net.value_b[k].clone()
    return trunk, mu, val


def test_population_forward_matches_per_slot_mlps():
    N, D, A, B = 3, 6, 2, 40
    net = _build(N, D, A)
    slots = torch.randint(0, N, (B,))
    obs = _obs(B, D, N, slots)
    mu, logstd, value, states = net({'obs': obs})
    assert mu.shape == (B, A) and logstd.shape == (B, A) and value.shape == (B, 1) and states is None
    for k in range(N):
        trunk, mu_k, val_k = _slot_mlp(net, k)
        rows = slots == k
        h = trunk(obs[rows, :D])
        assert torch.allclose(mu[rows], mu_k(h), atol=1e-5)
        assert torch.allclose(value[rows], val_k(h), atol=1e-5)
        assert torch.allclose(logstd[rows], net.sigma[k].expand(int(rows.sum()), A))


def test_slots_are_initialised_differently():
    net = _build()
    assert not torch.allclose(net.weights[0][0], net.weights[0][1])
    net2 = _build()                                   # same seed -> same init
    assert torch.allclose(net.weights[0][1], net2.weights[0][1])


def test_gradient_isolation_between_slots():
    N, D, A, B = 3, 6, 2, 30
    net = _build(N, D, A)
    slots = torch.randint(0, N, (B,))
    obs = _obs(B, D, N, slots)
    mu, _, value, _ = net({'obs': obs})
    loss = (mu[slots == 0] ** 2).sum() + (value[slots == 0] ** 2).sum()
    loss.backward()
    for W in list(net.weights) + [net.mu_w, net.value_w]:
        assert W.grad[0].abs().sum() > 0
        assert W.grad[1].abs().sum() == 0 and W.grad[2].abs().sum() == 0


def test_forward_handles_missing_slots_and_single_row():
    N, D, A = 4, 5, 3
    net = _build(N, D, A)
    obs = _obs(1, D, N, torch.tensor([2]))         # only slot 2 present
    mu, logstd, value, _ = net({'obs': obs})
    assert mu.shape == (1, A)
    trunk, mu_k, _ = _slot_mlp(net, 2)
    assert torch.allclose(mu, mu_k(trunk(obs[:, :D])), atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv312/bin/python -m pytest tests/test_population_network.py -q`
Expected: FAIL with `ModuleNotFoundError: rl_games.algos_torch.population_network`

- [ ] **Step 3: Implement the network**

```python
# rl_games/algos_torch/population_network.py
"""Population actor-critic: N independent MLP policies in one rl_games network.

Every parameter carries a leading slot axis (weights (N, in, out), biases
(N, out), heads, sigma (N, A)). A row's slot is the argmax of the LAST N
observation dims (a one-hot appended by EnvpoolSoccerVecEnv in population
mode). forward() groups rows by slot, pads to (N, G, D), runs the MLP with
batched matmuls and gathers the outputs back into row order, so the standard
PPO update trains all N policies at once and gradients never cross slots.

Config, under `network`:

    name: population_actor_critic
    population_size: 8
    seed: 0                      # slot k is initialised under seed + 100*k
    mlp: {units: [512, 256, 128], activation: elu}
    space: {continuous: {fixed_sigma: True, sigma_init: {name: const_initializer, val: 0}}}

Not supported: rnn, cnn, discrete actions, separate critic, dict observations.
"""

import numpy as np
import torch
import torch.nn as nn

from rl_games.algos_torch import network_builder


def slot_of(obs, population_size):
    """Slot index per row from the trailing one-hot (robust to normalisation)."""
    return obs[:, -population_size:].argmax(dim=1)


class PopulationBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        network_builder.NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return PopulationBuilder.Network(self.params, **kwargs)

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            kwargs.pop('num_seqs', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.N = int(params['population_size'])
            self.seed = int(params.get('seed', 0))
            mlp = params['mlp']
            units = list(mlp['units'])
            if not units:
                raise ValueError('population_actor_critic needs a non-empty mlp')
            self.activation = self.activations_factory.create(mlp.get('activation', 'elu'))
            space = params['space']['continuous']
            if not space.get('fixed_sigma', True):
                raise NotImplementedError('population_actor_critic supports fixed_sigma only')
            self.obs_dim = int(np.prod(input_shape)) - self.N
            self.actions_num = actions_num
            self._init_calls = 0
            sizes = [self.obs_dim] + units
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            for i in range(len(units)):
                w, b = self._stacked_linear(sizes[i], sizes[i + 1])
                self.weights.append(w)
                self.biases.append(b)
            self.mu_w, self.mu_b = self._stacked_linear(units[-1], actions_num)
            self.value_w, self.value_b = self._stacked_linear(units[-1], self.value_size)
            sigma_val = float(space.get('sigma_init', {}).get('val', 0.0))
            self.sigma = nn.Parameter(torch.full((self.N, actions_num), sigma_val))

        def _stacked_linear(self, in_size, out_size):
            """(N, in, out) weights + (N, out) biases, slot k under its own seed
            (nn.Linear default init, same as rl_games' 'default' initializer)."""
            ws, bs = [], []
            for k in range(self.N):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.seed + 100 * k + self._init_calls)
                    lin = nn.Linear(in_size, out_size)
                ws.append(lin.weight.detach().t().clone())
                bs.append(lin.bias.detach().clone())
            self._init_calls += 1
            return nn.Parameter(torch.stack(ws)), nn.Parameter(torch.stack(bs))

        def load(self, params):
            self.params = params

        def is_rnn(self):
            return False

        def is_separate_critic(self):
            return False

        def get_default_rnn_state(self):
            return None

        def get_value_layer(self):
            return None

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            B = obs.shape[0]
            slot = slot_of(obs, self.N)
            x = obs[:, :self.obs_dim]
            order = torch.argsort(slot, stable=True)
            counts = torch.bincount(slot, minlength=self.N)
            G = int(counts.max().item())
            starts = torch.cumsum(counts, 0) - counts
            sorted_slot = slot[order]
            pos = torch.arange(B, device=obs.device) - starts[sorted_slot]
            idx = torch.zeros(self.N, G, dtype=torch.long, device=obs.device)
            idx[sorted_slot, pos] = order
            mask = torch.zeros(self.N, G, dtype=torch.bool, device=obs.device)
            mask[sorted_slot, pos] = True
            h = x[idx]                                   # (N, G, D); pads duplicate row 0
            for w, b in zip(self.weights, self.biases):
                h = self.activation(torch.baddbmm(b.unsqueeze(1), h, w))
            mu_g = torch.baddbmm(self.mu_b.unsqueeze(1), h, self.mu_w)
            v_g = torch.baddbmm(self.value_b.unsqueeze(1), h, self.value_w)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(B, device=obs.device)
            mu = mu_g[mask][inv]                         # rows in sorted order -> original order
            value = v_g[mask][inv]
            logstd = self.sigma[slot]
            return mu, logstd, value, None
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv312/bin/python -m pytest tests/test_population_network.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rl_games/algos_torch/population_network.py tests/test_population_network.py
git commit -m "population_actor_critic: N stacked MLP policies routed by slot one-hot"
```

---

### Task 2: extract_slot — standalone actor_critic checkpoint from a slot

**Files:**
- Modify: `rl_games/algos_torch/population_network.py` (append function)
- Test: `tests/test_population_network.py` (append)

**Interfaces:**
- Produces: `extract_slot(state_dict, k, base_obs_dim) -> dict` — keys of the standard `ModelA2CContinuousLogStd` over `actor_critic` (`a2c_network.actor_mlp.{2i}.weight/bias`, `a2c_network.mu.*`, `a2c_network.value.*`, `a2c_network.sigma`, `running_mean_std.*` sliced to `base_obs_dim`, `value_mean_std.*` copied). Accepts `_orig_mod.` prefixed keys.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_extract_slot_round_trip_through_standard_model():
    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.algos_torch.population_network import PopulationBuilder, extract_slot
    from rl_games.algos_torch import model_builder
    N, D, A = 3, 6, 2
    units = [16, 8]
    model_builder.register_network('population_actor_critic', PopulationBuilder)
    pop_cfg = {'model': {'name': 'continuous_a2c_logstd'},
               'network': dict(_params(N, units, seed=5), name='population_actor_critic')}
    std_cfg = {'model': {'name': 'continuous_a2c_logstd'},
               'network': {'name': 'actor_critic', 'separate': False,
                           'mlp': {'units': units, 'activation': 'elu', 'd2rl': False,
                                   'initializer': {'name': 'default'}},
                           'space': pop_cfg['network']['space']}}
    build = lambda cfg, dim: ModelBuilder().load(cfg).build(
        {'actions_num': A, 'input_shape': (dim,), 'num_seqs': 1, 'value_size': 1,
         'normalize_value': True, 'normalize_input': True}).eval()
    pop = build(pop_cfg, D + N)
    with torch.no_grad():                               # make normalisation non-trivial
        pop.running_mean_std.running_mean.add_(torch.randn(D + N, dtype=torch.float64))
        pop.value_mean_std.running_mean.fill_(3.0)
    std = build(std_cfg, D)
    std.load_state_dict(extract_slot(pop.state_dict(), 1, D), strict=True)
    slots = torch.full((7,), 1)
    obs = _obs(7, D, N, slots)
    with torch.no_grad():
        ref = pop({'obs': obs, 'is_train': False})
        got = std({'obs': obs[:, :D], 'is_train': False})
    assert torch.allclose(ref['mus'], got['mus'], atol=1e-5)
    assert torch.allclose(ref['values'], got['values'], atol=1e-5)
    assert torch.allclose(ref['sigmas'], got['sigmas'], atol=1e-6)
    prefixed = {'_orig_mod.' + k: v for k, v in pop.state_dict().items()}
    assert set(extract_slot(prefixed, 1, D)) == set(extract_slot(pop.state_dict(), 1, D))
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv312/bin/python -m pytest tests/test_population_network.py -q -k extract`
Expected: FAIL with `ImportError: cannot import name 'extract_slot'`

- [ ] **Step 3: Implement**

```python
# append to rl_games/algos_torch/population_network.py

def extract_slot(state_dict, k, base_obs_dim):
    """Slice slot k out of a population checkpoint into a standard
    `actor_critic` (continuous_a2c_logstd) state dict with `base_obs_dim`
    inputs, so soccer_play.py / soccer_eval.py load it unchanged."""
    sd = {key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key: v
          for key, v in state_dict.items()}
    out = {}
    n_layers = len([key for key in sd if key.startswith('a2c_network.weights.')])
    for i in range(n_layers):
        out[f'a2c_network.actor_mlp.{2 * i}.weight'] = sd[f'a2c_network.weights.{i}'][k].t().contiguous()
        out[f'a2c_network.actor_mlp.{2 * i}.bias'] = sd[f'a2c_network.biases.{i}'][k].clone()
    out['a2c_network.mu.weight'] = sd['a2c_network.mu_w'][k].t().contiguous()
    out['a2c_network.mu.bias'] = sd['a2c_network.mu_b'][k].clone()
    out['a2c_network.value.weight'] = sd['a2c_network.value_w'][k].t().contiguous()
    out['a2c_network.value.bias'] = sd['a2c_network.value_b'][k].clone()
    out['a2c_network.sigma'] = sd['a2c_network.sigma'][k].clone()
    for key, v in sd.items():
        if key.startswith('running_mean_std.'):
            out[key] = v[:base_obs_dim].clone() if v.dim() == 1 else v.clone()
        elif key.startswith('value_mean_std.'):
            out[key] = v.clone()
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv312/bin/python -m pytest tests/test_population_network.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rl_games/algos_torch/population_network.py tests/test_population_network.py
git commit -m "population_actor_critic: extract_slot -> standalone actor_critic checkpoint"
```

---

### Task 3: Env population mode

**Files:**
- Modify: `rl_games/envs/envpool_soccer.py` (`__init__`, `_flatten_obs`, `_process_obs`, rewards, `step`, `get_number_of_agents`, new `set_pair_assignment` / `current_pairs`)
- Test: `tests/test_envpool_soccer.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `EnvpoolSoccerVecEnv(..., population=N)`: `env.population == N`, `get_number_of_agents() == 2*team_size`, `obs_dim == base + team_size*player_id_obs + N`, `set_pair_assignment(pairs: (num_envs, 2) int array)` applied at each match's next reset, `current_pairs() -> (num_envs, 2)`, `step(actions (num_envs*players, A))` returns rows for all players, infos on done carry `home_slot`, `away_slot`, `goal_diff` (home perspective), `opp_id` (= away_slot).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_envpool_soccer.py

# ----------------------------------------------------------- population mode

@needs_soccer
def test_population_mode_rows_and_slot_onehot():
    base = _make(2)
    try:
        base_dim = base.obs_dim
    finally:
        base.close()
    env = _make(3, population=4, player_id_obs=True)
    try:
        assert env.population == 4 and env.get_number_of_agents() == 4
        assert env.obs_dim == base_dim + 2 + 4
        pairs = np.array([[0, 1], [2, 2], [3, 0]])
        env.set_pair_assignment(pairs)
        obs = env.reset()
        assert obs.shape == (12, env.obs_dim)
        assert np.array_equal(env.current_pairs(), pairs)
        slot = obs[:, -4:].argmax(dim=1).view(3, 4)
        assert torch.equal(slot, torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2], [3, 3, 0, 0]]))
        pid = obs[:, base_dim:base_dim + 2].view(3, 4, 2)
        assert torch.equal(pid[0], torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
        obs, rew, done, info = env.step(torch.zeros(12, 3))
        assert rew.shape == (12,) and done.shape == (12,) and info['time_outs'].shape == (12,)
    finally:
        env.close()


@needs_soccer
def test_population_rewards_are_per_team():
    env = _make(2, population=2, shaping_weights={'vel_to_ball': 0.5, 'vel_ball_to_goal': 2.0,
                                                  'veloc_forward': 0, 'goal': 100})
    try:
        env.set_pair_assignment(np.array([[0, 1], [1, 0]]))
        env.reset()
        raw_obs, raw_info = env.env.reset()
        env._update_perm(raw_info)
        native = np.array([1, 1, -1, -1, 0, 0, 0, 0], dtype=np.float32)    # match 0: home scored
        native = native[np.argsort(env._perm)] if env._perm is not None else native
        r, goal = env._all_rewards(raw_obs, raw_info, native)
        assert r.shape == (2, 4) and np.array_equal(goal, [1, 0])
        closest = env._stat(raw_obs, raw_info, 'stats_closest_vel_to_ball')
        vbg = env._stat(raw_obs, raw_info, 'stats_vel_ball_to_goal')
        home_c = closest[:, :2].sum(1, keepdims=True); away_c = closest[:, 2:].sum(1, keepdims=True)
        exp = np.concatenate([0.5 * np.repeat(home_c, 2, 1), 0.5 * np.repeat(away_c, 2, 1)], 1) + 2.0 * vbg
        exp[0, :2] += 100; exp[0, 2:] -= 100
        assert np.allclose(r, exp.astype(np.float32), atol=1e-4)
    finally:
        env.close()


@needs_soccer
def test_population_pairs_applied_at_reset_and_reported():
    env = _make(3, population=3)
    try:
        env.set_pair_assignment(np.array([[0, 1], [1, 2], [2, 0]]))
        env.reset()
        env.set_pair_assignment(np.array([[2, 2], [0, 0], [1, 1]]))      # deferred
        for _ in range(60):
            _, _, done, info = env.step(torch.zeros(12, 3))
            if done.any():
                assert torch.equal(info['home_slot'].cpu(), torch.tensor([0, 1, 2]))
                assert torch.equal(info['away_slot'].cpu(), torch.tensor([1, 2, 0]))
                assert torch.equal(info['opp_id'].cpu(), info['away_slot'].cpu())
                assert info['goal_diff'].shape == (3,)
                d = done.view(3, 4)
                assert torch.equal(d[:, 0], d[:, 3])
                break
        else:
            raise AssertionError('no match finished')
        env.step(torch.zeros(12, 3))                                        # auto-reset applies pending
        assert np.array_equal(env.current_pairs(), [[2, 2], [0, 0], [1, 1]])
    finally:
        env.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py -q -k population`
Expected: 3 failed (`TypeError: Config.__new__() got an unexpected keyword argument 'population'`)

- [ ] **Step 3: Implement population mode**

In `__init__`, after `self.player_id_obs = ...`:

```python
        # population mode: all players are learner rows; each match pairs two
        # slots (home, away); a slot one-hot (last N dims) tags every row
        self.population = int(kwargs.pop('population', 0))
```

Replace the `obs_dim` line:

```python
        self.obs_dim = (self.base_obs_dim + (self.team_size if self.player_id_obs else 0)
                        + self.population)
```

After `self._opp_sigma_scale = ...`:

```python
        self._pairs = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._pending_pairs = None
```

Pool API additions (after `current_opponent_sigma_scale`):

```python
    def set_pair_assignment(self, pairs):
        """Population mode: desired (home_slot, away_slot) per match; applied
        when each match next resets."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(self.num_envs, 2)
        if self.population <= 0:
            raise RuntimeError('set_pair_assignment needs population mode')
        if pairs.min() < 0 or pairs.max() >= self.population:
            raise ValueError('slot out of range')
        self._pending_pairs = pairs.copy()

    def current_pairs(self):
        return self._pairs.copy()
```

`_apply_pending` gains, before the `if self._pending_assignment is None` line:

```python
        if self._pending_pairs is not None:
            self._pairs[mask] = self._pending_pairs[mask]
```

`_flatten_obs` appends the slot one-hot after the player id block:

```python
        if self.population > 0:
            row_slot = np.repeat(self._pairs, self.team_size, axis=1)          # (E, players)
            onehot = np.eye(self.population, dtype=np.float32)[row_slot]        # (E, players, N)
            flat = np.concatenate([flat, onehot], axis=2)
        return flat
```

`_process_obs` returns every row in population mode:

```python
    def _process_obs(self, obs, info):
        self.last_raw_obs, self.last_raw_info = obs, info   # for eval/diagnostics
        flat = torch.from_numpy(self._flatten_obs(obs, info)).to(self.device)
        if self.population > 0:
            self._away_obs = None
            return flat.reshape(self.total_players, self.obs_dim)
        home = flat[:, :self.team_size].reshape(self.num_home, self.obs_dim)
        self._away_obs = flat[:, self.team_size:]
        return home
```

Rewards: rename the body of `_home_rewards` into `_all_rewards` that returns `(E, players)` for both teams, and keep `_home_rewards` as a thin wrapper:

```python
    def _all_rewards(self, obs, info, reward):
        """Shaped reward for every player row (E, players); teams use their own
        players' stats (envpool computes stats_* relative to each player's
        goal). Returns (rewards, home goal signal)."""
        T, P = self.team_size, self.players
        reward = self._ordered(np.asarray(reward, dtype=np.float32), info).reshape(self.num_envs, P)
        goal = reward
        closest = self._stat(obs, info, 'stats_closest_vel_to_ball')
        # only the closest teammate reports a non-zero value; share it across the team
        closest = np.concatenate([np.repeat(closest[:, t * T:(t + 1) * T].sum(axis=1, keepdims=True), T, axis=1)
                                  for t in range(P // T)], axis=1)
        if self.clip_vel_to_ball:
            closest = np.maximum(closest, 0.0)
        vbg = self._stat(obs, info, 'stats_vel_ball_to_goal')
        fwd = self._stat(obs, info, 'stats_veloc_forward')
        k = self.shaping_scale
        r = (self.shaping['goal'] * goal
             + k * self.shaping['vel_to_ball'] * closest
             + k * self.shaping['vel_ball_to_goal'] * vbg
             + k * self.shaping['veloc_forward'] * fwd)
        if self.shaping['spread_out'] != 0.0:
            spread = self._ordered(np.asarray(obs['stats_teammate_spread_out'])).reshape(
                self.num_envs, P).astype(np.float32)
            r = r + k * self.shaping['spread_out'] * spread
        # stuck detector: the learner's robots (home, or everyone in population
        # mode) and the ball below stuck_speed for stuck_steps control steps
        n_robots = P if self.population > 0 else T
        vel = self._ordered(np.asarray(obs['sensors_velocimeter'])).reshape(
            self.num_envs, P, -1)[:, :n_robots, :2]
        robots_still = (np.linalg.norm(vel, axis=-1) < self.stuck_speed).all(axis=1)
        ball_rel = self._ordered(np.asarray(obs['ball_ego_linear_velocity'])).reshape(
            self.num_envs, P, -1)[:, 0, :2]
        still = robots_still & (np.linalg.norm(ball_rel, axis=-1) < self.stuck_speed)
        self._still_steps = np.where(still, self._still_steps + 1, 0)
        stuck = self._still_steps >= self.stuck_steps
        self._stuck_steps_total += stuck
        if self.shaping['stuck_penalty'] != 0.0:
            r = r - k * self.shaping['stuck_penalty'] * stuck[:, None].astype(np.float32)
        return r.astype(np.float32), goal[:, 0]

    def _home_rewards(self, obs, info, reward):
        r, goal = self._all_rewards(obs, info, reward)
        return r[:, :self.team_size], goal
```

(The existing `test_shaping_reward_matches_stats`, `test_goal_against`-free tests and `test_stuck_penalty_and_spread_bonus` stay green: spread_out was `[:, :1]` repeated before, and every player reports the same team value.)

`step`: actions and rewards for all rows in population mode:

```python
        if self.population > 0:
            full = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0).reshape(self.total_players, self.act_dim)
        else:
            home = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
            home = home.reshape(self.num_envs, self.team_size, self.act_dim)
            away = self._opponent_actions()
            full = np.concatenate([home, away], axis=1).reshape(self.total_players, self.act_dim)
```

```python
        if self.population > 0:
            rewards, goal_signal = self._all_rewards(obs, info, reward)
        else:
            rewards, goal_signal = self._home_rewards(obs, info, reward)
```

Row multiplicity: replace every `self.team_size` used for broadcasting in `step` with `rows = self.players if self.population > 0 else self.team_size` (`time_outs` repeat, `rewards.reshape(-1)`, `dones` repeat). In the `done.any()` block add:

```python
            if self.population > 0:
                infos['home_slot'] = torch.from_numpy(self._pairs[:, 0].copy()).to(self.device)
                infos['away_slot'] = torch.from_numpy(self._pairs[:, 1].copy()).to(self.device)
                infos['opp_id'] = infos['away_slot']
```

`get_number_of_agents`:

```python
    def get_number_of_agents(self):
        return self.players if self.population > 0 else self.team_size
```

Also update the module docstring with a "population mode" paragraph (rows, slot one-hot, pairs, both teams learn).

- [ ] **Step 4: Run the whole soccer file**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py -q`
Expected: 17 passed (14 old + 3 new)

- [ ] **Step 5: Commit**

```bash
git add rl_games/envs/envpool_soccer.py tests/test_envpool_soccer.py
git commit -m "envpool soccer: population mode (all players learn, slot pairs, per-team rewards)"
```

---

### Task 4: SoccerPopulationObserver

**Files:**
- Modify: `rl_games/common/soccer_observer.py` (hook in `SoccerObserver.process_infos`, new class)
- Test: `tests/test_envpool_soccer.py` (append)

**Interfaces:**
- Consumes: env `population`, `num_envs`, `set_pair_assignment`, infos `home_slot`/`away_slot`/`goal_diff`; `extract_slot` from Task 2.
- Produces: `SoccerPopulationObserver(population_config=None, anneal_config=None)` with `payoff (N,N)`, `counts (N,N)`, `sample_pairs() -> (num_envs, 2)`, `slot_winrates() -> (N,)`; writes `nn/slots/slot{k}_ep{E}.pth` (`{'model': sd, 'epoch': E, 'slot': k}`) every `slot_save_every` epochs.

- [ ] **Step 1: Write the failing test**

```python
def test_population_observer_payoff_and_pairs(tmp_path):
    from rl_games.common.soccer_observer import SoccerPopulationObserver

    class _Env:
        num_envs = 20
        population = 4

        def __init__(self):
            self.pairs = []

        def set_pair_assignment(self, pairs):
            self.pairs.append(np.array(pairs))

        def set_shaping_scale(self, s):
            pass

    algo = _FakeAlgo(num_agents=4)
    algo.vec_env = _Env()
    algo.nn_dir = str(tmp_path)
    obs = SoccerPopulationObserver(population_config={'remap_every': 1, 'p_self': 0.2, 'payoff_ema': 0.5,
                                                      'slot_save_every': 1000})
    obs.after_init(algo)
    pairs = algo.vec_env.pairs[-1]
    assert pairs.shape == (20, 2) and pairs.min() >= 0 and pairs.max() < 4
    assert (pairs[:, 0] == pairs[:, 1]).sum() == 4                      # 20% self matches
    # done rows are every 4th row; matches 0,1,2 finished: slot0 beat slot1 twice, drew with slot2
    infos = {'goal_diff': torch.tensor([1., 1., 0.]), 'home_goals': torch.zeros(3), 'away_goals': torch.zeros(3),
             'home_slot': torch.tensor([0, 0, 0]), 'away_slot': torch.tensor([1, 1, 2]),
             'opp_id': torch.tensor([1, 1, 2]), 'match_len': torch.full((3,), 300.)}
    obs.process_infos(infos, torch.tensor([[0], [4], [8]]))
    assert obs.counts[0, 1] == 2 and obs.counts[1, 0] == 2 and obs.counts[0, 2] == 1
    assert obs.payoff[0, 1] > 0.8 and obs.payoff[1, 0] < 0.2 and abs(obs.payoff[0, 2] - 0.5) < 1e-6
    wr = obs.slot_winrates()
    assert wr.shape == (4,) and wr[0] > wr[1]
    obs.after_print_stats(frame=1, epoch_num=1, total_time=0.0)
    assert len(algo.vec_env.pairs) == 2                                   # remapped
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py -q -k population_observer`
Expected: FAIL with `ImportError: cannot import name 'SoccerPopulationObserver'`

- [ ] **Step 3: Implement**

In `SoccerObserver.process_infos`, after `self._record_league(opp, ...)` inside the `if 'opp_id' in infos:` block, add a generic hook call (outside that block, after it):

```python
        self._record_match(infos, idx, diff.squeeze(1).cpu().numpy())

    def _record_match(self, infos, idx, goal_diffs):
        pass
```

New class at the end of the file:

```python
class SoccerPopulationObserver(SoccerObserver):
    """Population league: N learners in one population_actor_critic network,
    every match pairs two slots (both teams learn). Keeps the N x N payoff
    matrix (EMA of P(row beats column), draw = 0.5), resamples pairings every
    remap_every epochs and writes per-slot standalone checkpoints."""

    def __init__(self, population_config=None, anneal_config=None):
        super().__init__(anneal_config=anneal_config)
        cfg = population_config or {}
        self.remap_every = int(cfg.get('remap_every', 5))
        self.p_self = float(cfg.get('p_self', 0.1))
        self.mode = cfg.get('mode', 'uniform')           # 'uniform' | 'even' (PFSP toward 50/50)
        self.pfsp_floor = float(cfg.get('pfsp_floor', 0.02))
        self.payoff_ema = float(cfg.get('payoff_ema', 0.02))
        self.log_matrix_every = int(cfg.get('log_matrix_every', 50))
        self.slot_save_every = int(cfg.get('slot_save_every', 500))
        self._rng = np.random.RandomState(int(cfg.get('seed', 0)))
        self.N = None
        self.payoff = None
        self.counts = None

    def after_init(self, algo):
        super().after_init(algo)
        env = algo.vec_env
        self.N = int(env.population)
        self.payoff = np.full((self.N, self.N), 0.5)
        self.counts = np.zeros((self.N, self.N), dtype=np.int64)
        self._remap()

    # ------------------------------------------------------------- results

    def _record_match(self, infos, idx, goal_diffs):
        if 'home_slot' not in infos:
            return
        home = infos['home_slot'][idx].cpu().numpy()
        away = infos['away_slot'][idx].cpu().numpy()
        score = np.where(goal_diffs > 0, 1.0, np.where(goal_diffs < 0, 0.0, 0.5))
        a = self.payoff_ema
        for i, j, s in zip(home, away, score):
            if i == j:
                continue
            self.payoff[i, j] = (1 - a) * self.payoff[i, j] + a * s
            self.payoff[j, i] = (1 - a) * self.payoff[j, i] + a * (1 - s)
            self.counts[i, j] += 1
            self.counts[j, i] += 1

    def slot_winrates(self):
        off = ~np.eye(self.N, dtype=bool)
        return np.array([self.payoff[i][off[i]].mean() for i in range(self.N)])

    # --------------------------------------------------------- matchmaking

    def sample_pairs(self):
        n = self.algo.vec_env.num_envs
        n_self = int(round(n * self.p_self))
        pairs = np.zeros((n, 2), dtype=np.int64)
        s = self._rng.randint(0, self.N, size=n_self)
        pairs[:n_self] = np.stack([s, s], axis=1)
        m = n - n_self
        if self.mode == 'even' and self.N > 1:
            w = np.maximum(self.payoff * (1 - self.payoff), self.pfsp_floor)
            np.fill_diagonal(w, 0.0)
            flat = self._rng.choice(self.N * self.N, size=m, p=(w / w.sum()).ravel())
            pairs[n_self:, 0], pairs[n_self:, 1] = flat // self.N, flat % self.N
        else:
            i = self._rng.randint(0, self.N, size=m)
            j = (i + self._rng.randint(1, max(self.N, 2), size=m)) % self.N
            pairs[n_self:] = np.stack([i, j], axis=1)
        self._rng.shuffle(pairs)
        return pairs

    def _remap(self):
        self.algo.vec_env.set_pair_assignment(self.sample_pairs())

    # ------------------------------------------------------------- logging

    def _save_slots(self, epoch_num):
        from rl_games.algos_torch.population_network import extract_slot
        env = self.algo.vec_env
        base_dim = env.obs_dim - self.N
        out_dir = os.path.join(self.algo.nn_dir, 'slots')
        os.makedirs(out_dir, exist_ok=True)
        sd = self.algo.model.state_dict()
        for k in range(self.N):
            torch.save({'model': extract_slot(sd, k, base_dim), 'epoch': epoch_num, 'slot': k},
                       os.path.join(out_dir, f'slot{k}_ep{epoch_num}.pth'))
        print(f'[Population] epoch {epoch_num}: wrote {self.N} slot checkpoints to {out_dir}')

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if epoch_num % self.remap_every == 0:
            self._remap()
        wr = self.slot_winrates()
        if self.writer is not None:
            for k in range(self.N):
                self.writer.add_scalar(f'population/winrate_slot{k}', float(wr[k]), frame)
            self.writer.add_scalar('population/winrate_spread', float(wr.max() - wr.min()), frame)
        if epoch_num % self.log_matrix_every == 0:
            with np.printoptions(precision=2, suppress=True, linewidth=200):
                print(f'[Population] epoch {epoch_num} payoff (row beats col):\n{self.payoff}')
        if self.slot_save_every > 0 and epoch_num % self.slot_save_every == 0:
            self._save_slots(epoch_num)
```

Add `import os` at the top of `soccer_observer.py`. `_FakeAlgo` in the test has no `nn_dir`; the test sets it. rl_games' `A2CBase` exposes `self.nn_dir` (checkpoint directory) — used here.

- [ ] **Step 4: Run to verify**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py -q`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add rl_games/common/soccer_observer.py tests/test_envpool_soccer.py
git commit -m "SoccerPopulationObserver: payoff matrix, pair matchmaking, per-slot checkpoints"
```

---

### Task 5: Train script registration, config, smoke run, docs

**Files:**
- Modify: `scripts/soccer_train.py`
- Create: `rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml`
- Modify: `docs/ENVPOOL_SOCCER.md`, `tests/test_envpool_soccer.py` (config-load test)

**Interfaces:**
- Consumes: `PopulationBuilder`, `SoccerPopulationObserver`, env `population` key.

- [ ] **Step 1: Write the failing config test**

```python
def test_population_config_builds_model():
    import yaml
    from rl_games.algos_torch.model_builder import ModelBuilder
    from scripts.soccer_train import register_networks
    register_networks()
    cfg = yaml.safe_load(open('rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml'))['params']
    N = cfg['network']['population_size']
    assert cfg['config']['env_config']['population'] == N
    assert cfg['config'].get('torch_compile', True) is False
    model = ModelBuilder().load(cfg).build({'actions_num': 3, 'input_shape': (113 + N,), 'num_seqs': 1,
                                            'value_size': 1, 'normalize_value': True, 'normalize_input': True})
    obs = torch.cat([torch.randn(5, 113), torch.nn.functional.one_hot(torch.tensor([0, 1, 1, 7, 3]), N).float()], 1)
    out = model({'obs': obs, 'is_train': False})
    assert out['mus'].shape == (5, 3) and out['values'].shape == (5, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py -q -k population_config`
Expected: FAIL with `ImportError: cannot import name 'register_networks'`

- [ ] **Step 3: Implement**

`scripts/soccer_train.py`: add after the imports

```python
def register_networks():
    from rl_games.algos_torch import model_builder
    from rl_games.algos_torch.population_network import PopulationBuilder
    model_builder.register_network('population_actor_critic', PopulationBuilder)
```

and in `main()` before building the observer:

```python
    register_networks()
    env_cfg = conf.get('env_config', {})
    if env_cfg.get('population', 0):
        observer = SoccerPopulationObserver(population_config=conf.get('population', {}),
                                            anneal_config=conf.get('shaping_anneal'))
    elif args.league or env_cfg.get('opponent') == 'pool':
        ...
```

(import `SoccerPopulationObserver` alongside the others; `scripts/` needs an empty `scripts/__init__.py` so the test can import it — add it.)

Config `rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml` (copy of v9 with these differences):

```yaml
# EnvPool DeepMind 2v2 BOXHEAD soccer — POPULATION league v1.
#
# 8 independent MLP policies in one population_actor_critic network; every
# match pairs two of them (home slot, away slot) and BOTH teams learn from it.
# Rows per match = 4 (home0, home1, away0, away1). Recipe otherwise = v9:
# terminate_on_goal, 45 s, no reward clamp, entropy 0.002, shaping anneal to a
# 0.25 floor. Per-slot standalone checkpoints land in nn/slots/ every 500
# epochs; evaluate them with soccer_eval.py against the v9 (standalone) config.
#
# Launch:
#   python scripts/soccer_train.py -f rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml
params:
  seed: 42
  algo: {name: a2c_continuous}
  model: {name: continuous_a2c_logstd}
  network:
    name: population_actor_critic
    population_size: 8
    seed: 0
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init: {name: default}
        sigma_init: {name: const_initializer, val: 0}
        fixed_sigma: True
    mlp:
      units: [512, 256, 128]
      activation: elu
      initializer: {name: default}
  config:
    name: soccer_boxhead_population_v1
    env_name: envpool_soccer
    device: cuda
    torch_compile: False          # data-dependent group sizes in the population forward
    reward_shaper: {scale_value: 1.0}
    normalize_advantage: True
    normalize_input: True
    normalize_value: True
    value_bootstrap: True
    gamma: 0.998
    tau: 0.95
    learning_rate: 3.0e-4
    lr_schedule: adaptive
    kl_threshold: 0.008
    grad_norm: 1.0
    truncate_grads: True
    e_clip: 0.2
    entropy_coef: 0.002
    critic_coef: 1.0
    bounds_loss_coef: 0.0001
    clip_value: True
    # 256 matches x 4 robots = 1024 streams; horizon 64 -> 65536 samples/epoch
    num_actors: 256
    horizon_length: 64
    minibatch_size: 16384
    mini_epochs: 4
    max_epochs: 6500
    save_best_after: 100
    save_frequency: 500
    score_to_win: 100000
    games_to_track: 512
    shaping_anneal: {start_epoch: 1000, end_epoch: 5000, floor: 0.25}
    population:
      remap_every: 5
      p_self: 0.1
      mode: uniform
      payoff_ema: 0.02
      log_matrix_every: 50
      slot_save_every: 500
    env_config:
      walker_type: boxhead
      team_size: 2
      time_limit: 45.0
      disable_walker_contacts: True
      enable_field_box: False
      terminate_on_goal: True
      pitch_size_min: [14.0, 10.5]
      pitch_size_max: [20.0, 15.0]
      keep_aspect_ratio: True
      clip_vel_to_ball: True
      population: 8
      player_id_obs: True
      num_threads: 28
      seed: 42
      stuck_steps: 40
      stuck_speed: 0.1
      shaping_weights:
        vel_to_ball: 0.2
        vel_ball_to_goal: 1.0
        veloc_forward: 0.0
        goal: 100.0
        spread_out: 0.02
        stuck_penalty: 0.2
    player: {games_num: 20, deterministic: True}
```

- [ ] **Step 4: Run the tests, then a 20-epoch smoke run**

Run: `venv312/bin/python -m pytest tests/test_envpool_soccer.py tests/test_population_network.py -q`
Expected: 24 passed

Run: `venv312/bin/python -u scripts/soccer_train.py -f rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml --max-epochs 20 --num-actors 64 --name pop_smoke 2>&1 | grep -E "fps|Population|Error|Traceback" | tail -8`
Expected: fps lines up to epoch 20, no traceback; `[Population] epoch ... payoff` printed at epoch 0/50 boundary or not at all (log_matrix_every 50) — at least the fps lines.

Then verify a slot checkpoint loads into the standalone model:

Run: `venv312/bin/python scripts/soccer_train.py -f rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml --max-epochs 20 --num-actors 64 --name pop_smoke` already wrote no slots (slot_save_every 500); instead run in python:

```python
venv312/bin/python - <<'EOF'
import glob, torch, yaml, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from rl_games.algos_torch.population_network import extract_slot
from soccer_play import load_model
ck = sorted(glob.glob('runs/pop_smoke_*/nn/*.pth'))[-1]
sd = torch.load(ck, map_location='cpu', weights_only=False)['model']
torch.save({'model': extract_slot(sd, 3, 113)}, '/tmp/slot3.pth')
cfg = yaml.safe_load(open('rl_games/configs/envpool/ppo_soccer_boxhead_league_v9.yaml'))['params']
m = load_model(cfg, 113, 3, '/tmp/slot3.pth', 'cpu')
print(m({'obs': torch.randn(2, 113), 'is_train': False})['mus'].shape)
EOF
```
Expected: `torch.Size([2, 3])`

- [ ] **Step 5: Docs + commit**

Append to `docs/ENVPOOL_SOCCER.md` under League:

```markdown
### Population league (`ppo_soccer_boxhead_population_v1.yaml`)

`env_config.population: N` turns every robot of every match into a learner row
(`num_agents` = 4, rows home0, home1, away0, away1) and tags each row with a
slot one-hot (last N obs dims). The `population_actor_critic` network keeps N
MLPs as stacked `(N, ...)` parameters and routes rows by slot with padded
batched matmuls, so the normal PPO update trains all N policies at once.
`SoccerPopulationObserver` keeps the N x N payoff matrix
(`population/winrate_slot{k}`, matrix printed every 50 epochs), pairs slots
per match (uniform, or `mode: even` = PFSP toward 50/50), and writes
standalone per-slot checkpoints to `nn/slots/slot{k}_ep{E}.pth`. Evaluate
those with `soccer_eval.py -f ppo_soccer_boxhead_league_v9.yaml` (same
observation layout without the slot one-hot).
```

```bash
git add scripts/soccer_train.py scripts/__init__.py rl_games/configs/envpool/ppo_soccer_boxhead_population_v1.yaml docs/ENVPOOL_SOCCER.md tests/test_envpool_soccer.py
git commit -m "Population league: config v1, train-script registration, docs"
```

---

## Self-review

- Spec coverage: env mode (Task 3), network + extract (Tasks 1-2), observer (Task 4), config/trainer/docs (Task 5), tests listed in the spec map to Tasks 1-5. Out-of-scope items untouched.
- Names used consistently: `population`, `set_pair_assignment`, `current_pairs`, `_all_rewards`, `home_slot`/`away_slot`, `extract_slot(sd, k, base_obs_dim)`, `PopulationBuilder`, `register_networks`, `SoccerPopulationObserver(population_config, anneal_config)`.
- Known coupling accepted: advantage normalisation and adaptive LR shared across slots; `G = counts.max().item()` syncs once per forward.
