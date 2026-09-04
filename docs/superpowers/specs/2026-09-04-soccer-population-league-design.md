# Soccer population league — design (2026-09-04)

## Goal
Train N (=8) separate MLP policies for envpool 2v2 boxhead soccer in one
rl_games process. Every match pairs two learners (home slot i, away slot j);
both teams learn from the match. Diversity comes from separate initialisation
and separate experience; no frozen-snapshot pool in v1.

## Components

### 1. `EnvpoolSoccerVecEnv` population mode (`population: N`)
- All `2 * team_size` player rows of a match are learner streams, env-major:
  `[home0, home1, away0, away1]`; `get_number_of_agents()` returns 4.
- Per match a pair `(home_slot, away_slot)`; `set_pair_assignment(pairs)` is
  applied at the match's next reset (same lazy mechanism as opponent ids).
- Observation rows: base obs (+ player-id one-hot) + slot one-hot (N dims).
- Rewards for all rows with the existing shaping formula, computed per team
  from that team's players' `stats_*` (envpool computes them relative to each
  player's own goal). Native goal reward is already per player. Stuck
  detector per team.
- Infos on done: `goal_diff` (home perspective), `home_slot`, `away_slot`,
  `match_len`, plus the existing keys (`opp_id` = away slot for compatibility).
- No opponent inference in this mode.

### 2. `population_actor_critic` network (`rl_games/algos_torch/population_network.py`)
- Config: `population_size: N`, `mlp.units`, activation elu, `fixed_sigma`.
- Parameters stacked along a leading slot axis: per layer `W (N, in, out)`,
  `b (N, out)`; heads `mu (N, h, A)`, `value (N, h, 1)`; `sigma (N, A)`.
  Slot k initialised with rl_games default init under seed `base_seed + k`.
- `forward(obs_dict)`: `slot = argmax(obs[:, -N:])`, `x = obs[:, :-N]`; rows
  sorted by slot, padded to `(N, Gmax, D)` with an index tensor; layers via
  `torch.baddbmm`; outputs gathered back to row order. Returns
  `(mu, sigma_per_row, value, None)`.
- `extract_slot(state_dict, k, obs_dim)` -> standard `actor_critic` state
  dict (`a2c_network.actor_mlp.<i>.weight` ...) with running-mean-std sliced
  to the base obs so `soccer_play.py` / `soccer_eval.py` work unchanged.

### 3. `SoccerPopulationObserver` (`rl_games/common/soccer_observer.py`)
- Payoff matrix `P[i, j]` = EMA of P(slot i beats slot j) from finished
  matches (draw 0.5), counts matrix.
- Logs `population/winrate_slot{k}`, `population/payoff_spread`,
  the matrix as text every `log_matrix_every` epochs.
- Matchmaking every `remap_every` epochs: pairs uniform over ordered (i, j),
  `p_self` fraction i == j; optional PFSP (`mode: even`) weighting pairs by
  `P(1-P)`.
- Writes per-slot standalone checkpoints `nn/slots/slot{k}_ep{E}.pth` every
  `slot_save_every` epochs (extract_slot), for the eval script.
- Shaping anneal inherited from `SoccerObserver`.

### 4. Trainer / config
- Standard `a2c_continuous`; `num_agents` 4; batch 256 x 4 x 64 = 65536,
  minibatch 16384. Advantage normalisation and adaptive LR shared.
- `ppo_soccer_boxhead_population_v1.yaml`: v9 recipe (terminate_on_goal,
  45 s, no reward clamp, entropy 0.002, anneal floor 0.25), `population_size 8`.
- `scripts/soccer_train.py` registers the network and picks the observer when
  `env_config.population` is set.

## Tests (tests/test_envpool_soccer.py, tests/test_population_network.py)
- population forward == per-slot standard MLP with copied weights (N=3).
- gradient isolation: loss on slot-0 rows -> zero grad on slot-1 params.
- rows shuffled across slots are routed correctly (compare with per-row loop).
- env population mode: row layout, slot one-hot, `num_agents == 4`, home goal
  gives +goal to home rows and -goal to away rows, pair assignment applied at
  reset.
- observer payoff bookkeeping and pair sampling shape.
- extract_slot round trip: standalone model output == population output for
  that slot.

## Out of scope (v1)
Frozen snapshot opponents (needs row masking), mixed architectures,
per-learner reward weights, per-slot advantage normalisation.
