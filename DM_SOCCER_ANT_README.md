# DM Control Ant Soccer in rl_games

**Branch**: `feature/dm-soccer-ant-env`

Multi-agent ant soccer training using `dm_control.locomotion.soccer` wrapped
for rl_games PPO + frozen-pool self-play.

## Quick start

### Eval a checkpoint (fast, no rendering)
```bash
python eval_dm_soccer_ant.py \
    runs/dm_soccer_ant_random_opp_<ts>/nn/dm_soccer_ant_random_opp.pth \
    --config rl_games/configs/dm_control/ppo_soccer_ant_v3.yaml \
    --episodes 20 --opponent random --time-limit 30.0 \
    --no-terminate-on-goal --field-size 15 10 --goal-size 1.0 6.0 0.6
```

### Render an mp4 (slow on WSL2/EGL ~30× slowdown)
```bash
MUJOCO_GL=egl python play_dm_soccer_ant.py \
    runs/<...>/nn/<...>.pth \
    --config rl_games/configs/dm_control/ppo_soccer_ant_v3.yaml \
    --out demo.mp4 --episodes 3 --opponent random --time-limit 30.0 \
    --no-terminate-on-goal --field-size 15 10 --goal-size 1.0 6.0 0.6 \
    --height 240 --width 320
```

### Train (Phase 1 baseline, random opp)
```bash
python -u runner.py --train --file rl_games/configs/dm_control/ppo_soccer_ant.yaml
```

### Train (v6: high entropy + chase init, the most promising config)
```bash
python -u runner.py --train \
    --file rl_games/configs/dm_control/ppo_soccer_ant_v6.yaml \
    --checkpoint runs/dm_soccer_ant_random_opp_<ts>/nn/dm_soccer_ant_random_opp.pth
```

## Files

### Env wrapper
- `rl_games/envs/dm_soccer.py` — dm_soccer Ant adapter for rl_games
  - Custom `field_size=(W, H)` to shrink the default 32-48m × 24-36m pitch
  - Custom `goal_size=(D, W, H)` for wider goals
  - `terminate_on_goal=False` for multi-turn episodes (ball respawns)
  - 7 configurable shaping signals: `vel_to_ball`, `vel_ball_to_goal`,
    `veloc_forward`, `ball_field_progress`, `ball_near_goal_bonus`,
    `behind_ball_bonus`, `goal`
- `rl_games/envs/dm_soccer_opponent.py` — `FrozenA2COpponent` for self-play
  pool with recency-weighted sampling and `_to_cpu` Ray-safe weight load
- `rl_games/algos_torch/self_play_manager.py` — patched with `_to_cpu`
  helper (the missing pong-era patch that crashed the first push attempt)

### Configs (in `rl_games/configs/dm_control/`)
| Config | Field | Goals | Shaping | Init |
|---|---|---|---|---|
| `ppo_soccer_ant.yaml` (Phase 1) | default 32-48m | default 33%w | heavy chase | fresh |
| `ppo_soccer_ant_v2.yaml` | 15×10 | default | goal-dominated | Phase 1 |
| `ppo_soccer_ant_v3.yaml` | 15×10 | default | + ball_near_goal_bonus | fresh / Phase 1 |
| `ppo_soccer_ant_v4.yaml` | 15×10 | default | tiny shaping bridge | v3 |
| `ppo_soccer_ant_v5c.yaml` | 15×10 | wider 6m | + behind_ball_bonus | Phase 1 |
| `ppo_soccer_ant_v6.yaml` | 15×10 | wider 6m | v5c + entropy 0.05 | Phase 1 |
| `ppo_soccer_ant_1v1.yaml` | 15×10 | wider 6m | + behind_ball_bonus | fresh |

### Tools
- `eval_dm_soccer_ant.py` — fast goal-counting eval (no render)
- `play_dm_soccer_ant.py` — mp4 render with `--opponent {random,noop,self}`

## Training results (20-ep eval vs random opp, 15×10 pitch)

| Checkpoint | Goals/ep | Win rate |
|---|---|---|
| Phase 1 chase-policy | 0.15 | 10% |
| v3c2 (small field + ball_near_goal_bonus) | 0.15 | 10% |
| v4b (pure goal reward + tiny shaping) | 0.15 | 10% |
| v5c (+ behind_ball_bonus) | 0.04 (regression — shaping over-optimized) | 10% |

All shaping iterations plateau near the chase-policy baseline. The unlock
to consistently scoring (≥1 goal/ep) is more compute (DM's original
ant-soccer paper trained billions of frames; we reached ~30M).

## Key learnings (in case you continue)

1. **Default ant pitch is too big** (32-48m × 24-36m, ~70× ant body length).
   Random play never scores. `field_size=(15, 10)` is mandatory.
2. **Default goal width** is 33% of pitch height — try widening to 50-60%
   via `goal_size=(depth, width, height)` to make scoring tractable.
3. **`MUJOCO_GL=egl`** rendering on WSL2 is ~30× slower than real-time.
   Use eval (no render) for stats; render only the final demo.
4. **Self-play `update_score`** must sit ABOVE the chase-policy baseline
   reward, otherwise pushes thrash every ~10 epochs and PPO never settles.
5. **`_to_cpu` is mandatory** for Ray weight serialization (CUDA tensors
   can't deserialize in CPU-only worker processes). See
   `self_play_manager.py`.
6. **Shaping pathology is real**. Every shaped reward gets over-optimized
   without producing actual goals. Pure goal reward is too sparse.
   The combination needs careful balancing or much more training time.

## Suggested next experiments

1. **Overnight v6 run** — let `ppo_soccer_ant_v6.yaml` cook for 8+ hours
2. **1v1 curriculum** — train `ppo_soccer_ant_1v1.yaml` from scratch first,
   then transfer to 2v2 (network arch is the same; obs dim differs at 218 vs 296,
   so you'd need to start fresh on 2v2 with a separately-trained 1v1 policy
   used only as opponent)
3. **Imitation bootstrap** — write a hand-scripted "go behind ball, push
   forward" Python policy and use it as the initial pool member via
   `opponent_kind='policy'` and a custom factory
4. **PFSP / league** — extend `dm_soccer_opponent.py` with prioritized
   fictitious self-play (priority ∝ `(1 − win_rate)^p`) plus 1-2 main-exploiter slots
