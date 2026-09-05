# Frozen-teacher distillation inside PPO

`rl_games/common/distillation.py` adds a distillation loss to the existing PPO
agents (`a2c_discrete`, `a2c_continuous`). The teacher is a frozen rl_games
checkpoint that consumes privileged `states`; the student consumes `obs`.
Because PPO's on-policy buffer is rebuilt from the student's own rollouts every
epoch, the loss is DAgger by construction: supervised on the states the student
visits. Three schedules turn one mechanism into the usual variants:

| variant | `ppo_coef` | `coef` (KL) | `beta` |
|---|---|---|---|
| pure DAgger | 0 | 1 | 0.5 -> 0 |
| DAgger -> PPO fine-tune | 0 then 1 | 1 -> 0 | 0.5 -> 0 |
| teacher-guided RL | 1 | small constant | 0 |

## Config

```yaml
config:
  central_value_config:          # optional; the student's asymmetric critic on states
    network: {name: actor_critic, central_value: True, mlp: {units: [512, 512, 256], activation: elu}}
    normalize_input: True
    ...
  distillation:
    teacher_config: rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml
    teacher_checkpoint: runs/<teacher run>/nn/craftax_classic_symbolic.pth
    loss: kl                        # discrete: KL(teacher || student); continuous: kl | mse
    coef:     [[0, 1.0], [300, 1.0], [600, 0.0]]   # piecewise-linear in epochs
    ppo_coef: [[0, 0.0], [300, 0.0], [301, 1.0]]
    beta:     [[0, 0.5], [150, 0.0]]              # P(env action taken from the teacher)
    warm_start_value: True
```

What happens inside the agent:

- `states` are stored in the experience buffer and passed into every
  minibatch whenever the block is present (not only with a central value net).
- `get_action_values`: with probability `beta` a row's env action is the
  teacher's sample; the student's `neglogp` is recomputed for those rows so the
  PPO importance ratios stay valid.
- `calc_gradients`: `loss = ppo_coef * ppo_loss + coef * KL(teacher || student)`
  on the minibatch's states; logged as `losses/distill`, with
  `info/distill_coef`, `info/ppo_coef`, `info/teacher_beta`.
- Warm start: every teacher tensor whose name and shape match a tensor of the
  central-value model is copied (trunk, value head, input and value
  normalisers). The teacher's critic is a privileged-state value function, so
  the student's central value starts from it instead of from scratch. Use the
  teacher's MLP as the `central_value_config` network for a full match.

The env must return `{'obs': ..., 'states': ...}` and expose `state_space`
in `get_env_info()`.

## Vision env: Craftax-Classic

`rl_games/envs/craftax_vecenv.py` (env name `craftax`, JAX, 3.11 venv) renders
the same Craftax-Classic (Crafter) state as a 1345-d symbolic vector and a
63x63x3 image: `obs: symbolic | pixels | both`. 1024 envs run at ~90k
env-steps/s on one GPU; the pixel render adds ~1 ms per batch. Rendered MuJoCo
was rejected: envpool's renderer costs 7-10 ms per env per frame, serially.
`CraftaxObserver` (attached by `scripts/craftax_train.py`) logs
`craftax/score` (Crafter score = geometric mean of achievement rates) and
`craftax/ach_<name>`.

Configs in `rl_games/configs/craftax/`:

- `ppo_craftax_classic_symbolic.yaml` — teacher, MLP on symbolic states.
- `ppo_craftax_classic_pixels.yaml` — pixel student from scratch (CNN), baseline.
- `ppo_craftax_classic_pixels_distill.yaml` — pixel student distilled from the
  teacher, central value = teacher MLP on states, warm-started.

```bash
source venv/bin/activate
python scripts/craftax_train.py -f rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml
# edit distillation.teacher_checkpoint, then (both fit on one GPU with JAX prealloc capped)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python scripts/craftax_train.py -f rl_games/configs/craftax/ppo_craftax_classic_pixels.yaml
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 python scripts/craftax_train.py -f rl_games/configs/craftax/ppo_craftax_classic_pixels_distill.yaml
python -m pytest tests/test_distillation.py tests/test_craftax_env.py -q
```

## Results (2026-09-05, one seed each, 1500 epochs = 98M frames)

Teacher: symbolic MLP, 25 min at 106k fps. Students: same CNN, same frames,
trained side by side on one GPU (scratch 44k fps, distilled 32k fps).

| frames (M) | 10 | 20 | 40 | 60 | 80 | 98 |
|---|---|---|---|---|---|---|
| teacher (symbolic) reward | 12.2 | 14.0 | 15.4 | 16.2 | 16.6 | 17.0 |
| scratch (pixels) reward | 11.2 | 13.0 | 14.6 | 15.2 | 15.8 | 16.1 |
| distilled (pixels) reward | 12.1 | 13.4 | 16.1 | 16.4 | 16.5 | 16.5 |
| teacher Crafter score | 26 | 34 | 45 | 55 | 58 | 60 |
| scratch Crafter score | 19 | 30 | 39 | 46 | 53 | 54 |
| distilled Crafter score | 30 | 34 | 55 | 56 | 56 | 56 |

Achievement rates at 98M frames (teacher / scratch / distilled):
collect_iron 0.84 / 0.69 / 0.69, make_iron_pickaxe 0.49 / 0.20 / 0.19,
make_iron_sword 0.40 / 0.42 / 0.48, defeat_skeleton 0.80 / 0.62 / 0.63.

Reading:

- The distilled student reaches the scratch student's *final* level
  (reward 16.1, score 54) at 40M frames, i.e. ~2.5x fewer frames, and ends
  slightly higher (16.5 / 56 vs 16.1 / 54). Neither pixel student reaches the
  teacher (17.0 / 60) in this budget; the gap is in iron-pickaxe rate.
- Numbers before 20M frames for the distilled run include teacher actions
  (`beta` 0.5 -> 0 over the first 150 epochs / 10M frames); from 20M on they
  are the student alone.
- `losses/distill` (monitoring only after epoch 600) climbs from 0.7 to 2.1
  once PPO takes over: the fine-tuned student moves away from the teacher's
  action distribution while keeping its return, which is the point of the
  fine-tune phase.
- Single seed each; differences of ~0.4 reward at the end are within what a
  seed can move. The 40M-frame gap (16.1 vs 14.6, 55 vs 39) is not.
