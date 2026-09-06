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

## Robotics vision: Panda pick-cube (mujoco_playground)

`rl_games/envs/playground_vecenv.py` (env name `playground`, JAX + Warp
physics, mujoco 3.12's built-in batch renderer) exposes any mujoco_playground
env with `obs: state | pixels | both`; the state vector is the env's own
`_get_obs(data, info)` from the same physics state the pixels are rendered
from. `PandaPickCubeCartesian`: 66-d state, 64x64 gripper camera, 3-d
Cartesian actions, episodes end on success. Configs in
`rl_games/configs/playground/`, `scripts/playground_train.py`,
`scripts/playground_play.py`.

Setup notes: playground 0.2 needs mujoco 3.12 + mujoco-warp 3.12 +
`warp-lang==1.16.0`; under WSL2 the CUDA toolkit's stub libcuda shadows the
driver, so `LD_LIBRARY_PATH=/usr/lib/wsl/lib` is required (the scripts
re-exec with it). 1024 worlds train at ~30k frames/s (state) / ~7k (pixels).

Results (one seed each, 300 epochs = 9.8M frames):

Success rate (fraction of episodes with the cube at the target):

| frames (M) | 1 | 2 | 4 | 6 | 8 | 9.8 |
|---|---|---|---|---|---|---|
| scratch (pixels) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| scratch + state-aux (pixels) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| distilled (pixels) | 0.06 | 0.23 | 1.00 | 1.00 | 1.00 | 1.00 |
| teacher (state) | 0.00 | 0.00 | 0.61 | 0.98 | 1.00 | 1.00 |

Episode reward (shaped: gripper-to-box 4, box-to-target 8, lifted 0.5,
success 2, collisions 0.3; the env pays out increases only, so this is the
best progress reached in the episode):

| frames (M) | 1 | 2 | 4 | 6 | 8 | 9.8 |
|---|---|---|---|---|---|---|
| scratch (pixels) | 1.9 | 1.9 | 1.7 | 2.1 | 4.0 | 3.7 |
| scratch + state-aux (pixels) | 1.9 | 2.3 | 2.3 | 3.8 | 3.9 | 3.9 |
| distilled (pixels) | 3.0 | 4.4 | 10.5 | 10.6 | 10.5 | 10.6 |
| teacher (state) | 2.7 | 4.6 | 8.5 | 10.5 | 10.8 | 10.6 |

A reward near 4 with success 0 means the gripper reaches the cube (the
4-point term) but never carries it (the 8-point term).

The pixel student from scratch never grasps the cube in 9.8M frames; the
distilled pixel student reaches the teacher's 100% success by 4M frames,
i.e. as fast as the state teacher itself learned.

**Privileged state as an auxiliary target instead of a teacher**
(`ppo_panda_pick_pixels_stateaux.yaml`, network `actor_critic_state_aux`:
same CNN plus an MSE head regressing the running-normalised 66-d state from
the actor trunk, `state_aux: {}` in the algo config) does not help: the head
learns part of the state (training MSE 0.78 -> 0.17; on fresh rollouts R^2
0.85 for arm joints, 0.83 gripper position, 0.75 cube-to-gripper offset,
0.49 target-to-cube offset, but ~0 for cube orientation and velocities) yet
success stays at 0 and the reward curve is the scratch student's. On this task the bottleneck is
not the representation but exploration / credit assignment for a grasp that
random pixel policies never stumble on; the teacher's actions supply exactly
that, a state-prediction target does not.

What it took (the first attempt failed, `runs/panda_pick_pixels_distill_v1_failed_*`):

- **Fixed learning rate.** The distillation term makes consecutive policies
  differ by far more than `kl_threshold`, so rl_games' adaptive schedule sank
  the lr to 5e-6 from epoch 1 and the student barely moved. Use
  `lr_schedule: fixed` (or linear) with distillation.
- **Clip the teacher's means.** A continuous teacher's `mu` is unbounded
  (here ±3 with the env clipping actions at ±1); matching it is an unreachable
  target that fights the student's bounds loss until the means blow up.
  `teacher_mu_clip: 1.0` clamps the targets to the action range. With that,
  `loss: mse` on the means (behaviour cloning of the mean, sigma left to PPO)
  was the stable choice for continuous actions.

## Results: Craftax-Classic (2026-09-05, one seed each, 1500 epochs = 98M frames)

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
