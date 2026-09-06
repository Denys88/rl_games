# Frozen-teacher distillation inside PPO — design (2026-09-05)

## Goal
A distillation loss inside the existing PPO agent, with weight schedules,
so that DAgger, DAgger -> PPO fine-tune and teacher-guided RL are config
variants of one mechanism. The teacher is a frozen rl_games checkpoint that
consumes privileged `states`; the student consumes `obs`. The student's
central (asymmetric) value network is warm-started from the teacher.

First application: vision. Craftax-Classic (Crafter in JAX) renders the same
game state as a 1345-d symbolic vector (teacher) or a 63x63x3 image
(student), at ~90k env-steps/s for 1024 envs on the GPU.

## Components

### 1. `CraftaxVecEnv` (`rl_games/envs/craftax_vecenv.py`, env name `craftax`)
- kwargs: `game` (`Craftax-Classic-v1` | `Craftax-v1`), `obs`
  (`symbolic` | `pixels` | `both`), `seed`, `device`.
- Steps `jax.vmap(env.step)` under `jit` with craftax auto-reset (the obs
  returned on done is the new episode's first obs, same-step autoreset);
  torch tensors via dlpack (pattern of `PgxGoVecEnv`).
- `obs: both` returns `{'obs': uint8 (N, 63, 63, 3), 'states': float32 (N, 1345)}`;
  `get_env_info` exposes `observation_space` (Box uint8) and `state_space`
  (Box float32) so rl_games' central-value plumbing picks `states` up.
- `symbolic` / `pixels` return the single tensor. Discrete(17) actions.
- Info on done: `achievements` (N, 22) float from craftax info, logged by a
  `CraftaxObserver` as `craftax/<achievement>` rates and `craftax/score`
  (Crafter geometric-mean score over achievements).

### 2. `TeacherDistillation` (`rl_games/common/distillation.py`)
Config block `distillation` in the algo config:

```yaml
distillation:
  teacher_config: rl_games/configs/craftax/ppo_craftax_classic_symbolic.yaml
  teacher_checkpoint: runs/.../nn/craftax_classic_symbolic.pth
  loss: kl                 # discrete: KL(teacher || student); continuous: kl | mse
  coef:  [[0, 1.0], [400, 1.0], [800, 0.0]]     # piecewise-linear in epochs
  ppo_coef: [[0, 0.0], [400, 0.0], [401, 1.0]]  # 0 = pure DAgger phase
  beta: [[0, 0.5], [200, 0.0]]                  # P(env action taken from teacher)
  warm_start_value: True
```
- Teacher model built from `teacher_config`'s network with
  `input_shape = state_space.shape`, weights from the checkpoint's `model`,
  frozen, eval, on the agent device.
- `coef(epoch)`, `ppo_coef(epoch)`, `beta(epoch)`: piecewise-linear
  interpolation, constant outside the given range.
- `loss(res_dict, states)`: discrete `sum p_t (log p_t - log p_s)` from
  `res_dict['logits']`; continuous `kl` of diagonal Gaussians
  (`mus`, `sigmas`) or `mse` on `mus`. Mean over rows (masked by
  `rnn_masks` when present).
- `mix_actions(res_dict, states, rng)`: with probability `beta` per row the
  env action is the teacher's sample; `neglogpacs` is recomputed under the
  student for the substituted rows (discrete: `-log_softmax(logits)[a]`;
  continuous: Gaussian neglogp with `mus`/`sigmas`). Rows keep the student's
  own action otherwise.
- `warm_start_central_value(cv_model)`: copy every teacher tensor whose name
  and shape match a tensor of the central-value model (trunk, value head,
  `running_mean_std` on states, `value_mean_std`); print copied / skipped
  counts. Requires the CV network to use the teacher's architecture.

### 3. Agent hooks (`a2c_common.py`, `a2c_discrete.py`, `a2c_continuous.py`)
- `self.distill = TeacherDistillation(...)` when the block is present.
- `self.store_states = has_central_value or distill is not None`; the
  experience buffer allocates `states` (algo_info `store_states`), rollouts
  store `obs['states']`, `prepare_dataset` adds `states` to the main dataset
  (minibatches then carry `input_dict['states']`).
- `get_action_values`: after inference, `mix_actions` when `beta > 0`.
- `calc_gradients` (both): `loss = ppo_coef * loss + coef * distill_loss`;
  the distill term is logged through `aux_loss_dict['distill']`; the three
  schedule values are written as `info/distill_coef`, `info/ppo_coef`,
  `info/teacher_beta` each epoch.
- Warm start runs once in `__init__` after the central value net exists.

### 4. Configs (`rl_games/configs/craftax/`)
- `ppo_craftax_classic_symbolic.yaml` — teacher: discrete PPO, MLP
  [512, 512, 256] elu, 1024 envs, horizon 64, minibatch 16384, gamma 0.99.
- `ppo_craftax_classic_pixels.yaml` — student from scratch: CNN (32/64/64
  convs, `permute_input: True`, uint8 obs) + MLP 512, no teacher. Baseline.
- `ppo_craftax_classic_pixels_distill.yaml` — same CNN student, `obs: both`,
  `central_value_config` = teacher MLP on states (warm-started), the
  `distillation` block above.

### 5. Tests
- `tests/test_distillation.py`: schedule interpolation; discrete KL equals a
  manual computation and is 0 for identical logits; Gaussian KL and mse;
  `mix_actions` substitutes ~beta of rows and recomputed neglogp matches
  `Categorical(logits).log_prob`; warm start copies matching tensors and
  skips mismatched ones.
- `tests/test_craftax_env.py`: shapes/dtypes for the three obs modes,
  `state_space` in env info, actions accepted as torch int64, autoreset
  keeps shapes, achievements info on done. Integration: 2 epochs of the
  distill config with 16 envs and a teacher checkpoint saved from a
  freshly built (untrained) teacher.

## Out of scope
RNN students; dict `states`; multi-GPU teacher sync; DAgger data
aggregation across epochs (PPO's on-policy buffer is the DAgger dataset by
construction).
