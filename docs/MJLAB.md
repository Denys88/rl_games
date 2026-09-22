# MJLab (MuJoCo Lab)

[MJLab](https://github.com/NVlabs/mjlab) is a GPU-accelerated robotics simulation framework built on MuJoCo (via Warp). It provides vectorized environments running entirely on GPU with fast parallel physics.

## Setup

```bash
pip install -e ".[mujoco]"
pip install mjlab
```

## How to run

**Go1 Velocity (flat terrain)**
```bash
python runner.py --train --file rl_games/configs/mjlab/ppo_go1_velocity.yaml
```

**G1 Humanoid Velocity (flat terrain)**
```bash
python runner.py --train --file rl_games/configs/mjlab/ppo_g1_velocity.yaml
```

## Configs

| Environment | Config | Envs | Horizon | Epochs |
|-------------|--------|------|---------|--------|
| Go1 Velocity (flat) | `configs/mjlab/ppo_go1_velocity.yaml` | 4096 | 24 | 5000 |
| G1 Velocity (flat) | `configs/mjlab/ppo_g1_velocity.yaml` | 4096 | 24 | 5000 |

**Lift-Cube-Yam (manipulation)**
```bash
python runner.py --train --file rl_games/configs/mjlab/ppo_lift_cube_yam.yaml
```

**WujiHand in-hand cube reorientation** (external task plugin — install
[wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab) from a source clone,
`pip install -e <clone>`; its tasks register via mjlab entry points):
```bash
python runner.py --train --file rl_games/configs/mjlab/ppo_wujihand_reorient.yaml
```
Note for long-horizon manipulation configs: a positive entropy bonus on a global
`fixed_sigma` can drive a sigma runaway over 1B+ frame runs (reproduced in both
fp32 and bf16). The Lift-Cube-Yam config is **validated to task success**: episode success
0.85 over held-out evaluation episodes vs 0.72 for the reference rsl-rl recipe at the
same 491M-frame budget (asymmetric central-value critic on the env's privileged obs
group + value normalization + adaptive LR; see the config for the full recipe).

## Results

### Go1 Flat Velocity

Same-machine comparison against mjlab's own rsl-rl reference recipe at the
reference batch geometry (4096 envs × 24 steps), the reference curriculum
schedule, and mjlab's own default budget (10k iterations — the full protocol,
including the doubled stage-2 command ranges the curriculum enables after
iteration 5000):

| Trainer | Mean episode reward (last-100, full 10k protocol) |
|---------|---------------------------------------------------|
| mjlab rsl-rl reference | 83.2 |
| rl_games (`ppo_go1_velocity.yaml`) | **86.8** |

The shipped config stops at `max_epochs: 5000`; the table above was measured
with `max_epochs: 10000` (mjlab's default budget), which is the only override.
At the 5000-iteration mark (stage-1 command range only), the same runs read
94.0 (reference) vs **97.0** (rl_games, peak 98.9):

![Go1 Flat Velocity, first 5000 iterations of the 10k runs](pictures/mjlab/go1_flat_comparison_5000.png)

A compressed-curriculum variant (`velocity_stage_steps` moves the range
expansions to iterations 2500/5000, giving the hardest 2–3 m/s range 5000
training iterations that the default schedule never allocates) produces the
fastest policy: measured 1.07 m/s sustained at a 2.0 m/s command, roughly
double the stage-1 policies.

### Go1 Rough Velocity

Central value network significantly improves rough terrain performance (~60 vs ~45 reward).

![Go1 Rough Velocity](pictures/mjlab/go1_rough_training.png)

### G1 Humanoid Flat Velocity

Humanoid locomotion is a substantially harder task — mjlab's own default
budget for G1 is 30k iterations (3× Go1's), and short-budget reward
comparisons are misleading here: at 5000 iterations both stacks produce
policies that score reward without actually tracking velocity commands
(measure deployable behavior, not reward meters). We do not currently claim
a G1 comparison; the `ppo_g1_velocity.yaml` config is training-stable and
under active tuning against the reference's full-budget result.

Recipe (both locomotion configs): asymmetric central value on the privileged
`critic` obs group, same size as the actor net, trained at the full 5 mini-epochs —
halving CV epochs was tested and rejected (Go1 drops from 97.0 to 92.6; the
critic quality carries the advantage estimates throughout, not just early);
`schedule_type: standard` with `kl_threshold: 0.016`, entropy 0, truncation
bootstrap on.

### WujiHand In-Hand Cube Reorientation

In-hand reorientation to uniformly sampled SO(3) goals with switch-on-success,
trained on the unmodified wuji-mjlab task (reward design, DR and success
protocol exactly as released). Same-machine comparison (one RTX PRO 6000 per
run, same window) against the vendored rsl-rl fork that ships with wuji-mjlab,
identical single-GPU data budget (8192 envs × 40 steps, 5000 iterations,
1.64B frames). The 2-GPU run processes twice as many frames.
Score: goal reaches per episode (training metric, 50-iteration EMA). The raw
action smoothness is reported next to it because a policy can keep scoring
while its actions become unusable (see the stability notes below); the
reference's step-to-step action change is 0.39.

| Trainer | Goal reaches / episode at 5000 | Time to the reference's final score | Action change / step |
|---------|-------------------------------|-------------------------------------|----------------------|
| wuji-mjlab rsl-rl fork (published recipe, actor 512/256/128) | 16.9 | 2.14 h | 0.39 |
| wuji-mjlab rsl-rl fork at rl_games' network widths | 17.35 | 2.22 h | 0.40 |
| rl_games `ppo_wujihand_reorient.yaml`, seeds 42 / 7 / 123 | **19.6 / 18.9 / 18.9** | **1.38 / 1.44 / 1.54 h** | 0.37 |
| rl_games, same recipe with the KL-adaptive band 5e-5..2e-4 instead of the fixed rate | 18.8 / 18.2 / 18.9 | 1.56 / 1.86 / 1.60 h | 0.37 |
| rl_games, adaptive band, at the reference's network widths | 16.7 | — | 0.36 |
| rl_games, adaptive band, on 2 GPUs (2× frames per iteration) | **20.6** | 1.27 h | 0.36 |

The full-width single-GPU fixed and adaptive recipes each have three seeds
(42 / 7 / 123); the other rows each report one run. These runs had no terminal
action storm. The width comparisons help assess the recipes, but do not
isolate the trainer: exploration, normalization and minibatch geometry also
differ, and the smaller rl_games network used the adaptive schedule. These
are training metrics, not held-out evaluation scores. The fixed-rate recipe
reaches the published reference's final training score in 28–36% less wall
time on this machine; equal frames do not imply equal optimizer steps or
compute cost.

![WujiHand Reorient comparison](pictures/mjlab/wuji_reorient_comparison.png)

Recipe notes (all in the config): asymmetric central-value critic on the env's
privileged `critic` obs group (16384 × 4 mini-epochs), value normalization on,
truncation `value_bootstrap` on, minibatch 16384, a **fixed learning rate of
1e-4**, and a **global exploration std** (`fixed_sigma: true`,
`sigma_parametrization: softplus`, `min_sigma: 0.2`) with `entropy_coef: 0`.
The fixed rate is deliberate: final training scores improved by 4.0%, 3.9%
and 0.4% on the paired seeds, with a 2.8% higher mean score. The adaptive
controller often lowered the rate early, while the fixed-rate runs learned
faster. This supports the fixed recipe for this task; it does not establish
that KL is insensitive to learning rate or that early KL excess is harmless.
The existing scheduler reads a forward pass made before the optimizer step,
relative to a stored policy, so its signal includes earlier updates and
changes to observation normalization. Heavy-tailed advantages can affect
updates, but their causal contribution needs a controlled replay.

**Stability notes.** With a state-dependent std (`fixed_sigma: false`) this
task can collapse thousands of iterations into training: on rare states the
std head extrapolates to values of 40–200 while the batch mean stays at 0.2,
the task's penalty on the raw (unclamped) action explodes, and the policy
does not recover. Batch averages such as entropy and mean KL do not show it,
and an adaptive learning rate cannot act on a tail of states. Two settings
remove it, each verified on three seeds: a global std (the setting Shadow
Hand and DeXtreme trained with; the recipe above) or a ceiling on the
state-dependent std (`max_sigma: 1.0`, 17.6–18.6 reaches with the adaptive
band, 18.8–19.2 with the fixed rate). Do not clip
actions in the trainer while using a state-dependent std here (the task's
penalty then no longer restrains the std head), do not add an entropy bonus
to a global std on long runs, and read every score together with a smoothness
metric. `use_diagnostics: true` logs the batch-max std per mini-epoch
(`diagnostics/policy/sigma_max/<mini_epoch>`), which shows the tail long before a collapse.
Global std and entropy zero changed together, so their individual effects are not isolated. `CONFIG_PARAMS.md` documents `max_sigma`, `kl_reference`, `kl_schedule_source` and the diagnostics.

## Notebooks

- `notebooks/mjlab_training.ipynb` — end-to-end at notebook scale: Go1 velocity training
  (8192 envs, 1000 epochs, ~17 min on an RTX 4090), training curve, then rendering of the
  trained policy and a commanded-vs-achieved velocity probe (the notebook-scale walker
  achieves ~0.9 m/s at commanded 1.0; undertrained or under-diversified policies probe ~0).
  Env count A/B'd back-to-back on an RTX 4090 (minibatch 16384, 1000 epochs): 8192 envs
  reach 95.2 final reward in 17.1 min vs 4096's 88.8 in 11.8 min — same reward-per-frame
  curve, 2× data; throughput scales (200k vs 145k total FPS) and VRAM is no constraint
  (1.7 GiB peak — MJLab is compute-bound, not memory-bound).
- `notebooks/mjlab_training_colab.ipynb` — the same pipeline for Colab: installs rl_games
  from git (until the PyPI release) and mjlab from PyPI, auto-scales env count by GPU VRAM
  (8192 envs on ≥20 GiB runtimes, 4096 below).

**Rendering design — record-then-replay (2026-08-03):** the notebooks never render
from the simulation process. The rollout process (warp/CUDA, zero GL) dumps the
compiled `MjModel` plus per-frame `qpos`; a second process (plain `mujoco` +
EGL, zero warp) replays the states through `mujoco.Renderer` with a tracked
camera. Reason: on some cloud driver stacks (observed: Colab G4, sm_120,
driver 13.0) creating an EGL context in a process where the full mjlab env
holds CUDA segfaults — and with the GL context created first, it deadlocks
instead. Context-creation-order probes alone pass; the fault needs the full
env in-process, so the only robust fix is not sharing the process at all.
Both phases run as subprocesses of the notebook kernel: a native fault
surfaces as an exit code, never a kernel crash. (The step-by-step diagnostic
notebook that isolated this is in git history — removed once the fix was
confirmed on a Colab G4 runtime, 2026-08-03.)

**Versioning (updated 2026-08-02):** do not hand-pin `warp-lang`/`mujoco-warp` —
install `mjlab>=1.5.3` and let it resolve its own pair (warp 1.15.0 +
mujoco-warp 3.10.0.3 as of this writing). History: mjlab 1.5.0 with warp 1.15 /
mujoco-warp 3.10.0.2 crashed env resets (fixed in 3.10.0.3), and pinning back to
warp 1.14 segfaulted the raytracer on Blackwell (sm_120, Colab G4 tier) — warp
1.15's BVH out-of-bounds fix is required there.
