# MyoSuite

[MyoSuite](https://github.com/MyoHub/myosuite) — musculoskeletal control tasks
(elbow, finger, 39-muscle hand, 80-muscle legs) on MuJoCo. rl_games trains them
through [envpool](ENVPOOL.md); all 398 task ids are available as
`MyoSuite/<task-id>`, e.g. `MyoSuite/myoElbowPose1D6MRandom-v0`.

## Setup

Python 3.12 or 3.13: envpool 1.2.6 ships no Python 3.11 wheels (1.2.5 has
the deterministic-resets bug described below) and myosuite 2.12.2 declares
`<3.14`. myosuite also pins `gymnasium<1.3` and `mujoco<3.7`: installing it
into the training venv downgrades gymnasium 1.3.0 → 1.2.3 and mujoco
3.9.0 → 3.6.0 (dev-lock versions → highest allowed; the rl_games floors
`gymnasium>=1.0`, `mujoco>=3.0` are still met). To keep the lock versions,
install myosuite in a separate venv for evaluation.

```bash
pip install "envpool>=1.2.6"      # vectorized training (1.2.5 has deterministic resets, see below)
pip install myosuite              # native envs: evaluation + video rendering; downgrades gymnasium/mujoco, see above
```

## Configs

| config | task | envs | max_epochs | notes |
|---|---|---|---|---|
| `ppo_myo_elbow.yaml` | myoElbowPose1D6MRandom-v0 | 128 | 200 | solves in ~100 |
| `ppo_myo_finger_pose.yaml` | myoFingerPoseRandom-v0 | 128 | 300 | |
| `ppo_myo_hand_reach.yaml` | myoHandReachRandom-v0 | 128 | 2000 | |
| `ppo_myo_hand_pose.yaml` | myoHandPoseRandom-v0 | 128 | 4000 | LSTM policy |
| `ppo_myo_walk.yaml` | myoLegWalk-v0 | 128 | 1000 | reward plateaus early |
| `ppo_myo_die_reorient.yaml` | myoChallengeDieReorientP1-v0 | 128 | 10000 | hard; unsolved at 5000 |

```bash
python runner.py --train --file rl_games/configs/myosuite/ppo_myo_elbow.yaml
```

## Known limitations

**envpool <= 1.2.5 resets are deterministic** — no state/target randomization, so the `*Random` variants train against a single fixed target and envpool reward overstates performance on the real task ([envpool#432](https://github.com/sail-sg/envpool/issues/432)). Fixed in envpool 1.2.6; policies trained on 1.2.5 must be retrained. The envpool wrapper refuses MyoSuite tasks on envpool < 1.2.6 (`RuntimeError` at env construction); `env_config: allow_deterministic_resets: true` proceeds with a warning, for evaluation and debugging only. Always evaluate final policies on native myosuite.

**Locomotion can overfit exploration noise** — a walk policy may fall under
deterministic actions despite high training reward. Evaluate stochastically too.

## Videos

Restore the checkpoint with the rl_games player and step the native myosuite
env (spaces match envpool), grabbing frames with `mujoco.Renderer`. Best
cameras: `hand_side_inter` for hand tasks, `side_view` for locomotion.
