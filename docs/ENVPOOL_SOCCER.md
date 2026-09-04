# EnvPool DeepMind soccer — league PPO self-play

Multi-agent 2v2 soccer (`dm_control.locomotion.soccer`) trained with rl_games
PPO against a league of frozen checkpoints, using envpool's native C++ port of
the environment. Boxhead is the default walker; Ant and Humanoid share the
same wrapper (`walker_type`).

## Why envpool

The earlier dm_control wrapper ran one MuJoCo scene per Ray worker at
~1.7k match-steps/s across 24 workers. envpool's C++ soccer runs 256 matches
in one process at ~70k match-steps/s (28 threads, RTX 5090 host), so the same
frame budget costs ~40x less wall-clock and the away team can be driven by
GPU-resident torch policies instead of CPU copies inside Ray workers.

## Installing envpool with soccer

`DmcSoccer{Boxhead,Ant,Humanoid}-v1` landed on envpool `main` on 2026-08-30
(commit af675b7, "[mujoco] Add native dm_control locomotion and Soccer"),
two days *after* the v1.2.6 release, so no PyPI wheel has it yet. envpool
>= 1.2.6 also requires Python 3.12+ (this repo's main venv is 3.11).

The core (`envpool/core`, `envpool/python`) is unchanged between v1.2.6 and
main, so the cheap route is to build only the locomotion extension and drop
it into an installed 1.2.6:

```bash
uv venv --python 3.12 venv312 && source venv312/bin/activate
uv pip install "envpool==1.2.6" "envpool-assets>=0.4.1,<0.5.0"   # 0.4.x carries the soccer assets
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e . --no-deps tensorboard tensorboardX pyyaml psutil setproctitle opencv-python-headless pytest
uv pip install swig cmake ninja                                     # build tools, no sudo needed

# bazelisk binary (no go/npm needed)
curl -sL -o ~/.local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x ~/.local/bin/bazelisk

git clone --depth 1 https://github.com/sail-sg/envpool.git ~/envpool_src
cd ~/envpool_src
git apply /path/to/rl_games/rl_games/envs/envpool_soccer_pitch.patch      # pitch/goal size options (below)
cp third_party/pip_requirements/requirements-release.txt third_party/pip_requirements/requirements.txt
PATH=$VIRTUAL_ENV/bin:~/.local/bin:$PATH USE_BAZEL_VERSION=9.2.0 \
  bazelisk build --config=release //envpool/mujoco/locomotion:locomotion_envpool   # ~1 min

SP=$VIRTUAL_ENV/lib/python3.12/site-packages/envpool
mkdir -p $SP/mujoco/locomotion
cp bazel-bin/envpool/mujoco/locomotion/locomotion_envpool.so envpool/mujoco/locomotion/{__init__.py,registration.py} $SP/mujoco/locomotion/
sed -i 's/^import envpool.mujoco.dmc.registration  # noqa: F401$/&\nimport envpool.mujoco.locomotion.registration  # noqa: F401/' $SP/entry.py
python -c "import envpool; print([n for n in envpool.list_all_envs() if 'DmcSoccer' in n])"
```

Qt (needed only by procgen) and Java are not required for this target.
Once a release after 1.2.6 ships, `pip install envpool` replaces all of this,
minus the pitch patch.

### The pitch-size patch

Upstream soccer only offers `RandomizedPitch` with the dm_control default
bounds (32–48 x 24–36), which is ~70 body lengths for these walkers — random
play never scores. `rl_games/envs/envpool_soccer_pitch.patch` adds three
config keys to `DmcSoccer*` (empty = upstream behaviour):

| key | meaning |
|---|---|
| `pitch_size_min: [x, y]` | `RandomizedPitch(min_size=...)`; if `pitch_size_max` is omitted the pitch is fixed at this size |
| `pitch_size_max: [x, y]` | `RandomizedPitch(max_size=...)` |
| `goal_size: [depth, width, height]` | `Pitch(goal_size=...)`, same convention as dm_control |

The old dm_control wrapper's `field_size=(15, 10)`, `goal_size=(1.0, 6.0, 0.6)`
map 1:1 onto these keys.

## rl_games pieces

| file | role |
|---|---|
| `rl_games/envs/envpool_soccer.py` | `EnvpoolSoccerVecEnv` (vecenv type `ENVPOOL_SOCCER`, env `envpool_soccer`). Home team → learner (parameter sharing, env-major/agent-minor rows). Away team → random or a per-match pool member (frozen torch copy of the learner, GPU). Lean shaped reward from the `stats_*` keys. |
| `rl_games/common/soccer_observer.py` | `SoccerObserver` (match stats, split by opponent kind) and `SoccerLeagueObserver` (payoff, snapshots, PFSP matchmaking, main pushes). |
| `rl_games/common/league.py` | Shared with Go; now framework-agnostic (`host_fn`, lazy jax). |
| `rl_games/configs/envpool/ppo_soccer_boxhead_league.yaml` | The run config. |
| `scripts/soccer_train.py` | Attaches the observer (yaml cannot) and runs the Runner. |
| `tests/test_envpool_soccer.py` | Env layout/rewards/pool assignment + observer + league tests. |

### Env details worth knowing

* envpool returns match rows in *completion order* every step
  (`info['env_id']` is not sorted). The wrapper reorders rows to env-major
  each step via `info['players']['env_id']`; actions are sent env-major with
  the default `env_id`, which envpool maps as `repeat(arange(N), players)`.
* Player order inside a match is home team first, then away team
  (`soccer.cc`: `team = player / team_size`). Observations are ego-centric
  with team-relative goal keys, so the same policy plays either side unchanged.
* envpool auto-reset: the step after a finished match returns the new
  match's first observation with zero reward and ignores the action. The
  wrapper zeroes that transition's reward and applies pending opponent
  assignments there, so an opponent never changes mid-match.
* `stats_closest_vel_to_ball` is non-zero only for the closest teammate;
  the wrapper sums it over the team and shares it, reproducing the
  "closest-teammate ball-chase" reward from the dm_control experiments.
* Match infos at done (per match): `goal_diff`, `home_goals`, `away_goals`,
  `win`, `opp_id`, `match_len`, `scores` (= goal diff, so the default
  observer's `scores/mean` is goal diff).

### League

Opponent ids: `-1` random anchor, `0` latest main, `>0` pool snapshot.
Matchmaking per remap (default every 5 epochs, applied at each match's next
reset): 20% random, 30% main, 40% PFSP-weighted pool (`(1-p)^2`, variance
mode until 200 games), 10% uniform pool. Snapshots every 100 epochs or when
main beats ≥70% of rated members at ≥55%. Pool capped at 16 (evicts the
snapshot main beats hardest). Payoff EMA 0.02.

**Batched pool inference.** The env keeps every member's state dict stacked
along a leading "slot" axis and runs one `torch.func.vmap(functional_call)`
per step: matches are grouped by slot, padded to the largest group, and the
padded rows are masked out. The jit-scripted `RunningMeanStd` children of the
template are swapped for plain modules (TorchScript cannot run under vmap).
Spike numbers, 16 members x 32 rows, cuda: 9.6 ms/step looped vs 1.0 ms vmapped.
`League.build_stacked` / `stack_trees` (rl_games/common/league.py) handle both
torch state dicts and jax pytrees.

**Strategy-diversity knobs (v4, `ppo_soccer_boxhead_league_v4.yaml`).**
`player_id_obs: True` appends a one-hot of the robot's index within its team
to every row (home and away), so the parameter-shared policy can take roles.
`opponent_deterministic: False` makes the away team sample
`mu + sigma * scale * N(0,1)`; `scale` is drawn per match from
`opponent_sigma_scale: [lo, hi]` (redrawn at every match reset), so one
snapshot yields a family of opponents. Observation becomes 111 + 2 dims: v3
checkpoints cannot seed a v4 pool. `scripts/soccer_play.py
--opponent-sigma-scale LO HI` reproduces this at play time.

Parked (tests in the session scratchpad): a per-match unit "style" vector z
in the obs whose components rescale shaping terms (reward-randomised PG), so
that every snapshot is a family of play styles and z is a play-time knob.

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

Tensorboard: `soccer/*` (goal_diff, winrate, drawrate, goals_for/against,
`*_vs_random|self|pool`) and `league/*` (pool_size, min/mean winrate vs pool).
`soccer/goal_diff_vs_random` is the free stand-in for the old 30-episode
"eval vs random".

## Running

```bash
source venv312/bin/activate
python scripts/soccer_train.py -f rl_games/configs/envpool/ppo_soccer_boxhead_league.yaml
python -m pytest tests/test_envpool_soccer.py -q
# round-robin eval of checkpoints (+ vs random); best copied to best_agents/, table in RESULTS.md
python scripts/soccer_eval.py -f rl_games/configs/envpool/ppo_soccer_boxhead_league_v8.yaml \
    --checkpoints runs/<run>/nn/last_*ep_*000_*.pth --matches 16 --tag v8
```

**Reward clamp gotcha.** v1..v7 configs carried `reward_shaper: min_val: -10`
(inherited from the ant configs); rl_games clamps rewards from below, so a
conceded goal arrived as -10 against +100 for scoring and nobody learned to
defend. v8 drops it. v8 also ends the episode on the first goal with a 45 s
limit (Liu et al.), so the return is a single terminal ±100.
