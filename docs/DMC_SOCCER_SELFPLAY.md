# dm_control soccer self-play (envpool)

> **Status: experimental.** Research feature; interfaces and reward shaping
> may change. Detailed training results land with the planned self-play
> improvements pass.

Trains dm_control locomotion soccer 2v2 (BoxHead walkers) with PPO and
**symmetric self-play**: a single shared policy controls the home team while
an **opponent league** plays the away team. Observations are egocentric and
team-relative, so the same policy plays either side.

## Install

dm_control soccer is **native in envpool >= 1.2.7** — no fork is needed (the
earlier `BoxheadSoccer2v2-v1` fork build is gone). envpool 1.2.7 ships
cp312–cp314 wheels, so soccer needs **Python >= 3.12**:

```bash
pip install -e ".[envpool]"     # resolves envpool >= 1.2.7 on Python 3.12+
```

On Python 3.11 the extra falls back to envpool 1.2.5, which has no soccer;
use a 3.12+ interpreter for this feature.

Registered env ids (`envpool.list_all_envs()`), aliased
`dm_control/locomotion/soccer_*`:

| env id | walker | action dim | scripted opponents |
| --- | --- | --- | --- |
| `DmcSoccerBoxhead-v1` | BoxHead | 3 (`roll`, `steer`, `kick`) | yes |
| `DmcSoccerAnt-v1` | Ant | 8 | no (see below) |
| `DmcSoccerHumanoid-v1` | Humanoid | 56 | no (see below) |

The shipped config and the scripted league opponents (`chaser`, `keeper`)
encode the 3-DoF BoxHead action layout; the league asserts this and refuses
those types on a wider action space. `random`/`zero`/`league_*` work on any
walker.

### env_config keys

`env_name` (default `DmcSoccerBoxhead-v1`), `team_size` (default 2 — a match
has `2 * team_size` players), and any envpool task option, forwarded as-is:
`terminate_on_goal` (default true), `time_limit` (seconds; 45.0 = 1800 steps
at the 0.025 s control timestep), `enable_field_box`, `disable_walker_contacts`.
`max_episode_steps` is the adapter's own episode cap (default 600) and is what
actually truncates in the shipped config. `max_num_players` is **not** a knob:
envpool derives it as `2 * team_size` and ignores any override.

## Run

```bash
python runner.py --train --file rl_games/configs/dm_control/boxhead_soccer_2v2_selfplay.yaml
python runner.py --play  --file rl_games/configs/dm_control/boxhead_soccer_2v2_selfplay.yaml \
    --checkpoint runs/boxhead_soccer_2v2_selfplay/nn/boxhead_soccer_2v2_selfplay.pth

# render a match to mp4 / run a checkpoint tournament
python -m rl_games.envs.dmc_soccer_tools video --camera 3 --out match.mp4
python -m rl_games.envs.dmc_soccer_tools tournament --run-dir runs/boxhead_soccer_2v2_selfplay/nn
```

## Architecture

- One envpool env = one 2v2 match. `num_actors = matches x 2` controlled home
  players; rl_games sees each player as an independent actor sharing one policy.
- The away team is played by the **league** (`rl_games/envs/dmc_soccer_opponents.py`),
  one type per match: `zero`, `random_weak`, `random`, `chaser_weak`, `chaser`,
  `keeper`, `league_latest`, `league_old`. The last two are **frozen past
  checkpoints** of the training policy (lagged self-play), auto-refreshed from
  the run's checkpoint dir.
- Observations add a **within-team one-hot player id** (home_i and away_i share
  an id), so players can specialize into roles without breaking the home/away
  symmetry that shared-policy self-play needs.

### The native envpool row contract

Verified against envpool 1.2.7; `rl_games/envs/dmc_soccer_selfplay.py` is
written to it and `tests/test_dmc_soccer.py` pins it.

- **Per-player batching.** Every observation value *and the reward* is
  batched over `num_envs * 2 * team_size` rows — `reward` is `(rows,)`
  float32, not one entry per env. `terminated` / `truncated` stay
  **per-match** (`num_envs`), so termination is a property of the match, not
  of a player: a goal ends the episode for all four players at once. The
  adapter repeats each match's `done` across its controlled rows.
- **Row layout.** Rows arrive grouped per match, `2 * team_size` contiguous
  rows each, **home team first** (upstream indexes `team = player /
  team_size`). `info["players"]["env_id"]` labels every row,
  `info["env_id"]` every match.
- **The batch order is permuted.** envpool returns match blocks in
  thread-completion order, so `info["env_id"]` is an arbitrary permutation
  that changes step to step. The adapter re-sorts every batch back to
  env-major order — without that an rl_games row would mean a different
  match each step, corrupting per-row episode bookkeeping and RNN state.
- **Actions are not permuted.** With `env_id=None` envpool fills it with
  `arange(num_envs)`, so the action rows are always read env-major
  regardless of the order the last batch arrived in. Only the returned batch
  needs sorting.
- **Goal signal.** On a goal every player of the scoring team gets `+1` and
  every player of the other `-1` (upstream `rewards[player] = team ==
  scoring_team ? 1 : -1`); with `terminate_on_goal` the match also reports
  `terminated=True`. `stats_home_score` / `stats_away_score` are per-player
  **flags for that step** ("my team just scored" / "just conceded"), not
  running scores — the adapter accumulates goals itself for `scores`.
- **Shaping inputs.** `stats_vel_ball_to_goal` is the ball's velocity
  projected on the ball→opponent-goal direction. `stats_closest_vel_to_ball`
  is zero for every player that is not its team's nearest to the ball, which
  is what makes the `team_chase` sum-broadcast exact.
- **Autoreset is next-step.** The obs returned with `done=True` is the
  terminal obs; the following `step()` ignores its action and returns the new
  episode's first obs with a zero reward. The adapter declares
  `autoreset_mode: "next_step"` so the trainer masks that row.
- **Feature keys.** Upstream names other players `teammate_i_*` /
  `opponent_i_*` (5 suffixes each: `ego_position`, `ego_linear_velocity`,
  `ego_end_effectors_pos`, `ego_orientation`, `end_effectors_pos`), so team
  membership is carried by the key layout and needs no `is_teammate` flag.
  Flat obs width is 113 for `team_size: 2`, 70 for `team_size: 1`.

## Reward design — and the failure mode each piece prevents

```
r = goal_w_score * max(player_reward, 0)
  + dense(t) * vel_ball_w   * max(vel_ball_to_goal, 0)
  + dense(t) * vel_player_w * team_chase
  - time_w
```

Every term was added in response to an observed, reproducible failure:

1. **No concede penalty** (`goal_w_concede: 0`). Punishing concedes teaches
   ball-avoidance — the policy avoids `-goal_w` by never touching the ball.
2. **Goal >> dense stream.** Scoring *terminates* the episode, so if the dense
   shaping outweighs the discounted goal reward, *not finishing* is optimal
   (dribble-farming: reward rises while goals fall).
3. **One-sided ball progress** (`max(vel_ball_to_goal, 0)`). The two-sided
   version punishes the team whenever the *opponent* attacks — uncontrollable
   negatives teach learned helplessness.
4. **Team-level chase** (`team_chase: true`). Per-player chase rewards make
   everyone crowd the ball. One player near the ball is enough: the closest
   player's vel-to-ball is shared with the whole team, freeing the teammate
   to position.
5. **Opponent league.** Training against a single opponent type collapses
   late (overfit + defense lock). Scripted anchors + lagged self-play keep
   the optimization target diverse.
6. **Dense anneal** (`dense_anneal_steps`, `dense_floor`). Shaped rewards
   bootstrap chase/dribble in minutes but then cap skill at the proxy
   equilibrium. Annealing dense terms to a floor raises the effective goal
   weight ~6.7x over training; in our runs this broke a long goals/episode
   plateau (0.34 -> 0.46, still rising at the end).
7. **`entropy_coef: 0`.** With `fixed_sigma` an entropy bonus eventually
   dominates the mastered-reward gradient and inflates log-sigma without
   bound (observed entropy 5.5 -> 44), melting the policy into noise. The
   league provides exploration.

`flatten_obs` additionally zeroes NaN/inf and clips features to ±1e3: strong
kicks can destabilize MuJoCo, and the clip keeps diverged physics out of the
observation normalizer.

## Results (laptop CPU, 256 matches / 512 actors, ~64k frames/s end-to-end)

> Measured on the pre-1.2.7 fork build, before the port to native envpool.
> The reward shaping and the feature set are unchanged, but these numbers
> have not been reproduced on 1.2.7.

1.64B frames over 50k epochs, no collapses. Tournament vs fixed anchors
(goal-diff/episode): chaser +0.46, keeper +0.56, random +0.53; the final
checkpoint beats the mid-training one home and away. Monitoring note: in
self-play the goal *difference* averages zero and league opponents harden
with the policy, so track goals/episode (`scores/mean`) and episode length,
and use fixed-anchor tournaments for absolute skill.
