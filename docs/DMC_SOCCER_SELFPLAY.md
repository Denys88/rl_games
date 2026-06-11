# dm_control soccer self-play (EnvPool)

Trains dm_control locomotion soccer 2v2 (BoxHead walkers) with PPO and
**symmetric self-play**: a single shared policy controls the home team while
an **opponent league** plays the away team. Observations are egocentric and
team-relative, so the same policy plays either side.

Requires the EnvPool dm_control soccer envs (`BoxheadSoccer2v2-v1`) —
available from [envpool PR #1](https://github.com/Denys88/envpool/pull/1)
until released. For dev wheels built from source, point
`ENVPOOL_ASSETS_PATH` at the generated assets if the wheel ships without them.

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

- One EnvPool env = one 2v2 match. `num_actors = matches x 2` controlled home
  players; rl_games sees each player as an independent actor sharing one policy.
- The away team is played by the **league** (`rl_games/envs/dmc_soccer_opponents.py`),
  one type per match: `zero`, `random_weak`, `random`, `chaser_weak`, `chaser`,
  `keeper`, `league_latest`, `league_old`. The last two are **frozen past
  checkpoints** of the training policy (lagged self-play), auto-refreshed from
  the run's checkpoint dir.
- Observations add a **within-team one-hot player id** (home_i and away_i share
  an id), so players can specialize into roles without breaking the home/away
  symmetry that shared-policy self-play needs.

## Reward design — and the failure mode each piece prevents

```
r = goal_w_score * max(players_reward, 0)
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

Also required: envpool's dmc physics-divergence fix (episodes terminate on
bad qacc/qpos instead of leaking NaN observations — strong kicks can
destabilize MuJoCo), included in the envpool PR.

## Results (laptop CPU, 256 matches / 512 actors, ~64k frames/s end-to-end)

1.64B frames over 50k epochs, no collapses. Tournament vs fixed anchors
(goal-diff/episode): chaser +0.46, keeper +0.56, random +0.53; the final
checkpoint beats the mid-training one home and away. Monitoring note: in
self-play the goal *difference* averages zero and league opponents harden
with the policy, so track goals/episode (`scores/mean`) and episode length,
and use fixed-anchor tournaments for absolute skill.
