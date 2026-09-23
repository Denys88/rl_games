"""Eval & rendering tools for envpool dm_control soccer self-play.

Usage:
    python -m rl_games.envs.dmc_soccer_tools video --checkpoint ckpt.pth \
        --out match.mp4 --camera 3 [--opponent checkpoint|chaser|keeper|random]
    python -m rl_games.envs.dmc_soccer_tools tournament \
        --run-dir runs/<experiment>/nn --out tournament.md

The video subcommand renders a match (camera 3 = ball_cam_far is the best
overview; the model's top_down camera renders only skybox). The tournament
subcommand round-robins early/mid/final checkpoints against scripted anchors
and reports a goal-diff/episode cross table.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from rl_games.envs.dmc_soccer_opponents import FrozenPolicy, chaser, keeper
# flatten_obs is the adapter's: eval sees the training features (incl. the
# +/-1e3 clip), so checkpoints play here what they trained on
from rl_games.envs.dmc_soccer_selfplay import (
    NATIVE_ENV_ID, check_player_layout, flatten_obs, sort_batch)

# checkpoint dir of the shipped config (rl_games/configs/dm_control/
# boxhead_soccer_2v2_selfplay.yaml): <train_dir>/<full_experiment_name>/nn,
# the same path its league_ckpt_dir names
DEFAULT_RUN_DIR = "runs/boxhead_soccer_2v2_selfplay/nn"


def make_soccer_env(num_envs, seed, max_episode_steps, team_size=2,
                    env_name=NATIVE_ENV_ID, **kwargs):
    """Native envpool (>= 1.2.7) soccer env; envpool is imported here, not at
    module level, so the module imports without it (as with cv2).

    Returns (env, players): `players` is 2 * team_size, the number of rows
    every batch carries per match. It cannot be read off the observation
    space -- natively each space is one PLAYER's, whose leading axis is
    envpool's stack dim, not a players axis.
    """
    import envpool
    env = envpool.make_gymnasium(env_name, num_envs=num_envs, seed=seed,
                                 team_size=team_size,
                                 max_episode_steps=max_episode_steps, **kwargs)
    return env, 2 * team_size


def reset_sorted(env, num_matches, players):
    """env.reset() with the batch re-sorted into env-major row order."""
    obs, info = env.reset()
    check_player_layout(info, num_matches, players)
    (obs,) = sort_batch(obs, info, num_matches, players)
    return obs


def step_sorted(env, acts, num_matches, players):
    """env.step() with (M, P, A) actions and an env-major sorted batch back.

    Actions go in env-major order -- envpool reads them against
    arange(num_envs) -- and obs/reward/terminated/truncated come back
    permuted, so only the returned batch needs sorting.
    """
    obs, reward, term, trunc, info = env.step(
        np.asarray(acts, dtype=np.float64).reshape(num_matches * players, -1))
    return sort_batch(obs, info, num_matches, players, reward, term, trunc)


class TeamController:
    """Computes actions for one team (slice of players) each step."""

    def __init__(self, kind, ckpt=None):
        self.kind = kind
        self.net = FrozenPolicy(ckpt) if kind == "checkpoint" else None
        self.rng = np.random.RandomState(0)

    def act(self, flat_team, obs_team, act_dim=3):
        """flat_team: (M, T, obs_dim); obs_team: dict sliced to this team."""
        m, t = flat_team.shape[:2]
        if self.kind == "checkpoint":
            a = self.net.act(flat_team.reshape(m * t, -1))
            return a.reshape(m, t, act_dim)
        if self.kind == "chaser":
            return chaser(obs_team)
        if self.kind == "keeper":
            return keeper(obs_team)
        if self.kind == "random":
            return self.rng.uniform(-1, 1, (m, t, act_dim))
        if self.kind == "zero":
            return np.zeros((m, t, act_dim))
        raise ValueError(self.kind)


def team_obs_dict(obs, num_matches, players, team):
    """Slice the per-key obs to one team (0=home first half, 1=away)."""
    ts = players // 2
    sl = slice(0, ts) if team == 0 else slice(ts, players)
    return {
        k: obs[k].reshape(num_matches, players, -1)[:, sl]
        for k in ("ball_ego_position", "team_goal_mid")
    }


def latest_checkpoint(run_dir=DEFAULT_RUN_DIR):
    paths = sorted(glob.glob(os.path.join(run_dir, "*.pth")),
                   key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    return paths[-1]


def video_main(argv=None):
    parser = argparse.ArgumentParser(prog="dmc_soccer_tools video")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="match.mp4")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--opponent", default="checkpoint",
                        choices=["checkpoint", "chaser", "keeper", "random"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--team-size", type=int, default=2,
                        help="players per team (the env's team_size)")
    args = parser.parse_args(argv)

    import cv2
    ckpt = args.checkpoint or latest_checkpoint()
    print(f"home team checkpoint: {ckpt}")

    env, players = make_soccer_env(
        num_envs=1, seed=123, max_episode_steps=900, team_size=args.team_size,
        render_mode="rgb_array", render_width=args.width,
        render_height=args.height)

    home = TeamController("checkpoint", ckpt)
    away = (TeamController("checkpoint", ckpt)
            if args.opponent == "checkpoint"
            else TeamController(args.opponent))
    print(f"match: checkpoint vs {args.opponent}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 40.0,
                             (args.width, args.height))
    obs = reset_sorted(env, 1, players)
    goals = [0, 0]
    ep = 0
    frames = 0
    while ep < args.episodes:
        flat = flatten_obs(obs, 1, players)
        ts = players // 2
        a_home = home.act(flat[:, :ts], team_obs_dict(obs, 1, players, 0))
        a_away = away.act(flat[:, ts:], team_obs_dict(obs, 1, players, 1))
        acts = np.concatenate([a_home, a_away], axis=1)
        obs, reward, term, trunc = step_sorted(env, acts, 1, players)
        kwargs = {} if args.camera is None else {"camera_id": args.camera}
        frame = env.render(**kwargs)[0]
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))  # RGB->BGR
        frames += 1
        # native per-player reward: +1 for the scoring team, -1 for the
        # other; row 0 is a home player, so its sign names the scorer
        pr = reward.reshape(1, players)[0, 0]
        if pr > 0:
            goals[0] += 1
            print(f"  GOAL home (frame {frames})")
        elif pr < 0:
            goals[1] += 1
            print(f"  GOAL away (frame {frames})")
        if (term | trunc)[0]:
            ep += 1
            print(f"episode {ep} done at frame {frames}")
    writer.release()
    print(f"wrote {args.out}: {frames} frames ({frames/40:.0f}s), "
          f"score home {goals[0]} : {goals[1]} away")


def pick_checkpoints(run_dir, count=3):
    """early / mid / final periodic checkpoints by epoch number."""
    paths = glob.glob(os.path.join(run_dir, "last_*_ep_*_rew_*.pth"))
    by_ep = {}
    for p in paths:
        m = re.search(r"_ep_(\d+)_", p)
        if m:
            by_ep[int(m.group(1))] = p
    if not by_ep:
        # an empty list would silently round-robin the scripted anchors only
        # and still print a plausible table
        raise FileNotFoundError(
            f"no last_*_ep_*_rew_*.pth checkpoints in {run_dir}; pass "
            f"--run-dir <train_dir>/<full_experiment_name>/nn of the run")
    eps = sorted(by_ep)
    idx = [0, len(eps) // 2, len(eps) - 1][:count]
    return [(f"ckpt_ep{eps[i]}", by_ep[eps[i]]) for i in dict.fromkeys(idx)]


def play(env, players, home_ctrl, away_ctrl, num_matches, steps):
    """Returns (home_goals, away_goals) totals and episodes played."""
    obs = reset_sorted(env, num_matches, players)
    hg = ag = eps = 0
    ts = players // 2
    for _ in range(steps):
        flat = flatten_obs(obs, num_matches, players)
        a_h = home_ctrl.act(flat[:, :ts], team_obs_dict(obs, num_matches, players, 0))
        a_a = away_ctrl.act(flat[:, ts:], team_obs_dict(obs, num_matches, players, 1))
        obs, reward, term, trunc = step_sorted(
            env, np.concatenate([a_h, a_a], axis=1), num_matches, players)
        pr = reward.reshape(num_matches, players)[:, 0]  # a home player
        hg += (pr > 0).sum()
        ag += (pr < 0).sum()
        eps += (term | trunc).sum()
    return hg, ag, max(eps, 1)


def tournament_main(argv=None):
    parser = argparse.ArgumentParser(prog="dmc_soccer_tools tournament")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--matches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out", default="tournament.md")
    parser.add_argument("--team-size", type=int, default=2,
                        help="players per team (the env's team_size)")
    args = parser.parse_args(argv)

    contenders = pick_checkpoints(args.run_dir)
    contenders += [("chaser", None), ("keeper", None), ("random", None)]
    print("contenders:", [n for n, _ in contenders])

    env, players = make_soccer_env(num_envs=args.matches, seed=999,
                                   max_episode_steps=600,
                                   team_size=args.team_size)

    def ctrl(name, path):
        return (TeamController("checkpoint", path) if path
                else TeamController(name))

    names = [n for n, _ in contenders]
    table = {}
    for i, (na, pa) in enumerate(contenders):
        for j, (nb, pb) in enumerate(contenders):
            if i == j:
                continue
            hg, ag, eps = play(env, players, ctrl(na, pa), ctrl(nb, pb),
                               args.matches, args.steps)
            table[(na, nb)] = (hg - ag) / eps
            print(f"{na:>12} vs {nb:<12} goal-diff/ep = {table[(na, nb)]:+.2f}"
                  f"  ({hg}:{ag} over {eps} eps)")

    lines = ["# 2v2 BoxHead soccer tournament (goal diff/ep, row = home)\n",
             "| home \\ away | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for na in names:
        row = [f"{table.get((na, nb), 0):+.2f}" if na != nb else "—"
               for nb in names]
        lines.append(f"| **{na}** | " + " | ".join(row) + " |")
    avg = {na: np.mean([v for (a, b), v in table.items() if a == na])
           for na in names}
    lines.append("\nAverage goal-diff/ep as home: " + ", ".join(
        f"{n}: {v:+.2f}" for n, v in sorted(avg.items(), key=lambda x: -x[1])))
    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print("\n" + out)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("video", "tournament"):
        print(__doc__)
        sys.exit(1)
    sub, argv = sys.argv[1], sys.argv[2:]
    if sub == "video":
        video_main(argv)
    else:
        tournament_main(argv)


if __name__ == "__main__":
    main()
