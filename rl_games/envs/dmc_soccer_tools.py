"""Eval & rendering tools for EnvPool dm_control soccer self-play.

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

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.registration import make_gymnasium

import numpy as np

from rl_games.envs.dmc_soccer_opponents import FrozenPolicy, chaser, keeper
from rl_games.envs.dmc_soccer_selfplay import _OBS_KEYS


def flatten_obs(obs, num_matches, players):
    """envpool dict obs -> (M, P, obs_dim) float32, matching training obs."""
    parts = [obs[k].reshape(num_matches, players, -1) for k in _OBS_KEYS]
    team_size = players // 2
    slot = np.tile(np.arange(team_size), 2)
    onehot = np.broadcast_to(
        np.eye(team_size, dtype=np.float32)[slot][None],
        (num_matches, players, team_size))
    flat = np.concatenate(parts + [onehot], axis=-1).astype(np.float32)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


class TeamController:
    """Computes actions for one team (slice of players) each step."""

    def __init__(self, kind, ckpt=None):
        self.kind = kind
        self.net = FrozenPolicy(ckpt) if kind == "checkpoint" else None
        self.rng = np.random.RandomState(0)

    def act(self, flat_team, obs_team):
        """flat_team: (M, T, obs_dim); obs_team: dict sliced to this team."""
        m, t = flat_team.shape[:2]
        if self.kind == "checkpoint":
            a = self.net.act(flat_team.reshape(m * t, -1))
            return a.reshape(m, t, 3)
        if self.kind == "chaser":
            return chaser(obs_team)
        if self.kind == "keeper":
            return keeper(obs_team)
        if self.kind == "random":
            return self.rng.uniform(-1, 1, (m, t, 3))
        if self.kind == "zero":
            return np.zeros((m, t, 3))
        raise ValueError(self.kind)


def team_obs_dict(obs, num_matches, players, team):
    """Slice the per-key obs to one team (0=home first half, 1=away)."""
    ts = players // 2
    sl = slice(0, ts) if team == 0 else slice(ts, players)
    return {
        k: obs[k].reshape(num_matches, players, -1)[:, sl]
        for k in ("ball_ego_position", "team_goal_mid")
    }

# dev-build assets fallback (same as train.py)



def latest_checkpoint(run_dir="runs/boxhead_2v2_league_long/nn"):
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
    args = parser.parse_args(argv)

    import cv2
    ckpt = args.checkpoint or latest_checkpoint()
    print(f"home team checkpoint: {ckpt}")

    env = make_gymnasium(
        "BoxheadSoccer2v2-v1", num_envs=1, seed=123, max_episode_steps=900,
        render_mode="rgb_array", render_width=args.width,
        render_height=args.height)
    players = env.observation_space["ball_ego_position"].shape[0]

    home = TeamController("checkpoint", ckpt)
    away = (TeamController("checkpoint", ckpt)
            if args.opponent == "checkpoint"
            else TeamController(args.opponent))
    print(f"match: checkpoint vs {args.opponent}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 40.0,
                             (args.width, args.height))
    obs, _ = env.reset()
    goals = [0, 0]
    ep = 0
    frames = 0
    while ep < args.episodes:
        flat = flatten_obs(obs, 1, players)
        ts = players // 2
        a_home = home.act(flat[:, :ts], team_obs_dict(obs, 1, players, 0))
        a_away = away.act(flat[:, ts:], team_obs_dict(obs, 1, players, 1))
        acts = np.concatenate([a_home, a_away], axis=1)
        obs, _, term, trunc, info = env.step(acts)
        kwargs = {} if args.camera is None else {"camera_id": args.camera}
        frame = env.render(**kwargs)[0]
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))  # RGB->BGR
        frames += 1
        pr = info["players_reward"][0, 0]
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
        return []
    eps = sorted(by_ep)
    idx = [0, len(eps) // 2, len(eps) - 1][:count]
    return [(f"ckpt_ep{eps[i]}", by_ep[eps[i]]) for i in dict.fromkeys(idx)]


def play(env, players, home_ctrl, away_ctrl, num_matches, steps):
    """Returns (home_goals, away_goals) totals and episodes played."""
    obs, _ = env.reset()
    hg = ag = eps = 0
    ts = players // 2
    for _ in range(steps):
        flat = flatten_obs(obs, num_matches, players)
        a_h = home_ctrl.act(flat[:, :ts], team_obs_dict(obs, num_matches, players, 0))
        a_a = away_ctrl.act(flat[:, ts:], team_obs_dict(obs, num_matches, players, 1))
        obs, _, term, trunc, info = env.step(
            np.concatenate([a_h, a_a], axis=1))
        pr = info["players_reward"][:, 0]
        hg += (pr > 0).sum()
        ag += (pr < 0).sum()
        eps += (term | trunc).sum()
    return hg, ag, max(eps, 1)


def tournament_main(argv=None):
    parser = argparse.ArgumentParser(prog="dmc_soccer_tools tournament")
    parser.add_argument("--run-dir", default="runs/boxhead_2v2_league_long/nn")
    parser.add_argument("--matches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out", default="tournament.md")
    args = parser.parse_args(argv)

    contenders = pick_checkpoints(args.run_dir)
    contenders += [("chaser", None), ("keeper", None), ("random", None)]
    print("contenders:", [n for n, _ in contenders])

    env = make_gymnasium("BoxheadSoccer2v2-v1", num_envs=args.matches,
                         seed=999, max_episode_steps=600)
    players = env.observation_space["ball_ego_position"].shape[0]

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
        import cv2  # noqa: F401  (lazy: only the video path needs it)
        video_main(argv)
    else:
        tournament_main(argv)


if __name__ == "__main__":
    main()
