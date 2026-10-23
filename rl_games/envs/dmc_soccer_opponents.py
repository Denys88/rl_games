"""Opponent league for EnvPool dm_control soccer self-play.

Each match is assigned one of several opponent types for the away team.
Scripted opponents act from the away players' own egocentric observations;
league opponents run frozen past checkpoints of the training policy
(lagged self-play), refreshed from the run's checkpoint dir.

BoxHead action conventions: action = [roll, steer, kick], roll = -1 drives
forward (the actuator gear is negative), steer > 0 turns CCW, all in [-1, 1].
"""

import glob
import os

import numpy as np


def _steer_to(ang):
    return np.clip(2.0 * ang, -1, 1)


def chaser(obs_away, strength=1.0):
    """Chase the ball, kick when close."""
    bx = obs_away["ball_ego_position"][..., 0]
    by = obs_away["ball_ego_position"][..., 1]
    ang = np.arctan2(by, bx)
    dist = np.sqrt(bx * bx + by * by)
    roll = np.where(np.abs(ang) < 0.6, -1.0, 0.0) * strength
    steer = _steer_to(ang) * strength
    kick = np.where(dist < 1.0, 1.0, 0.0)
    return np.stack([roll, steer, kick], axis=-1)


def keeper(obs_away):
    """Sit between own goal and the ball; clear when the ball is close."""
    bx = obs_away["ball_ego_position"][..., 0]
    by = obs_away["ball_ego_position"][..., 1]
    gx = obs_away["team_goal_mid"][..., 0]
    gy = obs_away["team_goal_mid"][..., 1]
    # target: 60% of the way from own goal toward the ball
    tx = 0.6 * bx + 0.4 * gx
    ty = 0.6 * by + 0.4 * gy
    ang = np.arctan2(ty, tx)
    dist = np.sqrt(tx * tx + ty * ty)
    roll = np.where((np.abs(ang) < 0.6) & (dist > 0.5), -1.0, 0.0)
    steer = _steer_to(ang)
    ball_dist = np.sqrt(bx * bx + by * by)
    kick = np.where(ball_dist < 1.2, 1.0, 0.0)
    return np.stack([roll, steer, kick], axis=-1)


class FrozenPolicy:
    """Minimal torch replica of the rl_games actor (mlp + mu) for inference."""

    def __init__(self, checkpoint_path):
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
        # torch.compile'd models save keys with an "_orig_mod." prefix
        model = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["model"].items()}
        self.mean = model["running_mean_std.running_mean"].float()
        self.var = model["running_mean_std.running_var"].float()
        self.linears = []
        i = 0
        while f"a2c_network.actor_mlp.{i}.weight" in model:
            self.linears.append((
                model[f"a2c_network.actor_mlp.{i}.weight"].float(),
                model[f"a2c_network.actor_mlp.{i}.bias"].float()))
            i += 2  # Linear layers sit at even indices (activation between)
        self.mu_w = model["a2c_network.mu.weight"].float()
        self.mu_b = model["a2c_network.mu.bias"].float()
        self.path = checkpoint_path

    def act(self, obs):
        import torch

        with torch.no_grad():
            x = torch.from_numpy(obs).float()
            x = torch.clamp(
                (x - self.mean) / torch.sqrt(self.var + 1e-5), -5, 5)
            for w, b in self.linears:
                x = torch.nn.functional.elu(x @ w.T + b)
            mu = x @ self.mu_w.T + self.mu_b
            return torch.clamp(mu, -1, 1).numpy()


class OpponentLeague:
    """Assigns one opponent type per match and computes away-team actions."""

    DEFAULT_TYPES = (
        "zero", "random_weak", "random", "chaser_weak",
        "chaser", "keeper", "league_latest", "league_old",
    )

    def __init__(self, num_matches, types=None, ckpt_dir=None,
                 refresh_every=500, rng=None):
        self.types = list(types or self.DEFAULT_TYPES)
        self.assign = np.arange(num_matches) % len(self.types)
        self.ckpt_dir = ckpt_dir
        self.refresh_every = refresh_every
        self.rng = rng or np.random.RandomState(0)
        self._step = 0
        self._latest = None  # FrozenPolicy
        self._old = None

    def _refresh_league(self):
        if not self.ckpt_dir:
            return
        paths = sorted(glob.glob(os.path.join(self.ckpt_dir, "*.pth")),
                       key=os.path.getmtime)
        if not paths:
            return
        try:
            if self._latest is None or self._latest.path != paths[-1]:
                self._latest = FrozenPolicy(paths[-1])
            if len(paths) > 1:
                pick = paths[self.rng.randint(0, len(paths) - 1)]
                if self._old is None or self._old.path != pick:
                    self._old = FrozenPolicy(pick)
        except Exception:
            pass  # mid-write checkpoint etc.; keep previous nets

    def actions(self, obs_away_dict, flat_obs_away):
        """obs_away_dict: per-key arrays sliced to away players (M, P_away, ...)
        flat_obs_away: policy-format obs for away players (M, P_away, obs_dim).
        Returns (M, P_away, act_dim) actions."""
        self._step += 1
        if self._step % self.refresh_every == 1:
            self._refresh_league()

        m, pa = flat_obs_away.shape[:2]
        acts = np.zeros((m, pa, 3))
        for ti, tname in enumerate(self.types):
            sel = self.assign == ti
            if not sel.any():
                continue
            sub = {k: v[sel] for k, v in obs_away_dict.items()}
            if tname == "zero":
                a = np.zeros((sel.sum(), pa, 3))
            elif tname == "random":
                a = self.rng.uniform(-1, 1, (sel.sum(), pa, 3))
            elif tname == "random_weak":
                a = self.rng.uniform(-0.3, 0.3, (sel.sum(), pa, 3))
            elif tname == "chaser":
                a = chaser(sub)
            elif tname == "chaser_weak":
                a = chaser(sub, strength=0.5)
            elif tname == "keeper":
                a = keeper(sub)
            elif tname in ("league_latest", "league_old"):
                net = self._latest if tname == "league_latest" else self._old
                net = net or self._latest
                if net is None:  # no checkpoint yet: weak random warmup
                    a = self.rng.uniform(-0.3, 0.3, (sel.sum(), pa, 3))
                else:
                    fo = flat_obs_away[sel].reshape(-1, flat_obs_away.shape[-1])
                    a = net.act(fo).reshape(sel.sum(), pa, 3)
            else:
                raise ValueError(f"unknown opponent type {tname}")
            acts[sel] = a
        return acts
