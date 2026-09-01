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
    """Minimal torch replica of the rl_games actor (mlp + mu) for inference.

    Hardcodes the shipped config's network and checks none of it:
    `activation: elu` (a checkpoint trained with another activation loads
    without error and plays wrong actions), `fixed_sigma: True` with
    `mu_activation: None` (only the mu head is read and the mean is played,
    clamped to [-1, 1]), `normalize_input: True` (the running_mean_std.*
    keys must exist), Linear layers at even actor_mlp indices (no layernorm
    or dropout between them).
    """

    def __init__(self, checkpoint_path):
        import torch

        # mtime read BEFORE the load: a file rewritten during the load then
        # reads as stale at the next refresh instead of being kept
        self.mtime = os.path.getmtime(checkpoint_path)
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

    LEAGUE_TYPES = ("league_latest", "league_old")

    def __init__(self, num_matches, types=None, ckpt_dir=None,
                 refresh_every=500, rng=None):
        self.types = list(types or self.DEFAULT_TYPES)
        self.assign = np.arange(num_matches) % len(self.types)
        self.ckpt_dir = ckpt_dir
        self.refresh_every = refresh_every
        self.rng = rng or np.random.RandomState(0)
        self._step = 0
        # load failures already reported, keyed (path, exception type); a
        # path's keys clear when it loads, so a failure that returns after a
        # transient one is reported again instead of silenced for the run
        self._warned = set()
        self._warned_missing_dir = False
        self._latest = None  # FrozenPolicy
        self._old = None
        if not ckpt_dir and any(t in self.LEAGUE_TYPES for t in self.types):
            print('WARNING: league_latest/league_old configured without '
                  'league_ckpt_dir; they play weak-random for the whole run. '
                  'Set league_ckpt_dir to <train_dir>/<full_experiment_name>/nn')

    def _load(self, path):
        """FrozenPolicy for `path`, or None when the load fails (mid-write
        checkpoint, key mismatch), reported once per (path, exception type)."""
        try:
            net = FrozenPolicy(path)
        except Exception as e:
            # keep the previous nets but say so: a persistent load failure
            # silently degrades the league to weak-random opponents, which
            # invalidates a self-play run while looking like healthy training
            key = (path, type(e))
            if key not in self._warned:
                print(f'WARNING: opponent league checkpoint {path} failed to '
                      f'load ({type(e).__name__}: {e}); keeping previous '
                      f'opponents. Reported once per checkpoint and error.')
                self._warned.add(key)
            return None
        self._warned = {k for k in self._warned if k[0] != path}
        return net

    @staticmethod
    def _stale(net, path):
        # <name>.pth (the best checkpoint) is overwritten in place, so a path
        # match alone would keep its first-loaded weights for the whole run
        return (net is None
                or (net.path, net.mtime) != (path, os.path.getmtime(path)))

    def _refresh_league(self):
        if not self.ckpt_dir:
            return
        if not os.path.isdir(self.ckpt_dir):
            # a train_dir override or experiment rename: the glob below stays
            # empty for the whole run and league_* silently play weak-random
            if not self._warned_missing_dir:
                print(f'WARNING: league_ckpt_dir {self.ckpt_dir} does not '
                      f'exist; league_latest/league_old play weak-random until '
                      f'it does. Expected <train_dir>/<full_experiment_name>/nn '
                      f'of this run.')
                self._warned_missing_dir = True
            return
        paths = sorted(glob.glob(os.path.join(self.ckpt_dir, "*.pth")),
                       key=os.path.getmtime)
        if not paths:
            return  # present but empty before the first save: the warmup
        if self._stale(self._latest, paths[-1]):
            self._latest = self._load(paths[-1]) or self._latest
        if len(paths) > 1:
            pick = paths[self.rng.randint(0, len(paths) - 1)]
            if self._stale(self._old, pick):
                self._old = self._load(pick) or self._old

    def actions(self, obs_away_dict, flat_obs_away):
        """obs_away_dict: per-key arrays sliced to away players (M, P_away, ...)
        flat_obs_away: policy-format obs for away players (M, P_away, obs_dim).
        Returns (M, P_away, act_dim) actions."""
        self._step += 1
        # refresh on the first call and every refresh_every calls after it
        # (`_step % refresh_every == 1` never fires for league_refresh: 1)
        if (self._step - 1) % self.refresh_every == 0:
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
