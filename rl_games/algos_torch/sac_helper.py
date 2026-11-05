import torch
from torch import distributions as pyd
import math
import torch.nn.functional as F
import numpy as np


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


def resolve_action_groups(spec, action_dim):
    """Action groups as lists of action indices, or None for one group over every dim.

    ``spec`` maps a group name to its indices (ints or "start:stop" ranges), or is a list of such lists.
    The groups must partition range(action_dim).
    """
    if not spec:
        return None
    entries = spec.values() if isinstance(spec, dict) else spec
    groups = []
    for group in entries:
        ids = []
        for entry in group:
            if isinstance(entry, str) and ':' in entry:
                start, stop = (int(x) for x in entry.split(':'))
                ids.extend(range(start, stop))
            else:
                ids.append(int(entry))
        groups.append(ids)
    group_masks(groups, action_dim)  # validates
    return groups


def group_masks(groups, action_dim):
    """(H, A) bool masks of the action groups."""
    flat = sorted(i for ids in groups for i in ids)
    if flat != list(range(action_dim)):
        raise ValueError(f'action groups must partition range({action_dim}), got {groups}')
    masks = torch.zeros(len(groups), action_dim, dtype=torch.bool)
    for h, ids in enumerate(groups):
        masks[h, ids] = True
    return masks


def gaussian_entropy(num_dims, sigma):
    """Entropy of an isotropic Gaussian with std sigma: 0.5 n ln(2 pi e sigma^2)."""
    return 0.5 * num_dims * math.log(2.0 * math.pi * math.e * sigma ** 2)


def min_twin_log_probs(q, log_probs):
    """Per sample, the categorical log-probs of the twin with the lower expected Q.

    q (2, N), log_probs (2, N, bins) -> (N, bins).
    """
    twin = q.argmin(dim=0)
    return log_probs.gather(0, twin.view(1, -1, 1).expand(1, -1, log_probs.shape[-1])).squeeze(0)


def project_categorical(probs, reward, terminated, entropy_bonus, discount, v_min, v_max):
    """C51 projection of r + discount * (z - bonus) * (1 - terminated) onto the fixed bins.

    probs (N, bins) on z = linspace(v_min, v_max, bins); reward, terminated, entropy_bonus (N,);
    discount a float or (N,). Shifted atoms are clamped to [v_min, v_max] and each atom's mass is
    split linearly between its two neighbouring bins.
    """
    num_bins = probs.shape[-1]
    probs = probs.float()
    bins = torch.linspace(v_min, v_max, num_bins, device=probs.device)
    if torch.is_tensor(discount):
        discount = discount.float().view(-1, 1)
    keep = discount * (1.0 - terminated.float().view(-1, 1))
    atoms = reward.float().view(-1, 1) + keep * (bins.view(1, -1) - entropy_bonus.float().view(-1, 1))
    pos = (atoms.clamp(v_min, v_max) - v_min) * ((num_bins - 1) / (v_max - v_min))
    lower = pos.floor().clamp(0, num_bins - 1)
    upper_w = pos - lower
    lower = lower.long()
    upper = (lower + 1).clamp(max=num_bins - 1)
    out = torch.zeros_like(probs)
    out.scatter_add_(1, lower, probs * (1.0 - upper_w))
    out.scatter_add_(1, upper, probs * upper_w)
    return out


class ReturnScaler:
    """Per-head reward scale from the spread of the running discounted return.

    Per env and head G <- gamma * (1 - done) * G + r. Rewards are divided by
    max(sqrt(Var[G] + 1e-8), max|G| / max_return), so the largest return seen maps to at most
    max_return (the critic support) and typical returns to unit spread.
    """

    def __init__(self, num_heads, gamma, max_return, device):
        self.gamma = gamma
        self.max_return = max_return
        self.returns = None  # (B, H), allocated on the first update
        self.abs_max = torch.zeros(num_heads, device=device)
        self.mean = torch.zeros(num_heads, device=device)
        self.var = torch.ones(num_heads, device=device)
        self.count = torch.zeros((), device=device)

    @torch.no_grad()
    def update(self, rewards, done):
        """rewards (B, H), done (B,) bool."""
        if self.returns is None:
            self.returns = torch.zeros_like(rewards, dtype=torch.float32)
        keep = self.gamma * (~done.bool().view(-1)).float().unsqueeze(-1)
        self.returns = keep * self.returns + rewards.float()
        self.abs_max = torch.maximum(self.abs_max, self.returns.abs().amax(dim=0))
        # parallel merge of the batch moments, prior count 1e-4
        n = float(self.returns.shape[0])
        batch_mean = self.returns.mean(dim=0)
        batch_var = (self.returns.pow(2).mean(dim=0) - batch_mean.pow(2)).clamp(min=0.0)
        delta = batch_mean - self.mean
        total = self.count + n
        m2 = self.var * (self.count + 1e-4) + batch_var * n + delta.pow(2) * self.count * n / total
        self.mean = self.mean + delta * n / total
        self.var = m2 / total
        self.count = total

    def scale(self, rewards):
        """(N, H) -> (N, H)."""
        return rewards / torch.maximum(torch.sqrt(self.var + 1e-8), self.abs_max / self.max_return)

    def state_dict(self):
        return {'returns': self.returns, 'abs_max': self.abs_max, 'mean': self.mean, 'var': self.var,
                'count': self.count}

    def load_state_dict(self, state):
        device = self.mean.device
        self.returns = None if state['returns'] is None else state['returns'].to(device)
        for key in ('abs_max', 'mean', 'var', 'count'):
            setattr(self, key, state[key].to(device))


def truncated_zeta_pmf(mu, n_max):
    """P(n) for n = 1..n_max, proportional to n^-mu."""
    n = torch.arange(1, n_max + 1, dtype=torch.float64)
    pmf = n.pow(-mu)
    return pmf / pmf.sum()


class HeldNoise:
    """Exploration noise held over several steps.

    A standard-normal draw per env and action dim is reused for n consecutive policy steps; n is
    drawn once for all envs from a truncated zeta law P(n) ~ n^-mu on 1..n_max (mu 2, n_max 16:
    n = 1 with probability 0.63, mean 2.13).
    """

    def __init__(self, mu=2.0, n_max=16):
        self.pmf = truncated_zeta_pmf(mu, n_max)
        self.eps = None
        self.hold = 0
        self.used = 0

    def __call__(self, like):
        if self.eps is None or self.eps.shape != like.shape or self.used >= self.hold:
            self.eps = torch.randn_like(like)
            self.hold = int(torch.multinomial(self.pmf, 1).item()) + 1
            self.used = 0
        self.used += 1
        return self.eps
