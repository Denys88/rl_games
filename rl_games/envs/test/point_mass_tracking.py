"""Point-mass tracking: a GPU-vectorized test task with several action groups.

Each group is a 2-D point mass driven by an acceleration command that must track its own target
moving on a circle. Observations end in a future window of target offsets laid out
(n_future, groups, 2) row-major, for the trailing-window obs encoder. Actions are ordered by axis
([x of every group | y of every group]), so group g owns dims [g, G + g].

Per group and step: reward = exp(-distance / 0.2) - 0.05 * mean(a^2); leaving the 2 m box ends the
episode with -1 for the group at fault. Episodes time out after ``episode_length`` steps. Same-step
autoreset; infos carry 'time_outs', 'final_observation' and the per-group 'reward_heads'.
"""

import math

import torch
from gymnasium import spaces

from rl_games.common.ivecenv import IVecEnv


class PointMassTracking(IVecEnv):
    dt = 0.05
    accel = 8.0
    damping = 2.0
    box = 2.0

    def __init__(self, num_envs, num_groups=2, n_future=10, episode_length=200, device='cpu', seed=0,
                 stagger=True):
        """stagger: the first reset starts every env at a random point of its episode, so time-outs spread out."""
        self.num_envs, self.num_groups, self.n_future = num_envs, num_groups, n_future
        self.episode_length = episode_length
        self.stagger = stagger
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        b, g = num_envs, num_groups
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos, self.vel = z(b, g, 2), z(b, g, 2)
        self.center, self.radius, self.omega, self.phase = z(b, g, 2), z(b, g), z(b, g), z(b, g)
        self.t = torch.zeros(b, dtype=torch.long, device=self.device)
        self.window = n_future * g * 2
        self.obs_dim = 8 * g + self.window
        self.action_dim = 2 * g
        self.action_groups = [[i, g + i] for i in range(g)]

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _target(self, steps):
        """Target position and velocity (B, K, G, 2) at integer steps (B, K)."""
        time_s = steps.float().unsqueeze(-1) * self.dt
        angle = self.omega.unsqueeze(1) * time_s + self.phase.unsqueeze(1)
        cos, sin = angle.cos(), angle.sin()
        pos = self.center.unsqueeze(1) + self.radius[:, None, :, None] * torch.stack([cos, sin], dim=-1)
        vel = (self.radius * self.omega)[:, None, :, None] * torch.stack([-sin, cos], dim=-1)
        return pos, vel

    def _reset(self, envs):
        n, g = int(envs.numel()), self.num_groups
        if n == 0:
            return
        self.center[envs] = self._rand(n, g, 2) - 0.5
        self.radius[envs] = 0.3 + 0.5 * self._rand(n, g)
        sign = torch.where(self._rand(n, g) < 0.5, -1.0, 1.0)
        self.omega[envs] = sign * (0.5 + 1.5 * self._rand(n, g))
        self.phase[envs] = 2 * math.pi * self._rand(n, g)
        self.t[envs] = 0
        target, _ = self._target(self.t.unsqueeze(1))
        self.pos[envs] = target[envs, 0] + (self._rand(n, g, 2) - 0.5)
        self.vel[envs] = 0.0

    def _obs(self):
        b = self.num_envs
        steps = self.t.unsqueeze(1) + torch.arange(self.n_future + 1, device=self.device)
        target, target_vel = self._target(steps)  # (B, 1 + K, G, 2): now, then the future window
        window = target[:, 1:] - self.pos.unsqueeze(1)
        parts = [self.pos, self.vel, target[:, 0] - self.pos, target_vel[:, 0]]
        return torch.cat([p.reshape(b, -1) for p in parts] + [window.reshape(b, -1)], dim=-1)

    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        if self.stagger:
            self.t = (self._rand(self.num_envs) * self.episode_length).long()
            target, _ = self._target(self.t.unsqueeze(1))
            self.pos = target[:, 0] + (self._rand(self.num_envs, self.num_groups, 2) - 0.5)
        return self._obs()

    @torch.no_grad()
    def step(self, actions):
        b, g = self.num_envs, self.num_groups
        a = torch.as_tensor(actions, device=self.device).float().clamp(-1, 1).view(b, 2, g).transpose(1, 2)
        self.vel += self.dt * (self.accel * a - self.damping * self.vel)
        self.pos += self.dt * self.vel
        self.t += 1
        target, _ = self._target(self.t.unsqueeze(1))
        dist = (target[:, 0] - self.pos).norm(dim=-1)  # (B, G)
        out = (self.pos.abs() > self.box).any(dim=-1)
        reward = torch.exp(-dist / 0.2) - 0.05 * a.pow(2).mean(dim=-1) - out.float()
        terminated = out.any(dim=-1)
        truncated = (self.t >= self.episode_length) & ~terminated
        final_obs = self._obs()
        done = terminated | truncated
        self._reset(done.nonzero().squeeze(-1))
        infos = {'time_outs': truncated, 'final_observation': final_obs, 'reward_heads': reward}
        return self._obs(), reward.sum(dim=-1), done, infos

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {
            'observation_space': spaces.Box(-math.inf, math.inf, (self.obs_dim,)),
            'action_space': spaces.Box(-1.0, 1.0, (self.action_dim,)),
            'agents': 1,
            'autoreset_mode': 'same_step',
            'action_groups': self.action_groups,
        }
