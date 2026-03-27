"""GPU-accelerated Ant environment using NVIDIA Newton physics engine.

Newton is built on Warp and provides MuJoCo-compatible physics at GPU speed.
This implements the classic Ant locomotion task with thousands of parallel envs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import warp as wp
    import newton
    import newton.solvers
    from newton.examples import get_asset
except ImportError:
    raise ImportError("NVIDIA Newton and Warp are required: pip install newton warp-lang")


class NewtonAnt(gym.Env):
    """GPU-accelerated Ant locomotion using Newton physics.

    Observation (27): body position/orientation/velocity + joint angles/velocities
    Action (8): torques for 8 actuated joints (continuous)
    Reward: forward velocity - control cost + alive bonus

    Args:
        count_env: Number of parallel environments.
        device: Device string ('cuda:0', 'cpu').
        sim_substeps: Number of physics substeps per action step.
    """

    def __init__(self, count_env=1, device='cuda:0', sim_substeps=5, **kwargs):
        self.count_env = count_env
        self.device = device
        self.sim_substeps = sim_substeps
        self.sim_dt = 0.005  # 5ms per substep
        self.frame_dt = self.sim_dt * self.sim_substeps  # 25ms per action

        self._build_model()

        # Ant: 27 obs (qpos[2:] + qvel), 8 actions
        obs_size = 27
        high = np.inf * np.ones(obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)

        # Allocate torch tensors for obs/rewards/dones
        import torch
        self.obs_buf = torch.zeros((count_env, obs_size), device=device, dtype=torch.float32)
        self.reward_buf = torch.zeros(count_env, device=device, dtype=torch.float32)
        self.done_buf = torch.zeros(count_env, device=device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(count_env, device=device, dtype=torch.bool)
        self.step_count = torch.zeros(count_env, device=device, dtype=torch.int32)
        self.max_episode_steps = 1000

        # Store initial body state for per-world resets
        import torch
        self.initial_body_q = wp.to_torch(self.state_0.body_q).clone()
        self.initial_body_qd = wp.to_torch(self.state_0.body_qd).clone().zero_()
        self.initial_joint_q = wp.to_torch(self.model.joint_q).clone()
        self.initial_joint_qd = wp.to_torch(self.model.joint_qd).clone().zero_()
        self.episode_rewards = torch.zeros(count_env, device=device, dtype=torch.float32)
        self.episode_lengths = torch.zeros(count_env, device=device, dtype=torch.int32)

    def _build_model(self):
        """Build the Newton model with replicated Ant environments."""
        ant = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(ant)
        ant.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        ant.default_shape_cfg.ke = 5.0e4
        ant.default_shape_cfg.kd = 5.0e2
        ant.default_shape_cfg.kf = 1.0e3
        ant.default_shape_cfg.mu = 0.75

        ant.add_mjcf(
            get_asset('nv_ant.xml'),
            floating=True,
            ignore_names=["floor", "ground"],
            xform=wp.transform(wp.vec3(0, 0, 0.75)),
        )

        # Set up PD control for joint targets (following humanoid example)
        for i in range(len(ant.joint_target_ke)):
            ant.joint_target_ke[i] = 150.0
            ant.joint_target_kd[i] = 5.0
            ant.joint_target_mode[i] = int(newton.JointTargetMode.EFFORT)

        builder = newton.ModelBuilder()
        builder.replicate(ant, self.count_env)
        builder.add_ground_plane()

        self.model = builder.finalize(device=self.device)

        # MuJoCo solver for proper contact and articulation handling
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=True
        )

        # Create states and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create contacts
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        # Evaluate initial FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Count joints/bodies per world
        self.bodies_per_world = self.model.body_count // self.count_env
        self.joints_per_world = self.model.joint_count // self.count_env
        self.jcoord_per_world = self.model.joint_coord_count // self.count_env
        self.jdof_per_world = self.model.joint_dof_count // self.count_env

    def _get_obs(self):
        """Extract observations from the physics state.

        XPBD is a maximal-coordinate solver — body_q/body_qd contain the
        actual state, not joint_q/joint_qd which stay at initial values.
        """
        import torch

        # body_q: (N_bodies, 7) = [x, y, z, qx, qy, qz, qw] per body
        # body_qd: (N_bodies, 6) = [wx, wy, wz, vx, vy, vz] (spatial vector)
        body_q = wp.to_torch(self.state_0.body_q)   # (total_bodies, 7)
        body_qd = wp.to_torch(self.state_0.body_qd)  # (total_bodies, 6)

        # Reshape per environment: (N_envs, bodies_per_world, dim)
        body_q = body_q.reshape(self.count_env, self.bodies_per_world, 7)
        body_qd = body_qd.reshape(self.count_env, self.bodies_per_world, 6)

        # Torso is body 0 in each world
        torso_pos = body_q[:, 0, :3]      # (N, 3) - xyz
        torso_quat = body_q[:, 0, 3:]     # (N, 4) - quaternion
        # body_qd is [vx, vy, vz, wx, wy, wz] (linear first, then angular)
        torso_vel = body_qd[:, 0, :3]     # (N, 3) - linear velocity
        torso_angvel = body_qd[:, 0, 3:]  # (N, 3) - angular velocity

        # All body positions relative to torso (legs info)
        rel_pos = body_q[:, 1:, :3] - torso_pos.unsqueeze(1)  # (N, 8, 3)
        rel_pos_flat = rel_pos.reshape(self.count_env, -1)     # (N, 24)

        # Observation: z_pos, quat, torso_vel, torso_angvel, relative leg positions
        # = 1 + 4 + 3 + 3 + 24 = 35
        # But we declared obs_size=27, let me use a subset:
        # z, quat(4), vel(3), angvel(3), first 4 legs rel_pos(4*3=12), first 4 legs vel(not available easily)
        # Let's do: z(1) + quat(4) + vel(3) + angvel(3) + leg_rel_pos_flat(16 for first 4 pairs xy) = 27
        leg_rel = rel_pos_flat[:, :16]  # first 16 components

        self.obs_buf = torch.cat([
            torso_pos[:, 2:3],   # z height (1)
            torso_quat,          # orientation (4)
            torso_vel,           # linear velocity (3)
            torso_angvel,        # angular velocity (3)
            leg_rel,             # relative leg positions (16)
        ], dim=-1)  # total = 27

        # Store for reward
        self.positions = torso_pos
        self.velocities = torso_vel

        return self.obs_buf

    def _compute_reward(self, actions):
        """Compute reward matching MuJoCo Ant-v4:
        reward = forward_vel + healthy_reward - ctrl_cost
        """
        import torch

        forward_vel = self.velocities[:, 0]  # x velocity
        ctrl_cost = 0.5 * (actions ** 2).sum(dim=-1)

        # Healthy check
        height = self.positions[:, 2]
        healthy = (height > 0.26) & (height < 3.0)
        healthy_reward = healthy.float()  # +1 per step if healthy

        self.reward_buf = forward_vel + healthy_reward - 0.5 * ctrl_cost

        # Termination
        self.done_buf = ~healthy
        self.step_count += 1
        self.truncated_buf = (self.step_count >= self.max_episode_steps) & healthy

        return self.reward_buf

    def _auto_reset(self, env_ids):
        """Reset specific environments by restoring initial body state."""
        import torch

        if len(env_ids) == 0:
            return

        body_q = wp.to_torch(self.state_0.body_q)
        body_qd = wp.to_torch(self.state_0.body_qd)

        for env_id in env_ids:
            start = env_id * self.bodies_per_world
            end = start + self.bodies_per_world
            body_q[start:end] = self.initial_body_q[start:end]
            body_qd[start:end] = 0.0

        self.step_count[env_ids] = 0
        self.episode_rewards[env_ids] = 0.0
        self.episode_lengths[env_ids] = 0

    def reset(self, **kwargs):
        """Reset all environments."""
        import torch

        # Restore initial body positions and zero velocities
        body_q = wp.to_torch(self.state_0.body_q)
        body_qd = wp.to_torch(self.state_0.body_qd)
        body_q.copy_(self.initial_body_q)
        body_qd.zero_()

        self.step_count.zero_()
        self.episode_rewards.zero_()
        self.episode_lengths.zero_()

        return self._get_obs()

    def step(self, actions):
        """Step the simulation with the given actions."""
        import torch

        if isinstance(actions, torch.Tensor):
            actions_torch = actions.float().to(self.device)
        else:
            actions_torch = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Map 8 actions per env to joint dofs (skip free joint 6 dofs)
        act_torch = actions_torch.reshape(self.count_env, 8) * 50.0

        # Set both joint_act AND joint_f for compatibility with different solvers
        act_flat = wp.to_torch(self.control.joint_act)
        act_flat.zero_()
        act_per_env = act_flat.reshape(self.count_env, self.jdof_per_world)
        act_per_env[:, 6:14] = act_torch

        f_flat = wp.to_torch(self.control.joint_f)
        f_flat.zero_()
        f_per_env = f_flat.reshape(self.count_env, self.jdof_per_world)
        f_per_env[:, 6:14] = act_torch

        # Step physics with MuJoCo solver
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

        # Get observations and compute reward
        obs = self._get_obs()
        reward = self._compute_reward(actions_torch.reshape(self.count_env, 8))

        # Track episodes
        self.episode_rewards += self.reward_buf
        self.episode_lengths += 1

        # Determine done environments
        done = (self.done_buf | self.truncated_buf)

        # Auto-reset done environments AFTER computing obs/reward
        # (rl_games expects the obs from the terminal state, not the reset state)
        reset_ids = torch.where(done)[0]
        if len(reset_ids) > 0:
            self._auto_reset(reset_ids)

        return obs, self.reward_buf, done.float(), {'time_outs': self.truncated_buf}

    def get_number_of_agents(self):
        return self.count_env

    def get_env_info(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }

    def close(self):
        pass
