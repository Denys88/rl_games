"""GPU-accelerated Ant environment using NVIDIA Newton physics engine.

Based on Isaac Lab's LocomotionEnv Ant implementation.
Uses Newton/MuJoCo solver for GPU physics with proper observations,
reward shaping, and per-environment reset.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import math

try:
    import warp as wp
    import newton
    import newton.solvers
    from newton.examples import get_asset
except ImportError:
    raise ImportError("NVIDIA Newton and Warp are required: pip install newton warp-lang")


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q. q is [qx, qy, qz, qw]."""
    q_w = q[:, 3:4]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


class NewtonAnt(gym.Env):
    """GPU-accelerated Ant locomotion using Newton physics.

    Follows Isaac Lab's LocomotionEnv pattern:
    - Observations: z_height, local_vel, local_angvel, dof_pos, dof_vel, actions
    - Reward: forward_vel + healthy - ctrl_cost
    - Actions: joint efforts scaled by gear ratio

    Args:
        count_env: Number of parallel environments.
        device: Device string ('cuda:0', 'cpu').
        sim_substeps: Physics substeps per action step.
        action_scale: Multiplier for actions before applying as torques.
    """

    def __init__(self, count_env=1, device='cuda:0', sim_substeps=5,
                 action_scale=0.5, solver_type='mujoco', **kwargs):
        self.count_env = count_env
        self.device = device
        self.sim_substeps = sim_substeps
        self.sim_dt = 0.005
        self.frame_dt = self.sim_dt * self.sim_substeps
        self.action_scale = action_scale
        self.solver_type = solver_type

        self._build_model()

        # Observation: z(1) + quat(4) + vel(3) + angvel(3) + leg_rel_pos(16) = 27
        self.obs_size = 27
        high = np.inf * np.ones(self.obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)

        # Buffers
        self.obs_buf = torch.zeros((count_env, self.obs_size), device=device, dtype=torch.float32)
        self.reward_buf = torch.zeros(count_env, device=device, dtype=torch.float32)
        self.done_buf = torch.zeros(count_env, device=device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(count_env, device=device, dtype=torch.bool)
        self.step_count = torch.zeros(count_env, device=device, dtype=torch.int32)
        self.prev_actions = torch.zeros((count_env, 8), device=device, dtype=torch.float32)
        self.max_episode_steps = 1000
        self.termination_height = 0.26

        # Joint gear ratios from nv_ant.xml (all 15)
        self.joint_gears = torch.tensor([15.0] * 8, device=device, dtype=torch.float32)

        # Joint limits for normalization (from nv_ant.xml: hips +-40deg, ankles 30-100 or -100--30)
        self.dof_limits_lower = torch.tensor(
            [-40, 30, -40, -100, -40, -100, -40, 30], device=device, dtype=torch.float32
        ) * math.pi / 180.0
        self.dof_limits_upper = torch.tensor(
            [40, 100, 40, -30, 40, -30, 40, 100], device=device, dtype=torch.float32
        ) * math.pi / 180.0
        self.dof_range = self.dof_limits_upper - self.dof_limits_lower

        # Store initial state
        self.initial_body_q = wp.to_torch(self.state_0.body_q).clone()
        self.initial_body_qd = wp.to_torch(self.state_0.body_qd).clone().zero_()

        # Previous positions for velocity computation
        self.prev_torso_x = torch.zeros(count_env, device=device, dtype=torch.float32)
        self.prev_torso_y = torch.zeros(count_env, device=device, dtype=torch.float32)

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

        for i in range(len(ant.joint_target_ke)):
            ant.joint_target_ke[i] = 150.0
            ant.joint_target_kd[i] = 5.0
            ant.joint_target_mode[i] = int(newton.JointTargetMode.EFFORT)

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(ant, self.count_env)
        builder.add_ground_plane()

        self.model = builder.finalize(device=self.device)

        if self.solver_type == 'xpbd':
            self.solver = newton.solvers.SolverXPBD(self.model)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
            self._use_collide = True
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_contacts=True
            )
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
            self._use_collide = False

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.bodies_per_world = self.model.body_count // self.count_env
        self.joints_per_world = self.model.joint_count // self.count_env
        self.jcoord_per_world = self.model.joint_coord_count // self.count_env
        self.jdof_per_world = self.model.joint_dof_count // self.count_env

    def _get_obs(self):
        """Simple observations: z + quat + vel + angvel + relative leg positions."""
        body_q = wp.to_torch(self.state_0.body_q)    # (N_bodies, 7)
        body_qd = wp.to_torch(self.state_0.body_qd)  # (N_bodies, 6)

        body_q = body_q.reshape(self.count_env, self.bodies_per_world, 7)
        body_qd = body_qd.reshape(self.count_env, self.bodies_per_world, 6)

        # Torso state
        torso_pos = body_q[:, 0, :3]       # (N, 3) xyz
        torso_quat = body_q[:, 0, 3:]      # (N, 4) quaternion
        torso_vel = body_qd[:, 0, :3]      # (N, 3) linear velocity
        torso_angvel = body_qd[:, 0, 3:]   # (N, 3) angular velocity

        # Relative leg positions (x,y only for 8 legs = 16 values)
        leg_pos = body_q[:, 1:, :3]
        rel_pos = leg_pos - torso_pos.unsqueeze(1)
        leg_rel = rel_pos[:, :, :2].reshape(self.count_env, -1)  # (N, 16)

        # Observation: z(1) + quat(4) + vel(3) + angvel(3) + leg_rel(16) = 27
        self.obs_buf = torch.cat([
            torso_pos[:, 2:3],
            torso_quat,
            torso_vel,
            torso_angvel,
            leg_rel,
        ], dim=-1)

        self.torso_pos = torso_pos
        self.torso_vel = torso_vel

        return self.obs_buf

    def _compute_reward(self, actions):
        """Compute reward: forward_vel + healthy - ctrl_cost."""
        # Use XY speed (magnitude) — the ant may move in any horizontal direction
        dx = self.torso_pos[:, 0] - self.prev_torso_x
        dy = self.torso_pos[:, 1] - self.prev_torso_y
        forward_vel = torch.sqrt(dx**2 + dy**2) / self.frame_dt
        forward_vel = forward_vel.clamp(0.0, 10.0)
        self.prev_torso_x = self.torso_pos[:, 0].clone()
        self.prev_torso_y = self.torso_pos[:, 1].clone()

        ctrl_cost = (actions ** 2).sum(dim=-1)

        height = self.torso_pos[:, 2]
        healthy = (height > self.termination_height) & (height < 3.0)
        healthy_reward = healthy.float() * 0.05

        self.reward_buf = 10.0 * forward_vel - 0.005 * ctrl_cost + healthy_reward

        self.done_buf = ~healthy
        self.step_count += 1
        self.truncated_buf = (self.step_count >= self.max_episode_steps) & healthy

        return self.reward_buf

    def _auto_reset(self, env_ids):
        """Reset specific environments by restoring initial body state."""
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
        # Set prev positions to reset position to avoid velocity spike
        body_q = wp.to_torch(self.state_0.body_q).reshape(self.count_env, self.bodies_per_world, 7)
        self.prev_torso_x[env_ids] = body_q[env_ids, 0, 0]
        self.prev_torso_y[env_ids] = body_q[env_ids, 0, 1]
        self.prev_actions[env_ids] = 0.0

    def reset(self, **kwargs):
        """Reset all environments."""
        body_q = wp.to_torch(self.state_0.body_q)
        body_qd = wp.to_torch(self.state_0.body_qd)
        body_q.copy_(self.initial_body_q)
        body_qd.zero_()

        self.step_count.zero_()
        self.prev_actions.zero_()

        # Set prev positions to current (avoid velocity spike on first step)
        body_q_reshaped = body_q.reshape(self.count_env, self.bodies_per_world, 7)
        self.prev_torso_x = body_q_reshaped[:, 0, 0].clone()
        self.prev_torso_y = body_q_reshaped[:, 0, 1].clone()

        return self._get_obs()

    def step(self, actions):
        """Step the simulation with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions_torch = actions.float().to(self.device)
        else:
            actions_torch = torch.tensor(actions, device=self.device, dtype=torch.float32)

        actions_reshaped = actions_torch.reshape(self.count_env, 8)
        self.prev_actions = actions_reshaped.clone()

        # Apply actions: scale by action_scale and joint gears
        torques = actions_reshaped * self.action_scale * self.joint_gears  # (N, 8)

        # Write to joint_f (MuJoCo solver reads this)
        f_flat = wp.to_torch(self.control.joint_f)
        f_flat.zero_()
        f_per_env = f_flat.reshape(self.count_env, self.jdof_per_world)
        f_per_env[:, 6:14] = torques

        # Also write to joint_act
        act_flat = wp.to_torch(self.control.joint_act)
        act_flat.zero_()
        act_per_env = act_flat.reshape(self.count_env, self.jdof_per_world)
        act_per_env[:, 6:14] = torques

        # Step physics
        if self._use_collide:
            self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if not self._use_collide:
            self.solver.update_contacts(self.contacts, self.state_0)

        # Get observations and reward
        obs = self._get_obs()
        reward = self._compute_reward(actions_reshaped)

        done = self.done_buf | self.truncated_buf

        # Auto-reset after computing obs/reward
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
