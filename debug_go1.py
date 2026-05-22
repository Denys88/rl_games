"""Debug Go1 - check normalization stats and true position."""
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import numpy as np
import yaml
import warp as wp
wp.init()

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.envs.mjlab_vecenv import MjlabVecEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def main():
    checkpoint = "runs/MJLab_Go1_Velocity_v5_no_curriculum_28-16-53-36/nn/MJLab_Go1_Velocity_v5_no_curriculum.pth"
    config_path = "rl_games/configs/mjlab/ppo_go1_velocity_v4.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['params']['config']['player']['use_vecenv'] = True
    config['params']['config']['num_actors'] = 1
    config['params']['config']['env_config'].pop('num_actors', None)

    env_configurations.register('mjlab_go1_velocity', {'vecenv_type': 'MJLAB'})
    vecenv.register('MJLAB', lambda config_name, num_actors, **kwargs:
                     MjlabVecEnv(config_name, num_actors, **kwargs))

    runner = Runner()
    runner.load(config)
    runner.reset()
    agent = runner.create_player()
    agent.restore(checkpoint)

    # Print full running_mean_std
    if hasattr(agent.model, 'running_mean_std'):
        rms = agent.model.running_mean_std
        mean = rms.running_mean.cpu().numpy().flatten()
        var = rms.running_var.cpu().numpy().flatten()
        std = np.sqrt(var + 1e-5)
        print("\n=== Running Mean/Std (all 48 obs dims) ===")
        obs_names = ['lin_vel_x', 'lin_vel_y', 'lin_vel_z',
                     'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                     'grav_x', 'grav_y', 'grav_z'] + \
                    [f'jpos_{i}' for i in range(12)] + \
                    [f'jvel_{i}' for i in range(12)] + \
                    [f'act_{i}' for i in range(12)] + \
                    ['cmd_vx', 'cmd_vy', 'cmd_wz']
        for i, name in enumerate(obs_names):
            print(f"  [{i:2d}] {name:12s}: mean={mean[i]:8.4f}, std={std[i]:8.4f}")

        # Check what happens when cmd=1.5 is normalized
        print(f"\n=== Command normalization ===")
        print(f"  cmd_vx=1.5 -> normalized: {(1.5 - mean[45]) / std[45]:.2f}")
        print(f"  cmd_vx=1.0 -> normalized: {(1.0 - mean[45]) / std[45]:.2f}")
        print(f"  cmd_vx=0.5 -> normalized: {(0.5 - mean[45]) / std[45]:.2f}")
        print(f"  cmd_vx=0.0 -> normalized: {(0.0 - mean[45]) / std[45]:.2f}")

    # Create render env and check true root position
    task_name = "Mjlab-Velocity-Flat-Unitree-Go1"
    cfg = load_env_cfg(task_name)
    cfg.scene.num_envs = 1
    render_env = ManagerBasedRlEnv(cfg, device="cuda", render_mode="rgb_array")

    obs_dict, _ = render_env.reset()
    obs = obs_dict['actor']

    cmd_term = render_env.command_manager.get_term("twist")
    cmd_term.vel_command_b[:, 0] = 1.0
    cmd_term.vel_command_b[:, 1] = 0.0
    cmd_term.vel_command_b[:, 2] = 0.0

    # Try to find root position
    print(f"\n=== Env internals ===")
    print(f"scene type: {type(render_env.scene)}")
    print(f"scene dir: {[x for x in dir(render_env.scene) if not x.startswith('_')]}")

    # Check articulations
    if hasattr(render_env.scene, 'articulations'):
        print(f"articulations: {render_env.scene.articulations}")
    if hasattr(render_env.scene, 'robot'):
        print(f"robot: {render_env.scene.robot}")

    # Try different ways to get root state
    for attr in ['robot', 'articulations', 'rigid_objects']:
        if hasattr(render_env.scene, attr):
            obj = getattr(render_env.scene, attr)
            print(f"\nscene.{attr} = {obj}")
            if hasattr(obj, 'data'):
                print(f"  data = {obj.data}")
                if hasattr(obj.data, 'root_pos_w'):
                    print(f"  root_pos_w = {obj.data.root_pos_w}")

    # Run 200 steps and track displacement
    positions = []
    for step_i in range(200):
        cmd_term.vel_command_b[:, 0] = 1.0
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0

        obs_tensor = obs.float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, is_deterministic=True)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(obs.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        obs_dict, reward, terminated, truncated, info = render_env.step(action)
        obs = obs_dict['actor']

        # Try to get position from sim
        try:
            sim = render_env.sim
            qpos = sim.data.qpos
            pos = wp.to_torch(qpos).clone().cpu().numpy()[0, :3]
            positions.append(pos.copy())
        except Exception as e:
            if step_i == 0:
                print(f"Can't get qpos: {e}")
            # Try entities
            try:
                robot = render_env.scene.entities['robot']
                if step_i == 0:
                    print(f"  robot type: {type(robot)}")
                    print(f"  robot dir: {[x for x in dir(robot) if not x.startswith('_') and not callable(getattr(robot, x, None))]}")
                    data = robot.data
                    print(f"  data type: {type(data)}")
                    print(f"  data dir: {[x for x in dir(data) if not x.startswith('_')]}")
                data = robot.data
                if hasattr(data, 'root_link_pos_w'):
                    pos = data.root_link_pos_w[0].cpu().numpy()
                    if step_i % 25 == 0:
                        vel_b = data.root_link_lin_vel_b[0].cpu().numpy()
                        vel_w = data.root_link_lin_vel_w[0].cpu().numpy()
                        ang_vel_b = data.root_link_ang_vel_b[0].cpu().numpy()
                        heading = data.heading_w[0].cpu().numpy() if hasattr(data, 'heading_w') else None
                        quat = data.root_link_quat_w[0].cpu().numpy()
                        print(f"  step={step_i}: pos=[{pos[0]:.2f},{pos[1]:.2f}] "
                              f"vel_b=[{vel_b[0]:.2f},{vel_b[1]:.2f}] "
                              f"vel_w=[{vel_w[0]:.2f},{vel_w[1]:.2f}] "
                              f"ang_vel_z={ang_vel_b[2]:.3f} "
                              f"heading={heading:.2f}" if heading is not None else
                              f"  step={step_i}: pos=[{pos[0]:.2f},{pos[1]:.2f}] "
                              f"vel_b=[{vel_b[0]:.2f},{vel_b[1]:.2f}] "
                              f"vel_w=[{vel_w[0]:.2f},{vel_w[1]:.2f}] "
                              f"ang_vel_z={ang_vel_b[2]:.3f} "
                              f"quat={quat}")
                else:
                    pos = None
                if pos is not None:
                    positions.append(pos.copy())
            except Exception as e2:
                if step_i == 0:
                    print(f"Can't get entity pos either: {e2}")
                    import traceback; traceback.print_exc()
                break

        if (terminated | truncated)[0]:
            obs_dict, _ = render_env.reset()
            obs = obs_dict['actor']

    if positions:
        positions = np.array(positions)
        print(f"\n=== Position over {len(positions)} steps ===")
        print(f"Start: {positions[0]}")
        print(f"End:   {positions[-1]}")
        print(f"Displacement: {positions[-1] - positions[0]}")
        print(f"Total distance: {np.linalg.norm(positions[-1][:2] - positions[0][:2]):.3f} m")

    render_env.close()


if __name__ == "__main__":
    main()
