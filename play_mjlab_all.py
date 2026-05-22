"""Generate videos for all trained MJLab environments."""
import os
if os.path.isdir('/usr/lib/wsl/lib') and '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    import sys
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
import numpy as np
import yaml
import imageio
import warp as wp
wp.init()

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.envs.mjlab_vecenv import MjlabVecEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# All MJLab env registrations
MJLAB_ENVS = [
    'mjlab_go1_velocity', 'mjlab_go1_velocity_rough',
    'mjlab_g1_velocity', 'mjlab_g1_velocity_flat', 'mjlab_g1_velocity_rough',
    'mjlab_g1_tracking', 'mjlab_yam_lift_cube',
]

CONFIGS = [
    {
        'name': 'Go1 Flat Velocity',
        'config': 'rl_games/configs/mjlab/ppo_go1_velocity_v4.yaml',
        'checkpoint': 'runs/MJLab_Go1_Velocity_v5_no_curriculum_28-22-43-11/nn/MJLab_Go1_Velocity_v5_no_curriculum.pth',
        'task': 'Mjlab-Velocity-Flat-Unitree-Go1',
        'video': 'mjlab_go1_flat.mp4',
        'type': 'velocity',
    },
    {
        'name': 'Go1 Rough Velocity',
        'config': 'rl_games/configs/mjlab/ppo_go1_velocity_rough.yaml',
        'checkpoint': 'runs/MJLab_Go1_Velocity_Rough_28-23-09-02/nn/MJLab_Go1_Velocity_Rough.pth',
        'task': 'Mjlab-Velocity-Rough-Unitree-Go1',
        'video': 'mjlab_go1_rough.mp4',
        'type': 'velocity',
    },
    {
        'name': 'G1 Flat Velocity',
        'config': 'rl_games/configs/mjlab/ppo_g1_velocity_flat.yaml',
        'checkpoint': 'runs/MJLab_G1_Velocity_Flat_28-23-44-35/nn/MJLab_G1_Velocity_Flat.pth',
        'task': 'Mjlab-Velocity-Flat-Unitree-G1',
        'video': 'mjlab_g1_flat.mp4',
        'type': 'velocity',
    },
    {
        'name': 'G1 Rough Velocity',
        'config': 'rl_games/configs/mjlab/ppo_g1_velocity_rough.yaml',
        'checkpoint': 'runs/MJLab_G1_Velocity_Rough_29-00-42-36/nn/MJLab_G1_Velocity_Rough.pth',
        'task': 'Mjlab-Velocity-Rough-Unitree-G1',
        'video': 'mjlab_g1_rough.mp4',
        'type': 'velocity',
    },
    {
        'name': 'Yam Lift Cube',
        'config': 'rl_games/configs/mjlab/ppo_yam_lift_cube.yaml',
        'checkpoint': 'runs/MJLab_Yam_Lift_Cube_29-01-32-01/nn/MJLab_Yam_Lift_Cube.pth',
        'task': 'Mjlab-Lift-Cube-Yam',
        'video': 'mjlab_yam_lift_cube.mp4',
        'type': 'manipulation',
    },
]


def register_envs():
    for env_name in MJLAB_ENVS:
        env_configurations.register(env_name, {'vecenv_type': 'MJLAB'})


def create_agent(config_path, checkpoint):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['params']['config']['player']['use_vecenv'] = True
    config['params']['config']['num_actors'] = 1
    config['params']['config']['env_config'].pop('num_actors', None)

    runner = Runner()
    runner.load(config)
    runner.reset()
    agent = runner.create_player()
    agent.restore(checkpoint)
    return agent


def run_velocity_episode(agent, render_env, num_steps=500):
    """Run velocity tracking episode with forward/turn/forward commands."""
    obs_dict, _ = render_env.reset()

    # Reset heading to face +x
    robot = render_env.scene.entities['robot']
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = robot.data.root_link_pos_w
    root_state[:, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.float32)
    robot.data.write_root_state(root_state)
    render_env.scene.write_data_to_sim()

    obs_dict = render_env.observation_manager.compute()
    obs = obs_dict['actor']

    cmd_term = render_env.command_manager.get_term("twist")
    frames = []
    total_reward = 0

    for step_i in range(num_steps):
        if step_i < num_steps * 0.3:
            cmd_term.vel_command_b[:, 0] = 1.0
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0
        elif step_i < num_steps * 0.6:
            cmd_term.vel_command_b[:, 0] = 0.5
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.7
        else:
            cmd_term.vel_command_b[:, 0] = 1.0
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0

        frame = render_env.render()
        frames.append(frame)

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
        total_reward += reward[0].item()

        if (terminated | truncated)[0]:
            obs_dict, _ = render_env.reset()
            obs = obs_dict['actor']

    return frames, total_reward


def run_manipulation_episode(agent, render_env, num_steps=400):
    """Run manipulation episode (lift cube)."""
    obs_dict, _ = render_env.reset()
    obs = obs_dict['actor']

    frames = []
    total_reward = 0

    for step_i in range(num_steps):
        frame = render_env.render()
        frames.append(frame)

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
        total_reward += reward[0].item()

        if (terminated | truncated)[0]:
            obs_dict, _ = render_env.reset()
            obs = obs_dict['actor']

    return frames, total_reward


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None,
                       help='Run only this env (e.g. "go1_flat", "yam")')
    args = parser.parse_args()

    register_envs()

    for cfg in CONFIGS:
        if args.env and args.env not in cfg['video']:
            continue

        if not os.path.exists(cfg['checkpoint']):
            print(f"Skipping {cfg['name']} - checkpoint not found: {cfg['checkpoint']}")
            continue

        print(f"\n{'='*60}")
        print(f"Generating video: {cfg['name']}")
        print(f"{'='*60}")

        agent = create_agent(cfg['config'], cfg['checkpoint'])

        # Create render env
        env_cfg = load_env_cfg(cfg['task'])
        env_cfg.scene.num_envs = 1

        # Disable curriculum for play
        if hasattr(env_cfg, 'curriculum'):
            env_cfg.curriculum = {}
        if hasattr(env_cfg, 'commands') and 'twist' in env_cfg.commands:
            env_cfg.commands['twist'].rel_standing_envs = 0.0

        render_env = ManagerBasedRlEnv(env_cfg, device="cuda", render_mode="rgb_array")

        if cfg['type'] == 'velocity':
            frames, reward = run_velocity_episode(agent, render_env)
        else:
            frames, reward = run_manipulation_episode(agent, render_env)

        print(f"  Total reward: {reward:.1f}")

        imageio.mimsave(cfg['video'], frames, fps=30)
        print(f"  Video saved: {cfg['video']}")

        render_env.close()
        del render_env, agent
        torch.cuda.empty_cache()

    print("\nAll videos generated!")


if __name__ == "__main__":
    main()
