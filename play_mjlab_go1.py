"""Generate video of trained MJLab Go1 velocity tracking."""
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

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


def run_episode(agent, render_env, is_deterministic, label):
    obs_dict, _ = render_env.reset()

    # Reset robot heading to 0 (face +x direction)
    robot = render_env.scene.entities['robot']
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = robot.data.root_link_pos_w  # keep position
    root_state[:, 3:7] = torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.float32)  # identity quat (face +x)
    robot.data.write_root_state(root_state)
    render_env.scene.write_data_to_sim()

    # Re-get obs after heading reset
    obs_dict = render_env.observation_manager.compute()
    obs = obs_dict['actor']

    # Set velocity commands: walk forward, then turn, then forward again
    cmd_term = render_env.command_manager.get_term("twist")

    frames = []
    total_reward = 0
    steps = 0

    for step_i in range(500):
        if step_i < 150:
            # Walk forward
            cmd_term.vel_command_b[:, 0] = 1.0
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0
        elif step_i < 300:
            # Turn left while walking
            cmd_term.vel_command_b[:, 0] = 0.5
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.7
        else:
            # Walk forward again (new direction)
            cmd_term.vel_command_b[:, 0] = 1.0
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0

        frame = render_env.render()
        frames.append(frame)

        obs_tensor = obs.float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, is_deterministic=is_deterministic)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(obs.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Debug: print action stats and base velocity every 50 steps
        if step_i % 50 == 0:
            act_np = action.cpu().numpy().flatten()
            cmd = cmd_term.vel_command_b[0].cpu().numpy()
            # obs indices: 0-2 = base_lin_vel, 3-5 = base_ang_vel
            base_vel = obs[0, :3].cpu().numpy() if obs.dim() == 2 else obs[:3].cpu().numpy()
            # Also get true body position from sim
            try:
                root_pos = render_env.scene['robot'].data.root_pos_w[0].cpu().numpy()
                pos_str = f"pos=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}]"
            except:
                pos_str = ""
            print(f"  [{label}] step={step_i}: action mean={act_np.mean():.3f} std={act_np.std():.3f} "
                  f"range=[{act_np.min():.3f}, {act_np.max():.3f}] | "
                  f"base_vel=[{base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f}] | "
                  f"cmd=[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}] | {pos_str}")

        obs_dict, reward, terminated, truncated, info = render_env.step(action)
        obs = obs_dict['actor']
        total_reward += reward[0].item()
        steps += 1

        if (terminated | truncated)[0]:
            obs_dict, _ = render_env.reset()
            obs = obs_dict['actor']

    print(f"  [{label}] Total reward: {total_reward:.1f}, Steps: {steps}")
    return frames, total_reward


def main():
    checkpoint = "runs/MJLab_Go1_Velocity_v5_no_curriculum_28-16-53-36/nn/MJLab_Go1_Velocity_v5_no_curriculum.pth"
    config_path = "rl_games/configs/mjlab/ppo_go1_velocity_v4.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['params']['config']['player']['use_vecenv'] = True
    config['params']['config']['num_actors'] = 1
    config['params']['config']['env_config'].pop('num_actors', None)

    env_configurations.register('mjlab_go1_velocity', {
        'vecenv_type': 'MJLAB',
    })
    vecenv.register('MJLAB', lambda config_name, num_actors, **kwargs:
                     MjlabVecEnv(config_name, num_actors, **kwargs))

    runner = Runner()
    runner.load(config)
    runner.reset()
    agent = runner.create_player()
    agent.restore(checkpoint)

    # Print model info
    print(f"\nclip_actions: {agent.clip_actions}")
    print(f"normalize_input: {agent.normalize_input}")
    if hasattr(agent.model, 'running_mean_std'):
        rms = agent.model.running_mean_std
        print(f"running_mean_std mean: {rms.running_mean.cpu().numpy()[:5]}...")
        print(f"running_mean_std var:  {rms.running_var.cpu().numpy()[:5]}...")

    # Create render env
    task_name = "Mjlab-Velocity-Flat-Unitree-Go1"
    cfg = load_env_cfg(task_name)
    cfg.scene.num_envs = 1
    render_env = ManagerBasedRlEnv(cfg, device="cuda", render_mode="rgb_array")

    # Run deterministic with rotation sequence
    print("\n=== DETERMINISTIC (forward → turn → forward) ===")
    frames, reward = run_episode(agent, render_env, True, "play")

    imageio.mimsave("go1_trained.mp4", frames, fps=30)
    print("Video saved to go1_trained.mp4")

    from PIL import Image
    compressed = []
    for f in frames[::2]:
        img = Image.fromarray(f)
        img = img.resize((img.width // 2, img.height // 2))
        compressed.append(np.array(img))
    imageio.mimsave("go1_trained.gif", compressed, fps=15, loop=0)
    print("GIF saved to go1_trained.gif")


if __name__ == "__main__":
    main()
