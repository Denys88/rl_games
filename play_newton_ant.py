"""Generate video of trained Newton Ant using MuJoCo renderer.

Since Newton doesn't have a simple RGB renderer, we use MuJoCo to
render the same nv_ant.xml model while replaying the trained policy
through Newton physics.
"""
import torch
import numpy as np
import yaml
import imageio
import newton
import warp as wp
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations
from rl_games.envs.newton_ant import NewtonAnt
import os

# Set LD_LIBRARY_PATH for Warp CUDA in WSL
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

def register_newton_envs():
    env_configurations.register('newton_ant', {
        'env_creator': lambda **kwargs: NewtonAnt(**kwargs),
        'vecenv_type': 'WARP',
    })


def main():
    checkpoint = "runs/Newton_Ant_PPO_27-22-56-18/nn/Newton_Ant_PPO.pth"
    config_path = "rl_games/configs/newton/ppo_newton_ant.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    register_newton_envs()

    # Load the trained agent
    runner = Runner()
    runner.load(config)
    runner.reset()
    agent = runner.create_player()
    agent.restore(checkpoint)

    # Create a single Newton env for evaluation
    import warp as wp
    env = NewtonAnt(count_env=1, device='cuda:0')
    obs = env.reset()

    # Also create a MuJoCo env for rendering
    import mujoco
    from newton.examples import get_asset

    ant_xml = get_asset('nv_ant.xml')
    mj_model = mujoco.MjModel.from_xml_path(ant_xml)
    mj_data = mujoco.MjData(mj_model)

    # Set up offscreen renderer
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    frames = []
    total_reward = 0
    steps = 0

    for step_i in range(500):
        # Get action from trained policy
        obs_tensor = obs.float()
        with torch.no_grad():
            action = agent.get_action(obs_tensor, is_deterministic=True)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Step Newton env
        obs, reward, done, info = env.step(action)
        total_reward += reward[0].item()
        steps += 1

        # Use Newton's inverse kinematics to get joint angles from body state
        newton.eval_ik(env.model, env.state_0, env.state_0.joint_q, env.state_0.joint_qd)
        jq = wp.to_torch(env.state_0.joint_q).cpu().numpy()

        # Newton Z-up -> nv_ant.xml MuJoCo Y-up (floor zaxis="0 1 0")
        # Swap Y<->Z for position, adjust quaternion accordingly
        mj_data.qpos[0] = jq[0]   # x -> x
        mj_data.qpos[1] = jq[2]   # z -> y (height in MuJoCo)
        mj_data.qpos[2] = -jq[1]  # -y -> z
        # Quaternion: Newton [qx,qy,qz,qw] -> swap y/z -> MuJoCo [qw,qx,qy,qz]
        mj_data.qpos[3] = jq[6]   # qw
        mj_data.qpos[4] = jq[3]   # qx
        mj_data.qpos[5] = jq[5]   # qz -> qy
        mj_data.qpos[6] = -jq[4]  # -qy -> qz
        mj_data.qpos[7:15] = jq[7:15]  # 8 joint angles

        mujoco.mj_forward(mj_model, mj_data)

        # Render with tracking camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mj_model.body('torso').id
        cam.distance = 5.0
        cam.azimuth = 90
        cam.elevation = -20
        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        frames.append(frame)

        if done[0] > 0.5:
            break

    # Save video
    output_path = "newton_ant_trained.mp4"
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Video saved to {output_path}")
    print(f"Total reward: {total_reward:.1f}, Steps: {steps}")

    # Also save a GIF
    from PIL import Image
    compressed = []
    for f in frames[::2]:
        img = Image.fromarray(f)
        img = img.resize((img.width // 2, img.height // 2))
        compressed.append(np.array(img))
    imageio.mimsave("newton_ant_trained.gif", compressed, fps=15, loop=0)
    print("GIF saved to newton_ant_trained.gif")


if __name__ == "__main__":
    main()
