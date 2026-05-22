"""Generate a video of the trained Walker2d agent."""
import gymnasium as gym
import torch
import numpy as np
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
import yaml
import imageio

def main():
    checkpoint = "runs/Walker2d-v4_ray_22-13-16-27/nn/Walker2d-v4_ray.pth"
    config_path = "rl_games/configs/mujoco/walker2d.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use the runner to set up the player
    runner = Runner()
    runner.load(config)
    runner.reset()

    agent = runner.create_player()
    agent.restore(checkpoint)

    # Create env with rgb rendering
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    frames = []
    total_reward = 0
    steps = 0

    for _ in range(1000):
        frame = env.render()
        frames.append(frame)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, is_deterministic=True)
        action = action.squeeze().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    env.close()

    # Save full MP4
    imageio.mimsave("walker2d_trained.mp4", frames, fps=30)

    # Save compressed GIF (every 3rd frame, scaled down)
    from PIL import Image
    compressed = []
    for f in frames[::3]:
        img = Image.fromarray(f)
        img = img.resize((img.width // 2, img.height // 2))
        compressed.append(np.array(img))
    output_path = "walker2d_trained.gif"
    imageio.mimsave(output_path, compressed, fps=10, loop=0)
    print(f"Video saved to {output_path}")
    print(f"Total reward: {total_reward:.1f}, Steps: {steps}")

if __name__ == "__main__":
    main()
