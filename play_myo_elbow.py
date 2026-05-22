"""Generate video of trained MyoSuite elbow agent."""
import numpy as np
import torch
import imageio
import yaml

def main():
    from myosuite.utils import gym as myo_gym
    from rl_games.torch_runner import Runner

    checkpoint_path = 'runs/MyoElbowPose1D6MRandom_14-17-07-45/nn/MyoElbowPose1D6MRandom.pth'
    config_path = 'rl_games/configs/myosuite/ppo_myo_elbow.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    runner = Runner()
    runner.load(config)
    runner.reset()

    args = {'train': False, 'play': True, 'checkpoint': checkpoint_path, 'sigma': None}
    agent = runner.create_player()
    agent.restore(checkpoint_path)

    # Create a separate env for rendering
    env = myo_gym.make('myoElbowPose1D6MRandom-v0')
    renderer = env.unwrapped.sim.renderer

    num_episodes = 3
    all_frames = []
    total_rewards = []

    for ep in range(num_episodes):
        obs_tuple = env.reset()
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        ep_reward = 0

        for step in range(100):
            frame = renderer.render_offscreen(width=640, height=480)
            all_frames.append(frame.copy())

            # Use the player's get_action
            obs_tensor = agent.obs_to_torch(obs)
            action = agent.get_action(obs_tensor, is_deterministic=True)
            action_np = action.squeeze().cpu().numpy() if isinstance(action, torch.Tensor) else np.squeeze(action)

            result = env.step(action_np)
            obs, reward = result[0], result[1]
            done = result[2] if len(result) > 2 else False
            trunc = result[3] if len(result) > 3 else False
            ep_reward += reward

            if done or trunc:
                break

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.1f}, steps={step+1}")

    output_path = 'myo_elbow_trained.mp4'
    imageio.mimsave(output_path, all_frames, fps=30)
    print(f"\nSaved video to {output_path}")
    print(f"Mean reward: {np.mean(total_rewards):.1f} over {num_episodes} episodes")
    env.close()

if __name__ == '__main__':
    main()
