# Gymnasium Compatibility Notes

## Current Status

RL Games supports both Gym and Gymnasium environments through the `gym_compat` module. This module provides seamless compatibility for most use cases.

## Supported Environments

### Fully Supported
- **EnvPool environments**: Full support for Atari, Mujoco, and other EnvPool environments with Gymnasium
- **Basic Gym/Gymnasium environments**: CartPole, LunarLander, etc.
- **Brax environments**: Via Gymnasium interface
- **Most custom environments**: That follow standard Gym/Gymnasium APIs

### Limited Support
- **Non-EnvPool Atari environments with Gymnasium**: Due to API differences between Gym and Gymnasium wrappers, some edge cases may not work correctly. For production use, we recommend using EnvPool for Atari environments with Gymnasium.

## Technical Details

The `gym_compat` module:
1. Automatically detects Python version and imports appropriate backend (Gym for Python 3.8, Gymnasium for 3.9+)
2. Provides a unified `make` function that handles API differences
3. Wraps Gymnasium environments to provide backward compatibility with old Gym API (4-value step, 1-value reset)

## Recommendations

- For Atari environments, use EnvPool (`ppo_pong_envpool.yaml`) for best performance and compatibility
- For new projects, consider using Gymnasium-native APIs
- Report any compatibility issues to help improve support
