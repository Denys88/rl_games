"""Minimal compatibility layer for Gym/Gymnasium."""
import sys

# Use Gymnasium for Python >= 3.9, Gym for older versions
if sys.version_info >= (3, 9):
    import gymnasium as gym
    from gymnasium import spaces
    GYM_BACKEND = "gymnasium"
else:
    import gym
    from gym import spaces
    GYM_BACKEND = "gym"

# Re-export for convenience
__all__ = ["gym", "spaces", "make", "GYM_BACKEND", "wrap_gymnasium_env"]


def make(env_id, **kwargs):
    """Create env with normalized API."""
    env = gym.make(env_id, **kwargs)

    if GYM_BACKEND == "gymnasium":
        # Wrap to provide old-style API for consistency
        class CompatEnv(type(env)):
            def reset(self, **kwargs):
                obs, info = super().reset(**kwargs)
                return obs  # Drop info for compatibility

            def step(self, action):
                obs, reward, terminated, truncated, info = super().step(action)
                done = terminated or truncated
                info['time_outs'] = truncated
                return obs, reward, done, info

        env.__class__ = CompatEnv

    return env


def wrap_gymnasium_env(env):
    """Wrap a Gymnasium env for use with old Gym API."""
    if GYM_BACKEND == "gym":
        from rl_games.common.wrappers import OldGymWrapper
        return OldGymWrapper(env)
    else:
        # Already using Gymnasium, just normalize the API
        return make(lambda: env)
