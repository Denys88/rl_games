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


def _apply_old_gym_api(env):
    """Patch a gymnasium env instance to provide old gym-style API (4-tuple step, single-return reset)."""
    class CompatEnv(type(env)):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self._last_reset_info = info
            return obs

        def get_reset_info(self):
            """Get the info dict from the last reset() call."""
            return getattr(self, '_last_reset_info', {})

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            done = terminated or truncated
            info['time_outs'] = truncated and not terminated
            return obs, reward, done, info

    env.__class__ = CompatEnv
    return env


def make(env_id, old_gym_api=True, **kwargs):
    """Create env with normalized API.

    Args:
        env_id: Environment ID to create
        old_gym_api: If True (default), wrap gymnasium envs to provide old Gym API.
                     If False, return gymnasium envs with new API.
        **kwargs: Additional arguments for gym.make()
    """
    env = gym.make(env_id, **kwargs)

    if GYM_BACKEND == "gymnasium" and old_gym_api:
        _apply_old_gym_api(env)

    return env


def wrap_gymnasium_env(env):
    """Wrap a Gymnasium env for use with old Gym API."""
    if GYM_BACKEND == "gym":
        from rl_games.common.wrappers import OldGymWrapper
        return OldGymWrapper(env)
    else:
        return _apply_old_gym_api(env)
