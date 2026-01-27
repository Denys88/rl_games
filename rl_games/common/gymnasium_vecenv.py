from rl_games.common.ivecenv import IVecEnv
import gymnasium
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np

# Register ALE (Atari) environments if available
try:
    import ale_py
    gymnasium.register_envs(ale_py)
except ImportError:
    pass  # ale_py not installed, Atari envs won't be available


def wrap_atari(env, frame_stack=4, noop_max=30, frame_skip=4,
               terminal_on_life_loss=True, grayscale=True, scale=False):
    """Apply standard Atari preprocessing wrappers.

    Args:
        env: Base Atari environment
        frame_stack: Number of frames to stack (0 to disable)
        noop_max: Max no-ops at reset
        frame_skip: Frames to skip between actions
        terminal_on_life_loss: End episode on life loss
        grayscale: Convert to grayscale
        scale: Scale observations to [0,1)
    """
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=84,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,  # Don't add channel dim - FrameStack will stack along new axis
        scale_obs=scale
    )
    if frame_stack > 0:
        env = FrameStackObservation(env, frame_stack)
    return env


class GymnasiumVecEnv(IVecEnv):
    """Vectorized environment wrapper using gymnasium.vector.

    This replaces Ray-based parallelization for standard gymnasium environments,
    providing better performance and simpler setup.

    Args:
        config_name: Environment configuration name (used as env_name fallback)
        num_actors: Number of parallel environments
        env_name: Gymnasium environment ID (defaults to config_name)
        use_async: Use AsyncVectorEnv instead of SyncVectorEnv
        seed: Random seed for environments
        wrap_env: Optional function to wrap each env before vectorization
        **kwargs: Additional kwargs passed to gymnasium.make()
    """

    def __init__(self, config_name, num_actors, **kwargs):
        self.batch_size = num_actors
        # Use config_name as env_name if not explicitly provided
        env_name = kwargs.pop('env_name', config_name)
        use_async = kwargs.pop('use_async', False)
        self.seed_value = kwargs.pop('seed', None)
        wrap_env = kwargs.pop('wrap_env', None)

        # Store remaining kwargs for env creation
        self.env_kwargs = kwargs

        def make_env(idx):
            def _init():
                env = gymnasium.make(env_name, **self.env_kwargs)
                if wrap_env is not None:
                    env = wrap_env(env)
                if self.seed_value is not None:
                    env.reset(seed=self.seed_value + idx)
                return env
            return _init

        VecEnvClass = AsyncVectorEnv if use_async else SyncVectorEnv
        self.env = VecEnvClass([make_env(i) for i in range(num_actors)])

        # Get spaces from vectorized env (already batched)
        # We need single-env spaces for rl_games compatibility
        single_obs_space = self.env.single_observation_space
        single_action_space = self.env.single_action_space

        self.observation_space = single_obs_space
        self.action_space = single_action_space

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated | truncated
        info['time_outs'] = truncated
        return next_obs, reward, is_done, info

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def seed(self, seed):
        self.seed_value = seed

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_gymnasium_env(**kwargs):
    return GymnasiumVecEnv("", kwargs.pop('num_actors', 16), **kwargs)
