"""PufferLib vectorized environment integration for rl_games.

PufferLib is an optional dependency. Install it with:

    pip install rl_games[pufferlib]

Or standalone:

    pip install pufferlib

Note: PufferLib 3.x pins numpy<2.0 and gymnasium<=0.29.1.
To keep newer versions, install with --no-deps and manage dependencies manually:

    pip install pufferlib --no-deps
    pip install numpy>=2.0 gymnasium>=1.0

PufferLib's core vectorization API works fine with newer numpy/gymnasium
despite the conservative version pins.
"""

from rl_games.common.ivecenv import IVecEnv


class PufferLibVecEnv(IVecEnv):
    """Vectorized environment wrapper using PufferLib.

    PufferLib provides high-performance vectorized environment simulation
    with Serial, Multiprocessing, and Ray backends.
    """

    def __init__(self, config_name, num_actors, **kwargs):
        import pufferlib
        import pufferlib.vector

        self.batch_size = num_actors
        env_name = kwargs.pop('env_name', config_name)
        env_creator = kwargs.pop('env_creator', None)
        seed = kwargs.pop('seed', 0)

        backend_name = kwargs.pop('backend', 'Serial')
        backend_map = {
            'Serial': pufferlib.vector.Serial,
            'Multiprocessing': pufferlib.vector.Multiprocessing,
            'Ray': pufferlib.vector.Ray,
        }
        backend = backend_map.get(backend_name, pufferlib.vector.Serial)

        backend_kwargs = kwargs.pop('backend_kwargs', {})

        if env_creator is None:
            import gymnasium
            import pufferlib.emulation
            def _default_creator(**kw):
                return pufferlib.emulation.GymnasiumPufferEnv(
                    env_creator=gymnasium.make,
                    env_kwargs={'id': env_name},
                    **kw,
                )
            env_creator = _default_creator

        make_kwargs = dict(
            env_kwargs=kwargs if kwargs else None,
            backend=backend,
            num_envs=num_actors,
            **backend_kwargs,
        )

        self.env = pufferlib.vector.make(env_creator, **make_kwargs)

        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated | truncated
        if isinstance(info, list):
            info = {}
        info['time_outs'] = truncated
        return next_obs, reward, is_done, info

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = 1
        info['value_size'] = 1
        return info

    def close(self):
        if hasattr(self, 'env') and hasattr(self.env, 'close'):
            self.env.close()
