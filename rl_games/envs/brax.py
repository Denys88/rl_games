
import numpy as np
from rl_games.common.ivecenv import IVecEnv
import gym

class BraxEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import brax
        from brax import envs
        self.batch_size = num_actors
        print('num_actors ', num_actors)
        env_fn = envs.create_fn(env_name=kwargs.pop('env_name', 'ant'))
        self.env = env_fn(
            action_repeat=1,
            batch_size=num_actors,
            episode_length=kwargs.pop('episode_length', 1000))
        obs_high = np.inf * np.ones(self.env.observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(self.env.action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        def step(state, action):
            state = self.env.step(state, action)
            return state, state.obs, state.reward, state.done, {}

        def reset(key):
            state = self.env.reset(key)
            return state, state.obs

        import jax
        self._reset = jax.jit(reset, backend='gpu')
        self._step = jax.jit(step, backend='gpu')

    def step(self, action):
        self.state, next_obs, reward, is_done, info = self._step(self.state, action)
        next_obs = np.asarray(next_obs).astype(np.float32)
        reward = np.asarray(reward).astype(np.float32)
        is_done = np.asarray(is_done).astype(np.float32)
        return next_obs, reward, is_done, info

    def reset(self):
        import jax
        import jax.numpy as jnp
        rng = jax.random.PRNGKey(seed=0)
        rng = jax.random.split(rng, self.batch_size)
        self.state, obs = self._reset(rng)
        obs = np.asarray(obs).astype(np.float32)
        print(np.shape(obs))
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info

def create_brax_env(**kwargs):
    return BraxEnv("", kwargs.pop('num_actors', 256), **kwargs)

