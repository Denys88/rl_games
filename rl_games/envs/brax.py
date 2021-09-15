

from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np
import torch
import torch.utils.dlpack as tpack

def jax_to_torch(tensor):
    from jax._src.dlpack import (to_dlpack,)
    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor

def torch_to_jax(tensor):
    from jax._src.dlpack import (from_dlpack,)
    tensor = tpack.to_dlpack(tensor)
    tensor = from_dlpack(tensor)
    return tensor


class BraxEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import brax
        from brax import envs    
        import jax
        import jax.numpy as jnp

        self.batch_size = num_actors
        env_fn = envs.create_fn(env_name=kwargs.pop('env_name', 'ant'))
        self.env = env_fn(
            action_repeat=1,
            batch_size=num_actors,
            episode_length=kwargs.pop('episode_length', 1000))
        obs_high = np.inf * np.ones(self.env.observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(self.env.action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        def step(first_state, state, action):
            
            def test_done(a, b):
                if a is first_state.done or a is first_state.metrics or a is first_state.reward:
                    return b
                test_shape = [a.shape[0],] + [1 for _ in range(len(a.shape) - 1)]
                return jnp.where(jnp.reshape(state.done, test_shape), a, b)
            state = self.env.step(state, action)
            state = jax.tree_multimap(test_done, first_state, state)
            return state, state.obs, state.reward, state.done, {}

        def reset(key):
            state = self.env.reset(key)
            return state, state.obs

        self._reset = jax.jit(reset, backend='gpu')
        self._step = jax.jit(step, backend='gpu')

    def step(self, action):
        action = torch_to_jax(action)
        self.state, next_obs, reward, is_done, info = self._step(self.first_state, self.state, action)
        #next_obs = np.asarray(next_obs).astype(np.float32)
        #reward = np.asarray(reward).astype(np.float32)
        #is_done = np.asarray(is_done).astype(np.long)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        is_done = jax_to_torch(is_done)
        return next_obs, reward, is_done, info

    def reset(self):
        import jax
        import jax.numpy as jnp
        rng = jax.random.PRNGKey(seed=0)
        rng = jax.random.split(rng, self.batch_size)
        self.first_state, _ = self._reset(rng)
        self.state, obs = self._reset(rng)
        #obs = np.asarray(obs).astype(np.float32)

        return jax_to_torch(obs)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info

def create_brax_env(**kwargs):
    return BraxEnv("", kwargs.pop('num_actors', 256), **kwargs)

