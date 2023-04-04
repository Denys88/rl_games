from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np
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
        import jax
        from brax import envs
    #    import jax.numpy as jnp

        self.num_envs = num_actors
        self.env_name = kwargs.pop('env_name', 'ant')
        self.sim_backend = kwargs.pop('backend', 'positional') # can be 'generalized', 'positional', 'spring'
        self.seed = kwargs.pop('seed', 7)

        print('env_name', self.env_name)
        print('sim_backend', self.sim_backend)
        print('seed', self.seed)
        print('num_envs', self.num_envs)

        # self.env = envs.create_gym_env(env_name=self.env_name,
        #            batch_size= self.batch_size,
        #            seed = 0,
        #            backend = 'gpu'
        #            )

        #self.env = envs.get_environment(env_name=self.env_name, backend=self.sim_backend)

        self.env = envs.create(env_name=self.env_name, batch_size=self.num_envs, backend=self.sim_backend)

        self.state = None # will be initilized in reset()
        #self.env.reset(rng=jax.random.PRNGKey(seed=self.seed))

        self.jit_reset = jax.jit(self.env.reset)
        self.jit_step = jax.jit(self.env.step)

        obs_high = np.inf * np.ones(self.env.observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(self.env.action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def step(self, action):
        # print('action device', action.device)
        # print('action shape', action.shape)

        action = torch_to_jax(action)
        self.state = self.jit_step(self.state, action)
        next_obs = jax_to_torch(self.state.obs)
        reward = jax_to_torch(self.state.reward)
        is_done = jax_to_torch(self.state.done)

        # print('jax obs device', self.state.obs.device_buffer.device())

        # from jax import numpy as jnp
        # print('jax obs shape', jnp.shape(self.state.obs))

        return next_obs, reward, is_done, self.state.info

    def reset(self):
        import jax
        self.state = self.env.jit_reset(rng=jax.random.PRNGKey(seed=self.seed))

        #print('reset jax obs device', self.state.obs.device_buffer.device())

        return jax_to_torch(self.state.obs)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_brax_env(**kwargs):
    return BraxEnv("", kwargs.pop('num_actors', 256), **kwargs)