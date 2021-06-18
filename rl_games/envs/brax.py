
import numpy as np

class BraxEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import brax
        from brax import envs
        env_fn = envs.create_fn(env_name=kwargs.pop('env_name', 'ant'))
        self.env = env_fn(
            action_repeat=1,
            batch_size=kwargs.pop('num_actors', num_actors),
            episode_length=kwargs.pop('episode_length', 1000))
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        next_obs = np.asarray(next_obs).to(np.float32)
        reward = np.asarray(reward).to(np.float32)
        is_done = np.asarray(is_done).to(np.float32)
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs).to(np.float32)
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        return info

def create_brax_env(**kwargs):
    import brax
    from brax import envs
    env_fn = envs.create_fn(env_name=kwargs.pop('env_name', 'ant'))
    env = env_fn(
        action_repeat=1,
        batch_size=kwargs.pop('num_actors', kwargs.pop('num_actors', 256)),
        episode_length=kwargs.pop('episode_length', 1000))
    return env

