import rl_games.common.wrappers as wrappers
from rl_games.common.ivecenv import IVecEnv

# wrap your vector env so it resets for you under the hood
from gymnasium import spaces


def remove_batch_dim(space: spaces.Space) -> spaces.Space:
    """Recursively remove the first (batch) dimension from a Gym space."""
    if isinstance(space, spaces.Box):
        # assume shape = (B, *shape); drop the 0th index
        low  = space.low[0]
        high = space.high[0]
        return spaces.Box(low=low, high=high, dtype=space.dtype)
    elif isinstance(space, spaces.MultiDiscrete):
        # assume nvec = (B, n); take first row
        nvec = space.nvec[0]
        return spaces.MultiDiscrete(nvec)
    elif isinstance(space, spaces.MultiBinary):
        # n can be int or array-like
        n = space.n[0] if hasattr(space.n, "__len__") else space.n
        return spaces.MultiBinary(n)
    elif isinstance(space, spaces.Discrete):
        # Discrete spaces have no extra dims
        return space
    elif isinstance(space, spaces.Tuple):
        return spaces.Tuple(tuple(remove_batch_dim(s) for s in space.spaces))
    elif isinstance(space, spaces.Dict):
        return spaces.Dict({k: remove_batch_dim(s) for k, s in space.spaces.items()})
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


class ManiskillEnv(IVecEnv):
    def __init__(self, config_name, num_actors,  **kwargs):
        import gymnasium
        import mani_skill.envs
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

        self.batch_size = num_actors
        env_name=kwargs.pop('env_name')
        self.seed = kwargs.pop('seed', 0) # not sure how to set this in mani_skill
        self.env = gymnasium.make(
            env_name,
            num_envs=num_actors,
            **kwargs
        )
        #self.env = FlattenRGBDObservationWrapper(self.env, rgb=True, depth=False, state=False, sep_depth=False)
        # need to use this wrapper to have automatic reset for done envs
        self.env = ManiSkillVectorEnv(self.env)
        
        print(f"ManiSkill env: {env_name} with {num_actors} actors")
        print(f"Original observation space: {self.env.observation_space}")
        self.action_space = wrappers.OldGymWrapper.convert_space(remove_batch_dim(self.env.action_space))
        self.observation_space = wrappers.OldGymWrapper.convert_space(remove_batch_dim(self.env.observation_space))
        print(f"Converted action space: {self.action_space}")
        print(f"Converted observation space: {self.observation_space}")

    def step(self, action):
        next_obs, reward, done, truncated, info = self.env.step(action)
        is_done = done | truncated
        info['time_outs'] = truncated
        return next_obs, reward, is_done, info

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_maniskill_env(**kwargs):
    return ManiskillEnv(kwargs.pop('full_experiment_name', ''), kwargs.pop('num_actors', 16), **kwargs)