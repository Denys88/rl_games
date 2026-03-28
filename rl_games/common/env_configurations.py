import rl_games.envs.test
from rl_games.common import wrappers
from rl_games.common import tr_helpers
from rl_games.envs.brax import create_brax_env
from rl_games.envs.maniskill import create_maniskill_env
from rl_games.common.gymnasium_vecenv import create_gymnasium_env, wrap_atari
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FilterObservation
import numpy as np
import math


class HCRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.max([-10, reward])


class DMControlWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = self.env.observation_space['observations']
        self.observation_space.dtype = np.dtype('float32')

    def reset(self, **kwargs):
        self.num_stops = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observation, reward, done, info = result
        return observation, reward, done, info


class DMControlObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, obs):
        return obs['observations']


def create_default_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)
    env = gym.make(name, **kwargs)

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env


def create_goal_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)

    env = gym.make(name, **kwargs)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env



def create_myo(**kwargs):
    from myosuite.utils import gym
    name = kwargs.pop('name')
    env = gym.make(name, **kwargs)
    env = wrappers.OldGymWrapper(env)
    return env


def create_atari_gym_env(**kwargs):
    #frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    skip = kwargs.pop('skip',4)
    episode_life = kwargs.pop('episode_life',True)
    wrap_impala = kwargs.pop('wrap_impala', False)
    env = wrappers.make_atari_deepmind(name, skip=skip,episode_life=episode_life, wrap_impala=wrap_impala, **kwargs)
    return env


def create_dm_control_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = 'dm2gym:'+ kwargs.pop('name')
    env = gym.make(name, environment_kwargs=kwargs)
    env = DMControlWrapper(env)
    env = DMControlObsWrapper(env)
    env = wrappers.TimeLimit(env, 1000)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env




def create_smac(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv, MultiDiscreteSmacWrapper
    frames = kwargs.pop('frames', 1)
    transpose = kwargs.pop('transpose', False)
    flatten = kwargs.pop('flatten', True)
    has_cv = kwargs.get('central_value', False)
    as_single_agent = kwargs.pop('as_single_agent', False)
    env = SMACEnv(name, **kwargs)

    if frames > 1:
        if has_cv:
            env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=False, flatten=flatten)
        else:
            env = wrappers.BatchedFrameStack(env, frames, transpose=False, flatten=flatten)

    if as_single_agent:
        env = MultiDiscreteSmacWrapper(env)
    return env


def create_smac_v2(name, **kwargs):
    from rl_games.envs.smac_v2_env import SMACEnvV2
    frames = kwargs.pop('frames', 1)
    transpose = kwargs.pop('transpose', False)
    flatten = kwargs.pop('flatten', True)
    has_cv = kwargs.get('central_value', False)
    env = SMACEnvV2(name, **kwargs)

    if frames > 1:
        if has_cv:
            env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=False, flatten=flatten)
        else:
            env = wrappers.BatchedFrameStack(env, frames, transpose=False, flatten=flatten)
    return env


def create_smac_cnn(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv, MultiDiscreteSmacWrapper
    has_cv = kwargs.get('central_value', False)
    frames = kwargs.pop('frames', 4)
    transpose = kwargs.pop('transpose', False)
    as_single_agent = kwargs.pop('as_single_agent', False)

    env = SMACEnv(name, **kwargs)
    if has_cv:
        env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=transpose)
    else:
        env = wrappers.BatchedFrameStack(env, frames, transpose=transpose)
    if as_single_agent:
        env = MultiDiscreteSmacWrapper(env)
    return env


def create_test_env(name, **kwargs):
    import rl_games.envs.test
    env = gym.make(name, **kwargs)
    return env

def create_minigrid_env(name, **kwargs):
    import minigrid
    from minigrid.wrappers import (
        PositionBonus, ActionBonus,
        RGBImgObsWrapper, RGBImgPartialObsWrapper,
        ViewSizeWrapper, ImgObsWrapper,
    )

    state_bonus = kwargs.pop('state_bonus', False)
    action_bonus = kwargs.pop('action_bonus', False)
    rgb_fully_obs = kwargs.pop('rgb_fully_obs', False)
    rgb_partial_obs = kwargs.pop('rgb_partial_obs', True)
    view_size = kwargs.pop('view_size', 3)
    env = gym.make(name, **kwargs)

    if state_bonus:
        env = PositionBonus(env)
    if action_bonus:
        env = ActionBonus(env)

    if rgb_fully_obs:
        env = RGBImgObsWrapper(env)
    elif rgb_partial_obs:
        env = ViewSizeWrapper(env, view_size)
        env = RGBImgPartialObsWrapper(env, tile_size=84//view_size)

    env = ImgObsWrapper(env)
    print('minigrid_env observation space shape:', env.observation_space)
    return env


def create_multiwalker_env(**kwargs):
    from rl_games.envs.multiwalker import MultiWalker
    env = MultiWalker('', **kwargs)
    return env


def create_env(name, **kwargs):
    steps_limit = kwargs.pop('steps_limit', None)
    env = gym.make(name, **kwargs)
    if steps_limit is not None:
        env = wrappers.TimeLimit(env, steps_limit)
    return env


# Dictionary of env_name as key and a sub-dict containing env_type and a env-creator function
configurations = {
    'CartPole-v1' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'CartPoleMaskedVelocity-v1' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs : wrappers.MaskVelocityWrapper(gym.make('CartPole-v1'), 'CartPole-v1'),
    },
    'MountainCarContinuous-v0' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'MountainCar-v0' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Acrobot-v1' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Pendulum-v1' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'LunarLander-v3' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'PongNoFrameskip-v4' : {
        'vecenv_type' : 'GYMNASIUM',
        'default_env_config': {
            'env_name': 'ALE/Pong-v5',
            'frameskip': 1,  # Disable env's frameskip, let wrapper handle it
            'wrap_env': lambda env: wrap_atari(env, frame_skip=4, noop_max=30),
        },
    },
    'BreakoutNoFrameskip-v4' : {
        'vecenv_type' : 'GYMNASIUM',
        'default_env_config': {
            'env_name': 'ALE/Breakout-v5',
            'frameskip': 1,
            'wrap_env': lambda env: wrap_atari(env, frame_skip=4, noop_max=30),
        },
    },
    'MsPacmanNoFrameskip-v4' : {
        'vecenv_type' : 'GYMNASIUM',
        'default_env_config': {
            'env_name': 'ALE/MsPacman-v5',
            'frameskip': 1,
            'wrap_env': lambda env: wrap_atari(env, frame_skip=4, noop_max=30),
        },
    },
    'CarRacing-v2' : {
        'env_creator' : lambda **kwargs :  wrappers.make_car_racing('CarRacing-v2', skip=4),
        'vecenv_type' : 'RAY'
    },
    'LunarLanderContinuous-v3' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Ant-v4' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'HalfCheetah-v4' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Hopper-v4' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Walker2d-v4' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'Humanoid-v4' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'BipedalWalker-v3' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'BipedalWalkerCnn-v3' : {
        'env_creator' : lambda **kwargs : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalker-v3')), 4, False),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcore-v3' : {
        'vecenv_type' : 'GYMNASIUM',
    },
    'BipedalWalkerHardcoreCnn-v3' : {
        'env_creator' : lambda **kwargs : wrappers.FrameStack(gym.make('BipedalWalkerHardcore-v3'), 4, False),
        'vecenv_type' : 'RAY'
    },
    'smac' : {
        'env_creator' : lambda **kwargs : create_smac(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'smac_v2' : {
        'env_creator' : lambda **kwargs : create_smac_v2(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'smac_cnn' : {
        'env_creator' : lambda **kwargs : create_smac_cnn(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'dm_control' : {
        'env_creator' : lambda **kwargs : create_dm_control_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_gym' : {
        'env_creator' : lambda **kwargs : create_default_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_robot_gym' : {
        'env_creator' : lambda **kwargs : create_goal_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'atari_gym' : {
        'env_creator' : lambda **kwargs : create_atari_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'test_env' : {
        'env_creator' : lambda **kwargs : create_test_env(kwargs.pop('name'), **kwargs),
        'vecenv_type' : 'RAY'
    },
    'minigrid_env' : {
        'env_creator' : lambda **kwargs : create_minigrid_env(kwargs.pop('name'), **kwargs),
        'vecenv_type' : 'RAY'
    },
    'multiwalker_env' : {
        'env_creator' : lambda **kwargs : create_multiwalker_env(**kwargs),
        'vecenv_type' : 'GYMNASIUM'
    },
    'brax' : {
        'env_creator': lambda **kwargs: create_brax_env(**kwargs),
        'vecenv_type': 'BRAX'
    },
    'maniskill' : {
        'env_creator': lambda **kwargs: create_maniskill_env(**kwargs),
        'vecenv_type': 'MANISKILL'
    },
    'gymnasium' : {
        'env_creator': lambda **kwargs: create_gymnasium_env(**kwargs),
        'vecenv_type': 'GYMNASIUM'
    },
    'atari_gymnasium' : {
        'vecenv_type': 'GYMNASIUM',
        'default_env_config': {
            'frameskip': 1,  # Disable env's frameskip, let wrapper handle it
            'wrap_env': lambda env: wrap_atari(env, frame_skip=4, noop_max=30),
        },
    },
    'myo_gym' : {
        'env_creator' : lambda **kwargs : create_myo(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'pufferlib' : {
        'vecenv_type': 'PUFFERLIB'
    },
}

def get_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    result_shapes['value_size'] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    '''
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']

    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    '''
    if hasattr(env, "value_size"):
        result_shapes['value_size'] = env.value_size
    print(result_shapes)
    return result_shapes


def get_obs_and_action_spaces_from_config(config):
    env_config = config.get('env_config', {})
    env = configurations[config['env_name']]['env_creator'](**env_config)
    result_shapes = get_env_info(env)
    env.close()
    return result_shapes


def register(name, config):
    """Add a new key-value pair to the known environments (configurations dict).

    Args:
        name (:obj:`str`): Name of the env to be added.
        config (:obj:`dict`): Dictionary with env type and a creator function.

    """
    configurations[name] = config
