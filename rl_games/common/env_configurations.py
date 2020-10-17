from rl_games.common import wrappers
from rl_games.common import tr_helpers

import gym
from gym.wrappers import FlattenObservation, FilterObservation
import numpy as np

#FLEX_PATH = '/home/viktor/Documents/rl/FlexRobotics'
FLEX_PATH = '/home/trrrrr/Documents/FlexRobotics-master'



class HCRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        self.num_stops = 0
        self.stops_decay = 0
        self.max_stops = 30

    def reset(self, **kwargs):
        self.num_stops = 0
        self.stops_decay = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.num_stops > self.max_stops:
            print('too many stops!')
            reward = -100
            observation = self.reset()
            done = True
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return np.max([-10, reward])


class DMControlReward(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        self.num_stops = 0
        self.max_stops = 1000
        self.reward_threshold = 0.001

    def reset(self, **kwargs):
        self.num_stops = 0
 
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if reward < self.reward_threshold:
            self.num_stops += 1
        else:
            self.num_stops = max(0, self.num_stops-1)
        if self.num_stops > self.max_stops:
            #print('too many stops!')
            reward = -10
            observation = self.reset()
            done = True
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return reward


class DMControlObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def observation(self, obs):
        return obs['observations']


def create_default_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    is_procgen = kwargs.pop('procgen', False)
    limit_steps = kwargs.pop('limit_steps', False)
    env = gym.make(name, **kwargs)

    if frames > 1:
        if is_procgen:
            env = wrappers.ProcgenStack(env, frames, True)
        else:
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

def create_slime_gym_env(**kwargs):
    import slimevolleygym
    from rl_games.envs.slimevolley_selfplay import SlimeVolleySelfplay
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)
    self_play = kwargs.pop('self_play', False)
    if self_play:
        env = SlimeVolleySelfplay(name, **kwargs) 
    else:
        env = gym.make(name, **kwargs)
    return env

def create_connect_four_env(**kwargs):
    from rl_games.envs.connect4_selfplay import ConnectFourSelfPlay
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)
    self_play = kwargs.pop('self_play', False)
    if self_play:
        env = ConnectFourSelfPlay(name, **kwargs) 
    else:
        env = gym.make(name, **kwargs)
    return env

def create_atari_gym_env(**kwargs):
    #frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    env = wrappers.make_atari_deepmind(name, skip=4)
    return env    

def create_dm_control_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = 'dm2gym:'+ kwargs.pop('name')
    env = gym.make(name, environment_kwargs=kwargs)
    env = DMControlReward(env)
    env = DMControlObsWrapper(env)

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env

def create_super_mario_env(name='SuperMarioBros-v1'):
    import gym
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    import gym_super_mario_bros
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    return env

def create_super_mario_env_stage1(name='SuperMarioBrosRandomStage1-v1'):
    import gym
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    import gym_super_mario_bros
    stage_names = [
        'SuperMarioBros-1-1-v1',
        'SuperMarioBros-1-2-v1',
        'SuperMarioBros-1-3-v1',
        'SuperMarioBros-1-4-v1',
    ]

    env = gym_super_mario_bros.make(stage_names[1])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    #env = wrappers.AllowBacktracking(env)
    
    return env

def create_quadrupped_env():
    import gym
    import roboschool
    import quadruppedEnv
    return wrappers.FrameStack(wrappers.MaxAndSkipEnv(gym.make('QuadruppedWalk-v1'), 4, False), 2, True)

def create_roboschool_env(name):
    import gym
    import roboschool
    return gym.make(name)

def create_multiflex(path, num_instances=1):
    from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env_muli_env
    from autolab_core import YamlConfig
    import gym

    set_flex_bin_path(FLEX_PATH + '/bin')

    cfg_env = YamlConfig(path)
    env = make_flex_vec_env_muli_env([cfg_env] * num_instances)

    return env

def create_flex(path):
    from flex_gym.flex_vec_env import set_flex_bin_path, make_flex_vec_env
    from autolab_core import YamlConfig
    import gym

    set_flex_bin_path(FLEX_PATH + '/bin')

    cfg_env = YamlConfig(path)
    cfg_env['gym']['rank'] = 0
    env = make_flex_vec_env(cfg_env)

    return env

def create_smac(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    frames = kwargs.pop('frames', 1)
    transpose = kwargs.pop('transpose', False)
    flatten = kwargs.pop('flatten', True)
    has_cv = kwargs.get('central_value', False)
    env = SMACEnv(name, **kwargs)
    
    
    if frames > 1:
        if has_cv:
            env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=False, flatten=flatten)
        else:
            env = wrappers.BatchedFrameStack(env, frames, transpose=False, flatten=flatten)
    return env

def create_smac_cnn(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    env = SMACEnv(name, **kwargs)
    frames = kwargs.pop('frames', 4)
    transpose = kwargs.pop('transpose', False)
    if has_cv:
        env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=False)
    else:
        env = wrappers.BatchedFrameStack(env, frames, transpose=False)
        
    return env


def create_test_env(name, **kwargs):
    import rl_games.envs.test
    env = gym.make(name, **kwargs)
    return env


def create_minigrid_env(name, **kwargs):
    import gym_minigrid
    import gym_minigrid.wrappers
    state_bonus = kwargs.pop('state_bonus', False)
    action_bonus = kwargs.pop('action_bonus', False)
    fully_obs = kwargs.pop('fully_obs', False)

    env = gym.make(name, **kwargs)
    if state_bonus:
        env = gym_minigrid.wrappers.StateBonus(env)
    if action_bonus:
        env = gym_minigrid.wrappers.ActionBonus(env)
    if fully_obs:
        env = gym_minigrid.wrappers.RGBImgObsWrapper(env)
    else:
        env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env) # Get pixel observations
    env = gym_minigrid.wrappers.ImgObsWrapper(env) # Get rid of the 'mission' field

    print('minigird_env observation space shape:', env.observation_space)
    return env

configurations = {
    'CartPole-v1' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs : wrappers.MaskVelocityWrapper(gym.make('CartPole-v1'), 'CartPole-v1'),
    },
    'MountainCarContinuous-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs  : gym.make('MountainCarContinuous-v0'),
    },
    'MountainCar-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda : gym.make('MountainCar-v0'),
    },
    'Acrobot-v1' : {
        'env_creator' : lambda **kwargs  : gym.make('Acrobot-v1'),
        'vecenv_type' : 'RAY'
    },
    'Pendulum-v0' : {
        'env_creator' : lambda **kwargs  : gym.make('Pendulum-v0'),
        'vecenv_type' : 'RAY'
    },
    'LunarLander-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLander-v2'),
        'vecenv_type' : 'RAY'
    },
    'PongNoFrameskip-v4' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_atari_deepmind('PongNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'BreakoutNoFrameskip-v4' : {
        'env_creator' : lambda  **kwargs :  wrappers.make_atari_deepmind('BreakoutNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'CarRacing-v0' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_car_racing('CarRacing-v0', skip=4),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolAnt-v1' : {
        'env_creator' : lambda **kwargs  : create_roboschool_env('RoboschoolAnt-v1'),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBros-v1' : {
        'env_creator' : lambda :  create_super_mario_env(),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBrosRandomStages-v1' : {
        'env_creator' : lambda :  create_super_mario_env('SuperMarioBrosRandomStages-v1'),
        'vecenv_type' : 'RAY'
    },
    'SuperMarioBrosRandomStage1-v1' : {
        'env_creator' : lambda **kwargs  :  create_super_mario_env_stage1('SuperMarioBrosRandomStage1-v1'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHalfCheetah-v1' : {
        'env_creator' : lambda **kwargs  : create_roboschool_env('RoboschoolHalfCheetah-v1'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHumanoid-v1' : {
        'env_creator' : lambda : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoid-v1'), 1, True),
        'vecenv_type' : 'RAY'
    },
    'LunarLanderContinuous-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLanderContinuous-v2'),
        'vecenv_type' : 'RAY'
    },
    'RoboschoolHumanoidFlagrun-v1' : {
        'env_creator' : lambda **kwargs  : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoidFlagrun-v1'), 1, True),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalker-v3' : {
        'env_creator' : lambda **kwargs  : gym.make('BipedalWalker-v3'),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerCnn-v3' : {
        'env_creator' : lambda **kwargs  : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalker-v3')), 4, False),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcore-v3' : {
        'env_creator' : lambda **kwargs  : HCRewardEnv(gym.make('BipedalWalkerHardcore-v3')),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcoreCnn-v3' : {
        'env_creator' : lambda : wrappers.FrameStack(gym.make('BipedalWalkerHardcore-v3'), 4, False),
        'vecenv_type' : 'RAY'
    },
    'QuadruppedWalk-v1' : {
        'env_creator' : lambda **kwargs  : create_quadrupped_env(),
        'vecenv_type' : 'RAY'
    },
    'FlexAnt' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/ant.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'FlexHumanoid' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/humanoid.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'FlexHumanoidHard' : {
        'env_creator' : lambda **kwargs  : create_flex(FLEX_PATH + '/demo/gym/cfg/humanoid_hard.yaml'),
        'vecenv_type' : 'ISAAC'
    },
    'smac' : {
        'env_creator' : lambda **kwargs : create_smac(**kwargs),
        'vecenv_type' : 'RAY_SMAC'
    },
    'smac_cnn' : {
        'env_creator' : lambda **kwargs : create_smac_cnn(**kwargs),
        'vecenv_type' : 'RAY_SMAC'
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
    'slime_gym' : {
        'env_creator' : lambda **kwargs : create_slime_gym_env(**kwargs),
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
    'connect4_env' : {
        'env_creator' : lambda **kwargs : create_connect_four_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
}


def get_obs_and_action_spaces(name):
    env = configurations[name]['env_creator']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    # workaround for deepmind control
    if isinstance(observation_space, gym.spaces.dict.Dict):
        observation_space = observation_space['observations']
    return observation_space, action_space

def get_obs_and_action_spaces_from_config(config):
    result_shapes = {}
    env_config = config.get('env_config', {})
    env = configurations[config['env_name']]['env_creator'](**env_config)
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    env.close()
    # workaround for deepmind control

    if isinstance(observation_space, gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    if isinstance(observation_space, dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    return result_shapes

def get_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    return result_shapes

def register(name, config):
    configurations[name] = config