import networks 
import wrappers
import tr_helpers
import gym
import numpy as np

def create_super_mario_env(name='SuperMarioBros-v1'):
    import gym
    from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    env = gym_super_mario_bros.make(name)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    return env

def create_super_mario_env_stage1(name='SuperMarioBrosRandomStage1-v1'):
    import gym
    from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    stage_names = [
        'SuperMarioBros-1-1-v1',
        'SuperMarioBros-1-2-v1',
        'SuperMarioBros-1-3-v1',
        'SuperMarioBros-1-4-v1',
    ]

    env = gym_super_mario_bros.make(stage_names[1])
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    #env = wrappers.AllowBacktracking(env)
    
    return env


def create_quadrupped_env():
    import gym
    import roboschool
    import quadruppedEnv
    return wrappers.FrameStack(wrappers.MaxAndSkipEnv(gym.make('QuadruppedWalk-v1'),4, False), 2, True)

def create_roboschool_env(name):
    import gym
    import roboschool
    return gym.make(name)

class HCRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        if reward == -100:
            return -5
        if reward == 0:
            return -0.1
        return reward

configurations = {
    'CartPole-v1' : {
        'VECENV_TYPE' : 'RAY',
        'ENV_CREATOR' : lambda : gym.make('CartPole-v1'),
    },
    'MountainCarContinuous-v0' : {
        'VECENV_TYPE' : 'RAY',
        'ENV_CREATOR' : lambda : gym.make('MountainCarContinuous-v0'),
    },
    'MountainCar-v0' : {
        'VECENV_TYPE' : 'RAY',
        'ENV_CREATOR' : lambda : gym.make('MountainCar-v0'),
    },
    'Acrobot-v1' : {
        'ENV_CREATOR' : lambda : gym.make('Acrobot-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'Pendulum-v0' : {
        'ENV_CREATOR' : lambda : gym.make('Pendulum-v0'),
        'VECENV_TYPE' : 'RAY'
    },
    'LunarLander-v2' : {
        'ENV_CREATOR' : lambda : gym.make('LunarLander-v2'),
        'VECENV_TYPE' : 'RAY'
    },
    'PongNoFrameskip-v4' : {
        'ENV_CREATOR' : lambda :  wrappers.make_atari_deepmind('PongNoFrameskip-v4', skip=4),
        'VECENV_TYPE' : 'RAY'
    },
    'CarRacing-v0' : {
        'ENV_CREATOR' : lambda :  wrappers.make_car_racing('CarRacing-v0', skip=4),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolAnt-v1' : {
        'ENV_CREATOR' : lambda : create_roboschool_env('RoboschoolAnt-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'SuperMarioBros-v1' : {
        'ENV_CREATOR' : lambda :  create_super_mario_env(),
        'VECENV_TYPE' : 'RAY'
    },
    'SuperMarioBrosRandomStages-v1' : {
        'ENV_CREATOR' : lambda :  create_super_mario_env('SuperMarioBrosRandomStages-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'SuperMarioBrosRandomStage1-v1' : {
        'ENV_CREATOR' : lambda :  create_super_mario_env_stage1('SuperMarioBrosRandomStage1-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHalfCheetah-v1' : {
        'ENV_CREATOR' : lambda : create_roboschool_env('RoboschoolHalfCheetah-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHumanoid-v1' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoid-v1'), 2, True),
        'VECENV_TYPE' : 'RAY'
    },
    'LunarLanderContinuous-v2' : {
        'ENV_CREATOR' : lambda : create_roboschool_env('LunarLanderContinuous-v2'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHumanoidFlagrun-v1' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(create_roboschool_env('RoboschoolHumanoidFlagrun-v1'), 2, True),
        'VECENV_TYPE' : 'RAY'
    },
    'BipedalWalker-v2' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalker-v2')), 2, True),
        'VECENV_TYPE' : 'RAY'
    },
    'BipedalWalkerHardcore-v2' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalkerHardcore-v2')), 2, True),
        'VECENV_TYPE' : 'RAY'
    },
    'QuadruppedWalk-v1' : {
        'ENV_CREATOR' : lambda : create_quadrupped_env(),
        'VECENV_TYPE' : 'RAY'
    },

}


def get_obs_and_action_spaces(name):
    env = configurations[name]['ENV_CREATOR']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space

def register(name, config):
    configurations[name] = config