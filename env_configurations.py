import roboschool
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import networks 
import wrappers
import tr_helpers
import quadruppedEnv

def create_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    #env = wrappers.MaxAndSkipEnv(env, skip=2)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    return env

a2c_configurations = {
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
        'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(clip_value = 0, scale_value = 1.0/100.0),
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
        'ENV_CREATOR' : lambda : gym.make('RoboschoolAnt-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'SuperMarioBros-v1' : {
        'ENV_CREATOR' : lambda :  create_super_mario_env(),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHalfCheetah-v1' : {
        'ENV_CREATOR' : lambda : gym.make('RoboschoolHalfCheetah-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHumanoid-v1' : {
        'ENV_CREATOR' : lambda : gym.make('RoboschoolHumanoid-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'LunarLanderContinuous-v2' : {
        'ENV_CREATOR' : lambda : gym.make('LunarLanderContinuous-v2'),
        'VECENV_TYPE' : 'RAY'
    },
    'BipedalWalker-v2' : {
        'ENV_CREATOR' : lambda : gym.make('BipedalWalker-v2'),
        'VECENV_TYPE' : 'RAY'
    },
    'QuadruppedWalk-v1' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(wrappers.MaxAndSkipEnv(gym.make('QuadruppedWalk-v1'),8, False), 4, True),
        'VECENV_TYPE' : 'RAY'
    },
    'BipedalWalkerHardcore-v2' : {
        'ENV_CREATOR' : lambda : wrappers.FrameStack(gym.make('BipedalWalkerHardcore-v2'),4, True),
        'VECENV_TYPE' : 'RAY'
    },
}


def get_obs_and_action_spaces(name):
    env = a2c_configurations[name]['ENV_CREATOR']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space