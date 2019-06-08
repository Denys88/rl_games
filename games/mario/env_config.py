import gym
import wrappers
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

def create_super_mario_env():
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    #env = wrappers.MaxAndSkipEnv(env, skip=2)
    env = wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    return env

    
configurations = {
    'SuperMarioBros-v1' : {
        'ENV_CREATOR' : lambda :  create_super_mario_env(),
        'VECENV_TYPE' : 'RAY'
    },
}