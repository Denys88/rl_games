import gym
import roboschool

configurations = {
    'RoboschoolAnt-v1' : {
        'ENV_CREATOR' : lambda : gym.make('RoboschoolAnt-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHalfCheetah-v1' : {
        'ENV_CREATOR' : lambda : gym.make('RoboschoolHalfCheetah-v1'),
        'VECENV_TYPE' : 'RAY'
    },
    'RoboschoolHumanoid-v1' : {
        'ENV_CREATOR' : lambda : gym.make('RoboschoolHumanoid-v1'),
        'VECENV_TYPE' : 'RAY'
    }
}