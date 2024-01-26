import gym
import numpy as np
from rl_games.common import wrappers

class MarioEnv(gym.Env):
    def __init__(self, **kwargs):
        env_name=kwargs.pop('env_name', 'SuperMarioBros-v1')
        self.max_lives = kwargs.pop('max_lives', 16)
        self.movement = kwargs.pop('movement', 'SIMPLE')
        self.use_dict_obs_space = kwargs.pop('use_dict_obs_space', False)
        self.env = self._create_super_mario_env(env_name)
        if self.use_dict_obs_space:
            self.observation_space= gym.spaces.Dict({
                'observation' : self.env.observation_space,
                'reward' : gym.spaces.Box(low=0, high=1, shape=( ), dtype=np.float32),
                'last_action': gym.spaces.Box(low=0, high=self.env.action_space.n, shape=(), dtype=int)
            })
        else:
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space


    def _create_super_mario_env(self, name='SuperMarioBros-v1'):
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
        import gym_super_mario_bros
        movement = SIMPLE_MOVEMENT if self.movement == 'SIMPLE' else COMPLEX_MOVEMENT
        env = gym_super_mario_bros.make(name)
        env = JoypadSpace(env, movement)
        if 'Random' in name:
            env = wrappers.EpisodicLifeRandomMarioEnv(env)
        else:
            env = wrappers.EpisodicLifeMarioEnv(env, self.max_lives)
        env = wrappers.MaxAndSkipEnv(env, skip=4)
        env = wrappers.wrap_deepmind(
            env, episode_life=False, clip_rewards=False, frame_stack=True, scale=True, gray=False)
        return env

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        if self.use_dict_obs_space:
            next_obs = {
                'observation': next_obs,
                'reward': np.array(reward, dtype=float),
                'last_action': np.array(action, dtype=int)
            }
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset()
        
        if self.use_dict_obs_space:
            obs = {
                'observation': obs,
                'reward': np.array(0.0, dtype=float),
                'last_action': np.array(0, dtype=int),
            }
        return obs

    def render(self, mode, **kwargs):
        self.env.render(mode, **kwargs)

    def get_number_of_agents(self):
        return 1