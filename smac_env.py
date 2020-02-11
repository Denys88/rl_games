import gym
import numpy as np
from smac.env import StarCraft2Env

class SMACEnv(gym.Env):
    def __init__(self, name="3m", replay_save_freq=5000, **kwargs):
        gym.Env.__init__(self)
        self.env = StarCraft2Env(map_name=name)
        self.env_info = self.env.get_env_info()
        self.replay_save_freq = replay_save_freq
        self._game_num = 0
        self.n_actions = self.env_info["n_actions"]
        self.n_agents = self.env_info["n_agents"]

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.env_info['obs_shape'] + 1, ), dtype=np.float32)
        self.add_data = np.ones((self.n_agents,1))

    def _preproc_state_obs(self, state, obs):
        #return np.array(obs)
        return np.concatenate((obs, self.add_data), axis=1)

    def get_number_of_agents(self):
        return self.n_agents

    def reset(self):
        if self._game_num % self.replay_save_freq == 1:
            print('saving replay')
            self.env.save_replay()
        self._game_num += 1
        obs, state = self.env.reset()
        obses = self._preproc_state_obs(state, obs)

        return obses

    def step(self, actions):
        rewards, dones, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.env.get_state()
        obses = self._preproc_state_obs(state, obs)
        return obses, rewards, dones, info

    def get_action_mask(self):
        return np.array(self.env.get_avail_actions(), dtype=np.bool)

