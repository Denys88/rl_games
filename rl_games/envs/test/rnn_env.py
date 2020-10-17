import gym
import numpy as np


class TestRNNEnv(gym.Env):
    def __init__(self,  **kwargs):
        gym.Env.__init__(self)
        self.n_actions = 4
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)
        self.state_space = gym.spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32)
        self.obs_dict = {}
        self.max_steps = kwargs.pop('max_steps', 20)
        self.show_time = kwargs.pop('show_time', 2)
        self.min_dist = kwargs.pop('min_dist', 2)
        self.max_dist = kwargs.pop('min_dist', 8)
        self.hide_object = kwargs.pop('hide_object', False)
        self.reset()

    def get_number_of_agents(self):
        return 1

    def reset(self):
        self._curr_steps = 0
        self._current_pos = [0,0]
        bound = self.max_dist - self.min_dist
        rand_dir = - 2 * np.random.randint(0, 2, (2,)) + 1
        self._goal_pos = rand_dir * np.random.randint(self.min_dist, self.max_dist+1, (2,))
        obs = np.concatenate([self._current_pos, self._goal_pos], axis=None)
        return obs.astype(np.float32)

    def step(self, action):
        info = {}
        if action == 0:
            self._current_pos[0] += 1
        if action == 1:
            self._current_pos[0] -= 1
        if action == 2:
            self._current_pos[1] += 1
        if action == 3:
            self._current_pos[1] -= 1    
        self._curr_steps += 1

        if self._curr_steps > 1 and self.hide_object:
            obs = np.concatenate([self._current_pos, [0,0]], axis=None)
        else:
            obs = np.concatenate([self._current_pos, self._goal_pos], axis=None)
        
        reward = 0.0
        done = False
        
        if ((self._current_pos - self._goal_pos)**2).sum() < 0.0001:
            reward = 1.0
            done = True
        if self._curr_steps == self.max_steps:
            reward = -1.0
            done = True
        return obs.astype(np.float32), reward, done, info
    
    def has_action_mask(self):
        return False