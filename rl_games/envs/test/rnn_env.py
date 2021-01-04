import gym
import numpy as np


class TestRNNEnv(gym.Env):
    def __init__(self,  **kwargs):
        gym.Env.__init__(self)
        self.n_actions = 4
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32)
        self.state_space = gym.spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32)
        self.obs_dict = {}
        self.max_steps = kwargs.pop('max_steps', 61)
        self.show_time = kwargs.pop('show_time', 1)
        self.min_dist = kwargs.pop('min_dist', 2)
        self.max_dist = kwargs.pop('max_dist', 8)
        self.hide_object = kwargs.pop('hide_object', False)
        self.use_central_value = kwargs.pop('use_central_value', False)
        self.apply_dist_reward = kwargs.pop('apply_dist_reward', False)
        self.apply_exploration_reward = kwargs.pop('apply_exploration_reward', False)
        if self.apply_exploration_reward:
            pass
        self.reset()

    def get_number_of_agents(self):
        return 1

    def reset(self):
        self._curr_steps = 0
        self._current_pos = [0,0]
        bound = self.max_dist - self.min_dist
        rand_dir = - 2 * np.random.randint(0, 2, (2,)) + 1
        self._goal_pos = rand_dir * np.random.randint(self.min_dist, self.max_dist+1, (2,))
        obs = np.concatenate([self._current_pos, self._goal_pos, [1, 0]], axis=None)
        #print(self._goal_pos)
        if self.use_central_value:
            obses = {}
            obses["obs"] = obs.astype(np.float32)
            obses["state"] = obs.astype(np.float32)
        else:
            obses = obs.astype(np.float32)
        return obses

    def step(self, action):
        info = {}  
        self._curr_steps += 1

        if self._curr_steps > 1:
            if action == 0:
                self._current_pos[0] += 1
            if action == 1:
                self._current_pos[0] -= 1
            if action == 2:
                self._current_pos[1] += 1
            if action == 3:
                self._current_pos[1] -= 1          
        reward = 0.0
        done = False
        dist = self._current_pos - self._goal_pos
        if (dist**2).sum() < 0.0001:
            reward = 1.0
            info = {'scores' : 1} 
            done = True
        elif self._curr_steps == self.max_steps:
            reward = -1.0
            info = {'scores' : 0} 
            done = True

        dist_coef = -0.1
        if self.apply_dist_reward:
            reward += dist_coef * np.abs(dist).sum() / self.max_dist

        show_object = 0
        if self.hide_object:
            obs = np.concatenate([self._current_pos, [0,0], [show_object, self._curr_steps]], axis=None)
        else:
            show_object = 1
            obs = np.concatenate([self._current_pos, self._goal_pos, [show_object, self._curr_steps]], axis=None)

        if self.use_central_value:
            state = np.concatenate([self._current_pos, self._goal_pos, [show_object, self._curr_steps]], axis=None)
            obses = {}
            obses["obs"] = obs.astype(np.float32)
            obses["state"] = obs.astype(np.float32)
        else:
            obses = obs.astype(np.float32)
        return obses, reward, done, info
    
    def has_action_mask(self):
        return False