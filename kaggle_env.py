import gym
from kaggle_environments import evaluate, make
import numpy as np
import tensorflow as tf
import yaml
from runner import Runner
import dqnagent

class KaggleEnv(gym.Env):
    def __init__(self, name="connectx", is_conv=True, **kwargs):
        gym.Env.__init__(self)
        self.sess = kwargs.pop('sess', None)
   
        self.env = make(name, **kwargs)
        self.is_first = True
        self.is_conv = is_conv
        self.shuffle_agents = True

        self.action_space = gym.spaces.Discrete(7)
        if self.is_conv:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, 6, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6 * 7 *3, ), dtype=np.uint8)

        self.create_agent()
        self.agents = [None, lambda observation, configuration: self.my_agent(observation, configuration)]#self.random_agent
        #self.agents = [None, 'random']
    def update_weights(self, weigths):
        self.agent.set_weights(weigths)

    def build_obs(is_first, is_conv, obs):
        obs = obs['board']
        one_hot = KaggleEnv._one_hot_obs(is_first, obs)
        obs = KaggleEnv._preprocess(is_conv, one_hot)
        return obs

    def _one_hot_obs(is_first, obs):
        one_hot = np.zeros((7*6, 3), dtype=np.uint8)
        one_hot[np.arange(7*6), obs] = 1
        if not is_first:
            one_hot = one_hot[...,[0,2,1]]
        return one_hot

    def _preprocess(is_conv, one_hot):
        if is_conv:
            return np.reshape(one_hot, [7, 6, 3])
        else:
            return one_hot.flatten().astype(np.float32)

    def _reward_func(self, reward):
        if reward is None:
            return -2.0
        return reward * 2.0 - 1.0

    def reset(self):
        if self.shuffle_agents:
            self.agents[0], self.agents[1] = self.agents[1], self.agents[0]

        self.is_first = self.agents[0] == None
        self.trainer = self.env.train(self.agents)
        obs = self.trainer.reset()
        return KaggleEnv.build_obs(self.is_first, self.is_conv, obs)

    def step(self, action):
        action = int(action)
        next_state, reward, is_done, info = self.trainer.step(action)
        next_state = KaggleEnv.build_obs(self.is_first, self.is_conv, next_state)
        shaped_reward = self._reward_func(reward)
        if reward is None:
            print('atata:', is_done, shaped_reward)
        return next_state, shaped_reward, is_done, info
    
    def create_agent(self, config='configs/connectx_ppo.yaml'):
        if self.sess is None:
            return
        with tf.variable_scope('kaggle_env'):
            with open(config, 'r') as stream:
                config = yaml.safe_load(stream)
                runner = Runner()
                runner.load(config)
                runner.sess = self.sess
            config = runner.get_prebuilt_config()
            config['env_name'] = None
            self.agent = runner.create_agent(self.observation_space, self.action_space)

    def my_agent(self, observation, configuration):
        if np.sum(observation.board) == 0:
            return np.random.randint(7)
        obs = KaggleEnv.build_obs(not self.is_first, self.is_conv, observation)
        action, _, _ = self.agent.get_action_value([obs])
        if observation.board[action] == 1:
            for i in range(7):
                if observation.board[i] == 0:
                    return i
  
        return int(action)

    def random_agent(self, observation, configuration):
        from random import choice
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])