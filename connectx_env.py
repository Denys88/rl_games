import gym


class connectx(gym.Env):
    def __init__(self, is_conv, debug):
        super().__init_()
        self.env = make("connectx", debug=debug)
        self.first = True
        self.is_conv = is_conv

    def reset(self, agents=[None, agent]):
        if agents[0] == None:
            self.first = True
        self.trainer = env.train(agents)
        one_hot = self.one_hot_obs(self.trainger.reset())
        obs = self.preprocess(one_hot)
        return obs
        
    def one_hot_obs(self, obs):
        one_hot = np.zeros([obs.size, 3])
        one_hot[np.arange(obs.size), obs] = 1
        one_hot = one_hot[...,[0,1,2]]
        return one_hot

    def preprocess(self, one_hot):
        if self.is_conv:
            return np.reshape(one_hot, [6, 7, 3])
        else:
            return one_hot.flatten()

    
    