class CFREnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.obs = self.env.reset()
        self.reward = None
        self.done = False
        self.info = None
        return self.obs

    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        return self.obs, self.reward, self.done, self.info
