class MuZeroModel(object):
    def __init__(self, config):
      pass

    def policy(self, state):
        p = None
        v = None
        return p, v

    def dynamics(self, state, action):
        reward = None
        next_state = None
        return reward, next_state

    def representation(self, obs):
        state = None
        return state
      