class IVecEnv(object):
    def step(self, actions):
        raise NotImplementedError 

    def reset(self):
        raise NotImplementedError    

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        pass