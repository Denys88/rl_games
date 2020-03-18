import common.a2c_common

class DiscreteA2CAgent(common.DiscreteA2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        common.DiscreteA2CBase.__init__(self, base_name, observation_space, action_space, config)


    def update_epoch(self):
        pass

    def save(self, fn):
        pass

    def restore(self, fn):
        pass

    def get_masked_action_values(self, obs, action_masks):
        pass

    def get_values(self, obs):
        pass

    def get_weights(self):
        pass
    
    def set_weights(self, weights):
        pass

    def train(self):       
        pass

    def train_actor_critic(self):
        pass 