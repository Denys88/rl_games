

class BaseAlgorithm:
    def __init__(self, base_name, config):
        pass

    def set_eval(self):
        pass

    def set_train(self):
        pass

    @property
    def device(self):
        pass

    def reset_envs(self):
        pass

    def init_tensors(self):
        pass


    def env_reset(self):
        pass

    def clear_stats(self):
        pass

    def update_epoch(self):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def calc_gradients(self):
        pass

    def get_full_state_weights(self):
        pass

    def set_full_state_weights(self, weights):
        pass

    def get_weights(self):
        pass

    def get_stats_weights(self):
        pass

    def set_stats_weights(self, weights):
        pass

    def set_weights(self, weights):
        pass



