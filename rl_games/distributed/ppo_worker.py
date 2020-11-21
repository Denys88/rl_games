import torch
import numpy as np
import ray


class PPOWorker:
    def __init__(self, config):
        pass

    def set_learning_rate(self, lr):
        pass

    def set_model_params(self, params):
        pass

    def calc_ppo_gradients(self, batch_idx):
        pass

    def set_central_value_learning_rate(self, lr):
        pass

    def set_central_value_model_params(self, params):
        pass

    def calc_central_value_gradients(self, batch_dix):
        pass

    def play_steps(self):
        pass
