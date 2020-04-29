import torch
from torch import nn


class RNDCuriosity(nn.Module):
    def __init__(self, config, observation_space)
        self.random_network = None
        self.network = None
        self.obs_shape = observation_space.shape

    def forward(self, obs):
        with torch.no_grads():
            rnd_res = self.random_network(obs)
            
        res = self.network(obs)

        loss = (res - rnd_res)**2.mean()

        return loss



