import torch
from torch import nn


class RNDCuriosityNetwork(nn.Module):
    def __init__(self, config, observation_space)
        nn.Module.__init__(self)
        self.random_network = None
        self.network = None
        self.obs_shape = observation_space.shape

    def forward(self, obs):
        with torch.no_grads():
            rnd_res = self.random_network(obs)
            
        res = self.network(obs)

        loss = (res - rnd_res)**2.mean()

        return loss



class RNDCurisityTrain:
    def __init__(self, model, config, writter, preproc_obs_func):
        self.model = model
        self.config = config
        self.lr = config['lr']
        self.writter = writter
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self._preproc_obs = preproc_obs_func

    def train(self, obs):
        mini_epoch = config['mini_epoch']
        mini_batch = config['mini_batch']

        num_minibatches = obs.size()[0] / mini_batch
        for _ in range(mini_epoch):
            for i range(num_minibatches):
                obs_batch = obs[i * mini_batch: (i + 1) * mini_batch]
                obs_batch = self._preproc_obs(obs_batch)
                loss = self.model(obs_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss.item()
