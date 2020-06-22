import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd


# todo remove it completely
class CentralValueNetwork(nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network


class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, model, config, writter, _preproc_obs):
        nn.Module.__init__(self)
        state_config = {
            'input_shape' : state_shape,
        }
        
        self.model = CentralValueNetwork(model.build('cvalue', **cv_config)).cuda()
        self.config = config
        self.lr = config['lr']
        self.mini_epoch = self.config['mini_epochs']
        self.mini_batch = self.config['minibatch_size']

        self.writter = writter
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr))
        self._preproc_obs = _preproc_obs
        self.output_normalization = RunningMeanStd((1.0,), norm_only=True).cuda()
        self.frame = 0
        self.exp_percent = config.get('exp_percent', 1.0)

    def get_loss(self, obs):
        obs = self._preproc_obs(obs)
        self.model.eval()
        self.output_normalization.train()
        with torch.no_grad():
            loss = self.model(obs)
            loss = loss.squeeze()
            loss = self.output_normalization(loss)
            
            return loss.cpu()

    def train(self, obs):
        self.model.train()

        states = batch_dict['a_states']

        num_minibatches = np.shape(obs)[0] // mini_batch
        self.frame = self.frame + 1
        for _ in range(self.mini_epoch):
            # returning loss from last epoch
            avg_loss = 0
            for i in range(self.num_minibatches):
                obs_batch = obs[i * mini_batch: (i + 1) * mini_batch]
                obs_batch = self._preproc_obs(obs_batch)
                obs_batch = torch_ext.random_sample(obs_batch, self.exp_percent)
                loss = self.model(obs_batch).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

        self.writter.add_scalar('cval/train_loss', avg_loss, self.frame)
        return avg_loss / num_minibatches
