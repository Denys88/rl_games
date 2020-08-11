import torch
from torch import nn
import numpy as np
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common  import common_losses



class CentralValueTrain(nn.Module):
    def __init__(self, state_shape, model, config, writter, _preproc_obs):
        nn.Module.__init__(self)
        state_config = {
            'input_shape' : state_shape,
        }
        
        self.model = model.build('cvalue', **cv_config).cuda()
        self.config = config
        self.lr = config['lr']
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        self.clip_value = config['clip_value']
        self.writter = writter
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr))
        self._preproc_obs = _preproc_obs
        self.frame = 0

    def train(self, input_dict):
        self.model.train()

        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        obs_batch = input_dict['states']
        e_clip = input_dict.get('e_clip', 0.2)
        
        num_minibatches = np.shape(obs)[0] // mini_batch
        self.frame = self.frame + 1
        for _ in range(self.mini_epoch):
            # returning loss from last epoch
            avg_loss = 0
            for i in range(self.num_minibatches):
                obs_batch = obs[i * mini_batch: (i + 1) * mini_batch]
                loss = common_losses.critic_loss(value_preds_batch, values, e_clip, clip_value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

        self.writter.add_scalar('cval/train_loss', avg_loss, self.frame)
        return avg_loss / num_minibatches
