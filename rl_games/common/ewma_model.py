from copy import deepcopy
import torch
import numpy as np

class EwmaModel(torch.nn.Module):
    '''
    https://github.com/openai/ppo-ewma/blob/master/ppo_ewma/ppo.py
    '''
    def __init__(self, model, ewma_decay):
        super().__init__()
        self.model = model
        self.ewma_decay = ewma_decay
        self.model_ewma = deepcopy(model)
        self.total_weight = 1

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model_ewma(*args, **kwargs)

    def update(self, decay=None):
        if decay is None:
            decay = self.ewma_decay
        new_total_weight = decay * self.total_weight + 1
        decayed_weight_ratio = decay * self.total_weight / new_total_weight
        for p, p_ewma in zip(self.model.parameters(), self.model_ewma.parameters()):
            p_ewma.data.mul_(decayed_weight_ratio).add_(p.data / new_total_weight)
        self.total_weight = new_total_weight

    def reset(self):
        self.update(decay=0)


class AdamEwmaUpdate():
    def __init__(self, c):
        self.c = c

    def __call__(self, adam):
        c = self.c
        for g in adam.param_groups:
            g['lr'] = g['lr'] / np.sqrt(c)
            g['betas'] = (g['betas'][0]**(1/c), g['betas'][1]**(1/c))
