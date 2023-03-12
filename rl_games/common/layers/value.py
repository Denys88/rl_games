import torch
from torch import nn
from rl_games.common import common_losses
from rl_games.algos_torch.layers import symexp, symlog
from rl_games.common.extensions.distributions import TwoHotDist


class DefaultValue(nn.Module):
    def __init__(self, in_size, out_size):
        nn.Module.__init__(self)
        self.value_linear = nn.Linear(in_size, out_size)
        
    def loss(self, **kwargs):
        values = kwargs.get(value)
        targets = kwargs.get(targets) 
        returns = kwargs.get(returns)
        clip_value = kwargs.get(clip_value)

        return common_losses.critic_loss(values, targets, returns, clip_value)

    def forward(self, input):
        return self.value_linear(input)


class TwoHotEncodedValue(nn.Module):
    def __init__(self, in_size, out_size):
        nn.Module.__init__(self)
        assert(out_size==1)
        self.value_linear = nn.Linear(in_size, out_size)
        self.distr = TwoHotDist(logits=out)
        
    def loss(self, **kwargs):
        targets = kwargs.get(targets)
        x = symlog(targets)
        return -self.distr.log_prob(x) 

    def forward(self, input):
        distr = TwoHotDist()
        out = self.value_linear(input)
        self.distr = TwoHotDist(logits=out)
        out = self.distr.mode()
        return symexp(out)
        