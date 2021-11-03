import torch
import torch.nn as nn
import numpy as np
import rl_games.algos_torch.torch_ext as torch_ext

'''
updates moving statistics with momentum
'''
class MovingMeanStd(nn.Module):
    def __init__(self, insize, momentum = 0.25, epsilon=1e-05, per_channel=False, norm_only=False):
        super(MovingMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.momentum = momentum
        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("moving_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("moving_var", torch.ones(in_size, dtype = torch.float64))

    def forward(self, input, mask=None, unnorm=False):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis) # along channel axis
                var = input.var(self.axis)
            
            self.moving_mean = self.moving_mean * self.momentum + mean * (1 - self.momentum)
            self.moving_var = self.moving_var * self.momentum + var * (1 - self.momentum)

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.moving_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.moving_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.moving_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.moving_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.moving_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.moving_var.view([1, self.insize[0]]).expand_as(input)        
        else:
            current_mean = self.moving_mean
            current_var = self.moving_var
        # get output
        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)
        return y