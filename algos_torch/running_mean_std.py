import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, insize, momentum=0.99, epsilon=1e-05):

        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.insize = insize
        self.epsilon = epsilon
        if len(self.insize) == 4:
            self.axis = [0,2,3]
        if len(self.insize) == 3:
            self.axis = [0,2]
        if len(self.insize) == 1:
            self.axis = [0]

        self.running_mean = nn.Parameter(torch.zeros(self.insize[1]))
        self.running_std = nn.Parameter(torch.ones(self.insize[1]))

        self.reset_parameters()

    def forward(self, input, mode):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.std(self.axis)
            self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean # .to(input.device)
            self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (input.shape[0]/(input.shape[0]-1)*var)

        else:
            mean = self.running_mean
            var = self.running_var

        # change shape
        current_mean = mean.view([1, self.insize, 1, 1]).expand_as(input)
        current_var = var.view([1, self.insize, 1, 1]).expand_as(input)

        # get output
        y =  (input - current_mean) / (current_var + self.epsilon).sqrt()

        return y