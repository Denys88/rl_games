import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, insize, momentum=0.99, epsilon=1e-05):

        super(RunningMeanStd, self).__init__()
        self.momentum = momentum
        self.insize = insize
        self.epsilon = epsilon
        if len(self.insize) == 3:
            self.axis = [0,2,3]
        if len(self.insize) == 2:
            self.axis = [0,2]
        if len(self.insize) == 1:
            self.axis = [0]

        self.register_buffer("running_mean", torch.zeros(self.insize[0]))
        self.register_buffer("running_std", torch.ones(self.insize[0]))

    def forward(self, input):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            std = input.std(self.axis)
            self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean # .to(input.device)
            self.running_std = (self.momentum * self.running_std) + (1.0-self.momentum) * std

        else:
            mean = self.running_mean
            std = self.running_std

        # change shape
        if len(self.insize) == 3:
            current_mean = mean.view([1, self.insize[0], 1, 1]).expand_as(input)
            current_std = std.view([1, self.insize[0], 1, 1]).expand_as(input)
        if len(self.insize) == 2:
            current_mean = mean.view([1, self.insize[0], 1]).expand_as(input)
            current_std = std.view([1, self.insize[0], 1]).expand_as(input)
        if len(self.insize) == 1:
            current_mean = mean.view([1, self.insize[0]]).expand_as(input)
            current_std = std.view([1, self.insize[0]]).expand_as(input)            
        # get output
        y =  (input - current_mean) / (current_std + self.epsilon).sqrt()
        print(mean, std)
        return y