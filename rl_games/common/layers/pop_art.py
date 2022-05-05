import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopArt(torch.nn.Module):

    def __init__(self, linear_layer, norm_axes=1, beta=0.99999, epsilon=1e-5):

        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes

        self.linear_layer = linear_layer
        self.stddev = nn.Parameter(torch.ones(linear_layer.out_features), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(linear_layer.out_features), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(linear_layer.out_features), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.reset_parameters()

    @property
    def running_mean(self):
        mean, var = self.debiased_mean_var()
        return mean

    @property
    def running_var(self):
        mean, var = self.debiased_mean_var()
        return var

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.linear_layer.weight, a=math.sqrt(5))
        if self.linear_layer.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear_layer.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.linear_layer.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input, mask=None, unnorm=False):
        assert  mask==None
        if self.training:
            self.update(input)
        # get output
        if unnorm:
            y = self.denormalize(input)
        else:
            y = self.normalize(input)
        return y

    @torch.no_grad()
    def update(self, input):
        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = input.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev.data = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        self.linear_layer.weight.data = self.linear_layer.weight * old_stddev / new_stddev
        self.linear_layer.bias.data = (old_stddev * self.linear_layer.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out