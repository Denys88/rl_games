import torch
import torch.nn as nn
import numpy as np
import rl_games.algos_torch.torch_ext as torch_ext


class GeneralizedMovingStats(nn.Module):
    '''
    Updates moving statistics with momentum
    '''
    def __init__(
        self, insize, impl='mean_std', decay=0.99, max=1e5, eps=0.0, perclo=0.05,
        perchi=0.95
    ):
        super().__init__()
        self.impl = impl
        self.decay = decay
        self.max = max
        self.eps = eps
        self.perclo = perclo
        self.perchi = perchi
        if self.impl == 'off':
            pass
        elif self.impl == 'mean_std':
            self.step = torch.nn.Parameter(torch.ones((1), dtype=torch.int32), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
            self.sqrs = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        elif self.impl == 'mean_std_corr':
            self.step = torch.nn.Parameter(torch.ones((1), dtype=torch.int32), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
            self.sqrs = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)            
        elif self.impl == 'min_max':
            self.low = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        elif self.impl == 'perc_ema':
            self.low = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        elif self.impl == 'perc_ema_corr':
            self.step = torch.nn.Parameter(torch.ones((1), dtype=torch.int32), requires_grad=False)
            self.low = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        elif self.impl == 'mean_mag':
            self.mag = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        elif self.impl == 'max_mag':
            self.mag = torch.nn.Parameter(torch.zeros((insize), dtype=torch.float32), requires_grad=False)
        else:
            raise NotImplementedError(self.impl)

    def _get_stats(self):
        if self.impl == 'off':
            return 0.0, 1.0
        elif self.impl == 'mean_std':
            mean = self.mean
            var = self.sqrs - mean.pow(2)
            std = torch.sqrt(torch.clamp_min(var, 1 / self.max ** 2) + self.eps)
            return mean, std
        elif self.impl == 'mean_std_corr':
            corr = 1.0 - self.decay ** self.step.float()
            mean = self.mean / corr
            var = (self.sqrs / corr) - (self.mean / corr).pow(2)
            std = torch.sqrt(torch.clamp_min(var, 1 / self.max ** 2) + self.eps)
            return mean, std
        elif self.impl == 'min_max':
            offset = self.low
            invscale = torch.clamp_min(self.high-self.low, 1/self.max)
            return offset, invscale
        elif self.impl == 'perc_ema':
            offset = self.low
            invscale = torch.clamp_min(self.high - self.low, 1 / self.max)
            return offset, invscale
        elif self.impl == 'perc_ema_corr':
            corr = 1 - self.decay ** self.step.float()
            lo = self.low / corr
            hi = self.high / corr
            invscale = torch.clamp_min(hi - lo, 1 / self.max)
            return lo, invscale
        else:
            raise NotImplementedError(self.impl)

    @torch.compile(mode="reduce-overhead")
    def update_moving_stats(self, mean, sqrs, x, m):
        """
        Optimized moving statistics update using torch.compile for better performance.

        Args:
            mean: The current mean parameter
            sqrs: The current squared values parameter
            x: Input data to update statistics with
            m: Momentum factor

        Returns:
            Updated mean and sqrs values
        """
        x_mean = torch.mean(x, dim=0)
        # Avoid creating temporary tensor with x**2
        x_sqr_mean = torch.mean(x * x, dim=0)
        mean_factor = 1 - m
        mean.mul_(m).add_(mean_factor * x_mean)
        sqrs.mul_(m).add_(mean_factor * x_sqr_mean)
        return mean, sqrs

    def _update_stats(self, x):
        m = self.decay
        if self.impl == 'off':
            pass
        elif self.impl == 'mean_std':
            self.step.data += 1
            # Use the compiled function instead of separate operations
            self.mean, self.sqrs = self.update_moving_stats(self.mean.data, self.sqrs.data, x, m)
        elif self.impl == 'mean_std_corr':
            self.step.data += 1
            # Use the compiled function
            self.mean, self.sqrs = self.update_moving_stats(self.mean.data, self.sqrs.data, x, m)
        elif self.impl == 'min_max':
            low, high = torch.min(x), torch.max(x)
            self.low.data.mul_(m).add_((1 - m) * torch.minimum(self.low.data, low))
            self.high.data.mul_(m).add_((1 - m) * torch.maximum(self.high.data, high))
        elif self.impl == 'perc_ema':
            low, high = torch.quantile(x, self.perclo), torch.quantile(x, self.perchi)
            self.low.data.mul_(m).add_((1 - m) * low)
            self.high.data.mul_(m).add_((1 - m) * high)
        elif self.impl == 'perc_ema_corr':
            self.step.data += 1
            low, high = torch.quantile(x, self.perclo), torch.quantile(x, self.perchi)
            self.low.data.mul_(m).add_((1 - m) * low)
            self.high.data.mul_(m).add_((1 - m) * high)

    def forward(self, input, mask=None, denorm=False):
        if self.training:
            self._update_stats(input)

        offset, invscale = self._get_stats()
        if denorm:
            output = torch.empty_like(input)
            output.copy_(input)
            output.mul_(invscale).add_(offset)
        else:
            output = torch.empty_like(input)
            output.copy_(input)
            output.sub_(offset).div_(invscale)
            output.clamp_(-5.0, 5.0)
        return output
