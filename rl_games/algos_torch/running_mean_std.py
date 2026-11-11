from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def _running_stats_dtype():
    # MPS doesn't support float64. On a CUDA-less Mac the only accelerator
    # is MPS, so default the running-stat buffers to float32 there. JIT
    # scripting freezes buffer dtypes at script time, so this has to be
    # decided in __init__ rather than inside _apply.
    if not torch.cuda.is_available() and getattr(torch.backends, 'mps', None) is not None \
            and torch.backends.mps.is_available():
        return torch.float32
    return torch.float64


class RunningMeanStd(nn.Module):
    """Tracks the running mean and variance of input data."""
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, init_count=1):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            elif len(self.insize) == 2:
                self.axis = [0, 2]
            elif len(self.insize) == 1:
                self.axis = [0]
            else:
                # Fallback or error?
                # e.g. raise ValueError(f"Unexpected insize length: {len(self.insize)}")
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        dtype = _running_stats_dtype()
        self.register_buffer("running_mean", torch.zeros(in_size, dtype=dtype))
        self.register_buffer("running_var", torch.ones(in_size, dtype=dtype))
        # int64 so the sample count never saturates regardless of the running
        # mean/var dtype. float32 saturates at 2^24 (~16.7M samples), which a
        # long training run can hit and silently freezes the running stats.
        # Module._apply skips non-floating buffers, so .half()/.float() leave
        # this one alone too.
        # init_count seeds the sample count of the zero-mean/unit-var prior.
        # With the historical value of 1 the very first update batch outweighs
        # the prior thousands to one and the stats jump straight to the batch
        # stats, which recomputes early train-time policies far away from the
        # rollout policy that collected the data (a large first-epoch KL). A
        # larger init_count turns that jump into a damped update.
        self.register_buffer("count", torch.full((), int(init_count), dtype=torch.int64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count:int):
        # count is int64; cast to the float dtype for arithmetic only.
        count_f = count.to(mean.dtype)
        tot_count_f = count_f + batch_count

        delta = batch_mean - mean
        new_mean = mean + delta * batch_count / tot_count_f
        m_a = var * count_f
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count_f * batch_count / tot_count_f
        new_var = M2 / tot_count_f
        new_count = count + batch_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm:bool=False, mask:Optional[torch.Tensor]=None):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_var_with_masks(input, mask)
            else:
                mean = input.mean(self.axis) # along channel axis
                var = input.var(self.axis, unbiased=False)

            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size(0)
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
            else:
                current_mean = self.running_mean
                current_var = self.running_var
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        # get output
        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class RunningMeanStdObs(nn.Module):
    """Maintains running statistics for each observation key provided as a dictionary."""
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, init_count=1):
        assert(isinstance(insize, dict))
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict({
            k: RunningMeanStd(v, epsilon, per_channel, norm_only, init_count) for k, v in insize.items()
        })

    def forward(self, input: Dict[str, torch.Tensor], denorm: bool = False) -> Dict[str, torch.Tensor]:
        # loud key-mismatch guard: extra input keys would otherwise be
        # silently dropped, masking an env/config mismatch (missing keys
        # already fail loudly on the dict access below)
        assert len(input) == len(self.running_mean_std), \
            'observation dict keys do not match the normalizer keys'
        res: Dict[str, torch.Tensor] = {}
        for k, m in self.running_mean_std.items():
            res[k] = m(input[k], denorm)
        return res
