import torch
from torch import nn
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def multiply_hidden(h, mask):
    if isinstance(h, torch.Tensor):
        return h * mask
    else:
        return tuple(multiply_hidden(v, mask) for v in h)

class RnnWithDones(nn.Module):
    def __init__(self, rnn_layer):
        nn.Module.__init__(self)
        self.rnn = rnn_layer


    def forward(self, input, states, done_masks=None, bptt_len = 0):
        max_steps = input.size()[0]
        batch_size = input.size()[1]
        out_batch = []

        for i in range(max_steps):
            if done_masks is not None:
                dones = done_masks[i].float().unsqueeze(0)
                states = multiply_hidden(states, 1.0-dones)
            if (bptt_len > 0) and (i % bptt_len == 0):
                states = repackage_hidden(states)
            out, states = self.rnn(input[i].unsqueeze(0), states)
            out_batch.append(out)
        return torch.cat(out_batch, dim=0), states


class LSTMWithDones(RnnWithDones):
    def __init__(self, *args, **kwargs):
        lstm = torch.nn.LSTM(*args, **kwargs)
        RnnWithDones.__init__(self, lstm)

class GRUWithDones(RnnWithDones):
    def __init__(self, *args, **kwargs):
        gru = torch.nn.GRU(*args,**kwargs)
        RnnWithDones.__init__(self, gru)