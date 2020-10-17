import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl

def mean_mask(input, mask, sum_mask):
    return (input * rnn_masks).sum() / sum_mask

def shape_whc_to_cwh(shape):
    if len(shape) == 2:
        return (shape[1], shape[0])
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])
    
    return shape

def save_scheckpoint(filename, state):
    print("=> saving checkpoint '{}'".format(filename + '.pth'))

    torch.save(state, filename + '.pth')

def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename + '.pth'))
    state = torch.load(filename)
    return state

def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x

def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
    return parameterized_truncated_normal(uniform, mu, sigma, a, b)

def sample_truncated_normal(shape=(), mu=0.0, sigma=1.0, a=-2, b=2):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)), mu, sigma, a, b)

def variance_scaling_initializer(tensor, mode='fan_in',scale = 2.0):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    sigma = np.sqrt(1.3 * scale / fan)
    with torch.no_grad():
        return sample_truncated_normal(tensor.size(), sigma=sigma)


def random_sample(obs_batch, prob):
    num_batches = obs_batch.size()[0]
    permutation = torch.randperm(num_batches).cuda()
    start = 0
    end = int(prob * num_batches)
    indices = permutation[start:end]
    return torch.index_select(obs_batch, 0, indices)


def apply_masks(losses, mask=None):
    sum_mask = None
    if mask is not None:
        sum_mask = mask.sum()
        res_losses = [(l * mask).sum() / sum_mask for l in losses]
    else:
        res_losses = [torch.mean(l) for l in losses]
    
    return res_losses, sum_mask



class CoordConv2d(nn.Conv2d):
    pool = {}
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels + 2, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
    @staticmethod
    def get_coord(x):
        key = int(x.size(0)), int(x.size(2)), int(x.size(3)), x.type()
        if key not in CoordConv2d.pool:
            theta = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
            coord = torch.nn.functional.affine_grid(theta, torch.Size([1, 1, x.size(2), x.size(3)])).permute([0, 3, 1, 2]).repeat(
                x.size(0), 1, 1, 1).type_as(x)
            CoordConv2d.pool[key] = coord
        return CoordConv2d.pool[key]
    def forward(self, x):
        return torch.nn.functional.conv2d(torch.cat([x, self.get_coord(x).type_as(x)], 1), self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LayerNorm2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.register_buffer("gamma", torch.ones(features).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("beta", torch.ones(features).unsqueeze(-1).unsqueeze(-1))

        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)



class DiscreteActionsEncoder(nn.Module):
    def __init__(self, actions_max, mlp_out, emb_size, num_agents, use_embedding):
        super().__init__()
        self.actions_max = actions_max
        self.emb_size = emb_size
        self.num_agents = num_agents
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = torch.nn.Embedding(actions_max, emb_size)
        else:
            self.emb_size = actions_max
        
        self.linear = torch.nn.Linear(self.emb_size * num_agents, mlp_out)

    def forward(self, discrete_actions):
        if self.use_embedding:
            emb = self.embedding(discrete_actions)
        else:
            emb = torch.nn.functional.one_hot(discrete_actions, num_classes=self.actions_max)
        emb = emb.view( -1, self.emb_size * self.num_agents).float()
        emb = self.linear(emb)
        return emb