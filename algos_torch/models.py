import algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class BaseModel():
    def __init__(self):
        pass
    def is_rnn(self):
        return False
    
    def is_separate_critic(self):
        return False

    


class ModelA2C(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network

    def build(self, config):
        return ModelA2C.Network(self.network_builder.build('a2c', **config))

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            action_masks = input_dict.pop('action_masks', None)
            prev_actions = input_dict.pop('prev_actions', None)
            inputs = input_dict.pop('inputs')
            logits, value = self.a2c_network(inputs)

            if not is_train:
                u = torch.cuda.FloatTensor(logits.size()).uniform_()
                rand_logits = logits - torch.log(-torch.log(u))
                if action_masks is not None:
                    logits = logits - (1.0 - action_masks) * 1e10

                    #rand_logits = rand_logits + inf_mask
                    #logits = logits + inf_mask
                
                selected_action = torch.distributions.Categorical(logits=logits).sample().long()

                neglogp = F.cross_entropy(logits, selected_action, reduction='none')
                return  neglogp, value, selected_action, logits
            else:
                entropy = -1.0 * ((F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1))).sum(dim=1).mean()
                prev_neglogp = F.cross_entropy(logits, prev_actions, reduction='none')
                return prev_neglogp, value, entropy



class ModelA2CContinuous(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network

    def build(self, config):
        return ModelA2CContinuous.Network(self.network_builder.build('a2c', **config))

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            prev_actions = input_dict.pop('prev_actions', None)
            inputs = input_dict.pop('inputs')
            mu, sigma, value = self.a2c_network(inputs)
            distr = torch.distributions.Normal(mu, sigma)
            if not is_train:
                selected_action = distr.sample().squeeze()
                neglogp = distr.log_prob(selected_action).sum(dim=1)
                return  neglogp, value, selected_action, mu, sigma
            else:
                entropy = distr.entropy().mean()
                prev_neglogp = distr.log_prob(prev_actions).sum(dim=1)
                return prev_neglogp, value, entropy


class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network

    def build(self, config):
        net = self.network_builder.build('a2c', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelA2CContinuousLogStd.Network(net)

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            prev_actions = input_dict.pop('prev_actions', None)
            inputs = input_dict.pop('inputs')

            mu, logstd, value = self.a2c_network(inputs)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if not is_train:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                return  neglogp, value, selected_action, mu, sigma
            else:
                entropy = distr.entropy().sum(dim=-1).mean()
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return prev_neglogp, value, entropy, mu, sigma

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

