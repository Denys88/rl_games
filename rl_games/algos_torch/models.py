import rl_games.algos_torch.layers
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
        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            action_masks = input_dict.pop('action_masks', None)
            prev_actions = input_dict.pop('prev_actions', None)
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                categorical = torch.distributions.Categorical(logits=logits)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : torch.squeeze(value),
                    'entropy' : entropy,
                    'rnn_state' : states
                }
                return result
            else:
                if action_masks is not None:
                    inf_mask = torch.log(action_masks.float())
                    logits = logits + inf_mask

                categorical = torch.distributions.Categorical(logits=logits)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogp' : torch.squeeze(neglogp),
                    'value' : torch.squeeze(value),
                    'action' : selected_action,
                    'logits' : logits,
                    'rnn_state' : states
                }
                return  result


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

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            prev_actions = input_dict.pop('prev_actions', None)
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : torch.squeeze(value),
                    'entropy' : entropy,
                    'rnn_state' : states,
                    'mu' : mu,
                    'sigma' : sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogp' : torch.squeeze(neglogp),
                    'value' : torch.squeeze(value),
                    'action' : selected_action,
                    'entropy' : entropy,
                    'rnn_state' : states,
                    'mu' : mu,
                    'sigma' : sigma
                }
                return  result          


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

        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            prev_actions = input_dict.pop('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                #prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                #print(((prev_neglogp - prev_neglogp2)**2).sum())
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : torch.squeeze(value),
                    'entropy' : entropy,
                    'rnn_state' : states,
                    'mu' : mu,
                    'sigma' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogp' : torch.squeeze(neglogp),
                    'value' : torch.squeeze(value),
                    'action' : selected_action,
                    'rnn_state' : states,
                    'mu' : mu,
                    'sigma' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

