import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.algos_torch.torch_ext import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal


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


        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class ModelA2CMultiDiscrete(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network

    def build(self, config):
        return ModelA2CMultiDiscrete.Network(self.network_builder.build('a2c', **config))

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            logits, value, states = self.a2c_network(input_dict)
            if is_train:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:   
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]                
                
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
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

        def kl(self, p_dict, q_dict):
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, sigma, value, states = self.a2c_network(input_dict)
            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : torch.squeeze(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
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
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

class ModelSACContinuous(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self)
        self.network_builder = network

    def build(self, config):
        return ModelSACContinuous.Network(self.network_builder.build('sac', **config))
    
    class Network(nn.Module):
        def __init__(self, sac_network):
            nn.Module.__init__(self)
            self.sac_network = sac_network

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist



