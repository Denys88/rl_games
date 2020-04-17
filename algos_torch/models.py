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
        return ModelA2C.Network(self.network_builder.build('a2c', **config))

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
        return ModelA2C.Network(self.network_builder.build('a2c', **config))

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            prev_actions = input_dict.pop('prev_actions', None)
            inputs = input_dict.pop('inputs')
            mu, logstd, value = self.a2c_network(inputs)
            sigma = tf.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if not is_train:
                selected_action = distr.sample().squeeze()
                neglogp = self.neglogp(selected_action, mean, sigma, logstd)
                return  neglogp, value, selected_action, mu, sigma
            else:
                entropy = distr.entropy().mean()
                prev_neglogp = self.neglogp(prev_actions, mean, sigma, logstd)
                return prev_neglogp, value, entropy

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (tf.square((x - mean) / std).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]) \
                + logstd.sum(dim=-1)

class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        self.network = network

    def __call__(self, dict, reuse=False):

        name = dict['name']
        inputs = dict['inputs']
        actions_num = dict['actions_num']
        prev_actions_ph = dict['prev_actions_ph']
        is_train = prev_actions_ph is not None

        mean, logstd, value = self.network(name, inputs=inputs, actions_num=actions_num, continuous=True, is_train=True, reuse=reuse)
    

        std = tf.exp(logstd)
        norm_dist = tfd.Normal(mean, std)

        action = mean + std * tf.random_normal(tf.shape(mean))
        #action = tf.squeeze(norm_dist.sample(1), axis=0)
        #action = tf.clip_by_value(action, -1.0, 1.0)
        
        entropy = tf.reduce_mean(tf.reduce_sum(norm_dist.entropy(), axis=-1))
        if prev_actions_ph is None:
            neglogp = self.neglogp(action, mean, std, logstd)
            return  neglogp, value, action, entropy, mean, std

        prev_neglogp = self.neglogp(prev_actions_ph, mean, std, logstd)
        return prev_neglogp, value, action, entropy, mean, std

    def neglogp(self, x, mean, std, logstd):
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
            + tf.reduce_sum(logstd, axis=-1)
