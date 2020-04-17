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