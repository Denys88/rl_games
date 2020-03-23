import tensorflow as tf
import algos_torch.networks
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def is_rnn(self):
        return False
    
    def is_separate_critic(self):
        return False

    


class ModelA2C(BaseModel):
    def __init__(self, network):
        self.network = network
        
    def forward(self, dict):

        is_train = dict.pop('is_train', True)
        action_masks = dict.pop('action_masks', None)
        prev_actions = dict.pop(prev_actions, None)
        inputs = dict.pop('inputs')
        logits, value = self.network(name, inputs=inputs)
        probs = F.softmax(logits, dim=1)

        if not is_train:
            u = torch.rand(logits.size())
            rand_logits = logits - F.math.log(-F.math.log(u))
            if action_masks is not None:
                inf_mask, _ = torch.max(F.log(action_masks.float()), np.iinfo(np.float32).max)
                rand_logits = rand_logits + inf_mask
                logits = logits + inf_mask
            _, selected_action = torch.max(rand_logits, dim=1)
            neglogp = F.cross_entropy(logits, selected_action)
            return  neglogp, value, action, logits
        else:
            entropy = torch.sum(-probs * F.log_softmax(logits, dim=-1), dim=-1).mean()
            prev_neglogp = F.cross_entropy(logits, prev_actions)
            return prev_neglogp, value, None, entropy