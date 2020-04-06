import common.a2c_common
from torch import optim
import torch 
from torch import nn
class DiscreteA2CAgent(common.a2c_common.DiscreteA2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        common.a2c_common.DiscreteA2CBase.__init__(self, base_name, observation_space, action_space, config)
        
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : self.state_shape,
            'games_num' : 1,
            'batch_num' : 1,
        } 
        self.model = self.network.build(config)
        self.model.cuda()
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr))

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        pass

    def restore(self, fn):
        pass

    def get_masked_action_values(self, obs, action_masks):
        obs = torch.Tensor(obs).cuda()
        action_masks = torch.Tensor(action_masks).cuda()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
            'action_masks' : action_masks
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        return action.detach().cpu().numpy(), value.detach().cpu().numpy(), neglogp.detach().cpu().numpy(), None, None


    def get_action_values(self, obs):
        obs = torch.Tensor(obs).cuda()
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        return action.detach().cpu().numpy(), value.detach().cpu().numpy(), neglogp.detach().cpu().numpy(), None

    def get_values(self, obs):
        obs = torch.Tensor(obs).cuda()
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs
        }
        with torch.no_grad():
            neglogp, value, action, logits = self.model(input_dict)
        return value.detach().cpu().numpy()

    def get_weights(self):
        pass
    
    def set_weights(self, weights):
        pass

    def train_actor_critic(self, input_dict):
        self.model.train()
        value_preds_batch = torch.FloatTensor(input_dict['old_values']).cuda()
        old_action_log_probs_batch = torch.FloatTensor(input_dict['old_logp_actions']).cuda()
        advantage = torch.FloatTensor(input_dict['advantages']).cuda()
        return_batch = torch.FloatTensor(input_dict['returns']).cuda()
        actions_batch = torch.FloatTensor(input_dict['actions']).cuda()
        obs_batch = torch.FloatTensor(input_dict['obs']).cuda()        
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        input_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'inputs' : obs_batch
        }
        action_log_probs, values, entropy = self.model(input_dict)

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - curr_e_clip,
                            1.0 + curr_e_clip) * advantage
        a_loss = torch.max(-surr1, -surr2).mean()
        values = torch.squeeze(values)
        if self.clip_value:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses,
                                         value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        c_loss = c_loss.mean()
        loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()

        return a_loss.item(), c_loss.item(), entropy.item(), kl, lr, lr_mul