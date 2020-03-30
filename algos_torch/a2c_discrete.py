import common.a2c_common

class DiscreteA2CAgent(common.DiscreteA2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        common.DiscreteA2CBase.__init__(self, base_name, observation_space, action_space, config)
        SELF.opt = optim.Adam(model.parameters(), lr=0.001)

    def update_epoch(self):
        pass

    def save(self, fn):
        pass

    def restore(self, fn):
        pass

    def get_masked_action_values(self, obs, action_masks):
        pass

    def get_values(self, obs):
        pass

    def get_weights(self):
        pass
    
    def set_weights(self, weights):
        pass

    def train_actor_critic(self, input_dict):
        input_dict = {}
        value_preds_batch = torch.FloatTensor(input_dict['old_values']).cuda()
        old_action_log_probs_batch = torch.FloatTensor(input_dict['old_logp_actions']).cuda()
        advantage = torch.FloatTensor(input_dict['advantages']).cuda()
        return_batch = torch.FloatTensor(input_dict['rewards']).cuda()
        actions_batch = torch.FloatTensor(input_dict['actions']).cuda()
        obs_batch = torch.FloatTensor(input_dict['obs']).cuda()        
        #input_dict['masks']
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        values, action_log_probs, entropy, _ = self.network(obs_batch, actions_batch)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - curr_e_clip,
                            1.0 + curr_e_clip) * advantage
        a_loss = -torch.min(surr1, surr2).mean()

        if self.clip_value:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            c_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            c_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.optimizer.zero_grad()
        (c_loss * self.critic_coef + a_loss -
         entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        return a_loss.item(), c_loss.item(), entropy.item(), kl, lr, lr_mul