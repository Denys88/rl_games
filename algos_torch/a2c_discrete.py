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

    def train_actor_critic(self):
        input_dict = {}
        input_dict = {}
        input_dict['old_values'] = values[batch]
        old_action_log_probs_batch = torch.FloatTensor(input_dict['old_logp_actions']).cuda()
        input_dict['advantages'] = advantages[batch]
        input_dict['rewards'] = returns[batch]
        input_dict['actions'] = actions[batch]
        input_dict['obs'] = obses[batch]
        input_dict['masks'] = dones[batch]
        input_dict['learning_rate'] = self.last_lr

        values, action_log_probs, dist_entropy, _ = self.network(
            obs_batch, recurrent_hidden_states_batch, masks_batch,
            actions_batch)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()