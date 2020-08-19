from torch import nn
import torch

def critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    values = torch.squeeze(values)
    if clip_value:
        value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.max(value_losses,
                                         value_losses_clipped)
    else:
        c_loss = (return_batch - values)**2

    return c_loss


def actor_loss(old_action_log_probs_batch, action_log_probs, advantage, is_ppo, curr_e_clip):
    if is_ppo:
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip) * advantage
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_log_probs * advantage)
    
    return a_loss