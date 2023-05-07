from torch import nn
import torch
import math


<<<<<<< HEAD
def critic_loss(value_preds_batch, values, curr_e_clip:float, return_batch, clip_value:float):
=======
def critic_loss(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    return default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    #return model.get_value_layer().loss(value_preds_batch=value_preds_batch, values=values, curr_e_clip=curr_e_clip, return_batch=return_batch, clip_value=clip_value)

def default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value):
>>>>>>> master
    if clip_value:
        value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.maximum(value_losses, value_losses_clipped)
    else:
        c_loss = (return_batch - values)**2

    return c_loss


def smooth_clamp(x, mi:float, mx:float):
    return 1.0/(1.0 + torch.exp((-(x-mi)/(mx-mi)+0.5)*4)) * (mx-mi) + mi


def smoothed_actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo:bool, curr_e_clip:float):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * smooth_clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = torch.maximum(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss


def actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo:bool, curr_e_clip:float):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.maximum(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss


def decoupled_actor_loss(behavior_action_neglog_probs, action_neglog_probs, proxy_neglog_probs, advantage, curr_e_clip:float):
    logratio = proxy_neglog_probs - action_neglog_probs
    #neglogp_adj = -torch.maximum(-behavior_action_neglog_probs, -action_neglog_probs.detach() - math.log(100))
    pg_losses1 = -advantage * torch.exp(behavior_action_neglog_probs - action_neglog_probs)

    clipped_logratio = torch.clamp(logratio, math.log(1.0 - curr_e_clip), math.log(1.0 + curr_e_clip))
    pg_losses2 = -advantage * torch.exp(clipped_logratio - proxy_neglog_probs + behavior_action_neglog_probs)
    pg_losses = torch.maximum(pg_losses1,pg_losses2)

    return pg_losses
