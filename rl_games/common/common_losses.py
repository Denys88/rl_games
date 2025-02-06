import torch
from torch import nn
import math


def critic_loss(model, value_preds_batch, values, curr_e_clip: float, return_batch, clip_value: float):
    """
    Computes the critic loss using the default critic loss function.
    """
    return default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)


def default_critic_loss(value_preds_batch, values, curr_e_clip: float, return_batch, clip_value: float):
    """
    Computes the default critic loss with optional clipping.
    """
    if clip_value:
        delta = values - value_preds_batch
        value_pred_clipped = value_preds_batch + delta.clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.max(value_losses, value_losses_clipped)
    else:
        c_loss = (return_batch - values)**2

    return c_loss


def smooth_clamp(x, mi: float, mx: float):
    """
    Smoothly clamps the input x between mi and mx.
    """
    return 1/(1 + torch.exp((-(x-mi)/(mx-mi)+0.5)*4)) * (mx-mi) + mi


def smoothed_actor_loss(
    old_action_neglog_probs_batch,
    action_neglog_probs,
    advantage,
    is_ppo: bool,
    curr_e_clip: float
):
    """
    Computes the smoothed actor loss for PPO.
    """
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * smooth_clamp(
            ratio,
            1.0 - curr_e_clip,
            1.0 + curr_e_clip
        )
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = action_neglog_probs * advantage

    return a_loss


def actor_loss(
    old_action_neglog_probs_batch,
    action_neglog_probs,
    advantage,
    is_ppo: bool,
    curr_e_clip: float
):
    """
    Computes the actor loss for PPO.
    """
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = action_neglog_probs * advantage

    return a_loss


def decoupled_actor_loss(
    behavior_action_neglog_probs,
    action_neglog_probs,
    proxy_neglog_probs,
    advantage,
    curr_e_clip: float
):
    """
    Computes the decoupled actor loss with clipping.
    """
    logratio = proxy_neglog_probs - action_neglog_probs
    pg_losses1 = -advantage * torch.exp(
        behavior_action_neglog_probs - action_neglog_probs
    )
    clipped_logratio = torch.clamp(
        logratio,
        math.log(1.0 - curr_e_clip),
        math.log(1.0 + curr_e_clip)
    )
    pg_losses2 = -advantage * torch.exp(
        clipped_logratio - proxy_neglog_probs + behavior_action_neglog_probs
    )
    pg_losses = torch.max(pg_losses1, pg_losses2)

    return pg_losses
