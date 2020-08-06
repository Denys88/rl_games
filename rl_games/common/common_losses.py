from torch import nn


def critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value):
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