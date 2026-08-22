"""Fused single-kernel PPO loss (continuous actions) with Triton.

Replaces the ~30 separate elementwise/reduction CUDA launches of the standard
PPO loss path (neglogp, ratio, clipped surrogate, clipped value loss, entropy,
bounds loss, masking, reductions and the policy KL) with a single forward
kernel launch plus a single analytic backward kernel launch.

Feature parity with A2CAgent.calc_losses / calc_gradients:
    - PPO clipped surrogate or plain policy-gradient loss (`ppo` flag)
    - hard clamp or smooth clamp of the ratio (`use_smooth_clamp`)
    - clipped or unclipped value loss (`clip_value`), multi-head values
    - 'bound' / 'regularisation' / disabled bounds loss
    - Gaussian entropy bonus
    - optional RNN masks with the exact `torch_ext.apply_masks` normalization
    - policy KL (torch_ext.policy_kl) computed in the same forward launch
    - value loss can be disabled (central value case, `has_value_loss`)

Gradient conventions (subgradients of max/clamp at ties/boundaries) follow
PyTorch exactly, so results match the eager implementation bit-for-bit up to
floating point reduction order.

Public entry point: fused_ppo_loss(...). Falls back to a numerically
identical pure-PyTorch implementation when Triton/CUDA is unavailable.

Disable via config `use_fused_ppo_kernel: False` or env RLG_NO_TRITON=1.
"""

import math

import torch

from rl_games.triton_config import USE_TRITON

_LOG2PI = math.log(2.0 * math.pi)

_BOUND_TYPES = {'none': 0, 'regularisation': 1, 'bound': 2}


def fused_ppo_loss_available(device) -> bool:
    return USE_TRITON and torch.device(device).type == 'cuda'


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if USE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _tie_grads(m1, m2):
        """Gradient split of torch.max(m1, m2): 1/0, or 0.5/0.5 on ties."""
        g1 = tl.where(m1 > m2, 1.0, tl.where(m1 == m2, 0.5, 0.0))
        g2 = tl.where(m2 > m1, 1.0, tl.where(m1 == m2, 0.5, 0.0))
        return g1, g2

    @triton.jit
    def _ppo_loss_fwd_kernel(
        mu_ptr, sigma_ptr, actions_ptr, old_neglogp_ptr, adv_ptr,
        old_mu_ptr, old_sigma_ptr,
        values_ptr, old_values_ptr, returns_ptr,
        mask_ptr, partials_ptr,
        e_clip, n_rows,
        A: tl.constexpr, A_PAD: tl.constexpr,
        V: tl.constexpr, V_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_PPO: tl.constexpr, USE_SMOOTH: tl.constexpr,
        CLIP_VALUE: tl.constexpr, BOUND_TYPE: tl.constexpr,
        HAS_VALUE_LOSS: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        LOG2PI = 1.8378770664093453
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        row_valid = rows < n_rows
        cols = tl.arange(0, A_PAD)
        m2d = row_valid[:, None] & (cols[None, :] < A)
        offs = rows[:, None] * A + cols[None, :]

        mu = tl.load(mu_ptr + offs, mask=m2d, other=0.0)
        sigma = tl.load(sigma_ptr + offs, mask=m2d, other=1.0)
        act = tl.load(actions_ptr + offs, mask=m2d, other=0.0)

        d = (act - mu) / sigma
        log_sigma = tl.where(m2d, tl.log(sigma), 0.0)
        sum_log_sigma = tl.sum(log_sigma, axis=1)
        neglogp = 0.5 * tl.sum(d * d, axis=1) + 0.5 * LOG2PI * A + sum_log_sigma

        old_nlp = tl.load(old_neglogp_ptr + rows, mask=row_valid, other=0.0)
        adv = tl.load(adv_ptr + rows, mask=row_valid, other=0.0)

        # actor loss
        if IS_PPO:
            ratio = tl.exp(old_nlp - neglogp)
            surr1 = adv * ratio
            lo = 1.0 - e_clip
            hi = 1.0 + e_clip
            if USE_SMOOTH:
                s = tl.sigmoid(((ratio - lo) / (hi - lo) - 0.5) * 4.0)
                clipped = s * (hi - lo) + lo
            else:
                clipped = tl.minimum(tl.maximum(ratio, lo), hi)
            surr2 = adv * clipped
            a_row = tl.maximum(-surr1, -surr2)
        else:
            a_row = neglogp * adv

        # entropy of diagonal Gaussian
        ent_row = sum_log_sigma + A * (0.5 + 0.5 * LOG2PI)

        # bounds loss
        if BOUND_TYPE == 1:
            b_row = tl.sum(tl.where(m2d, mu * mu, 0.0), axis=1)
        elif BOUND_TYPE == 2:
            mu_hi = tl.maximum(mu - 1.1, 0.0)
            mu_lo = tl.minimum(mu + 1.1, 0.0)
            b_row = tl.sum(tl.where(m2d, mu_hi * mu_hi + mu_lo * mu_lo, 0.0), axis=1)
        else:
            b_row = tl.zeros([BLOCK_N], dtype=tl.float32)

        # critic loss (summed over value heads)
        if HAS_VALUE_LOSS:
            vcols = tl.arange(0, V_PAD)
            vm2d = row_valid[:, None] & (vcols[None, :] < V)
            voffs = rows[:, None] * V + vcols[None, :]
            val = tl.load(values_ptr + voffs, mask=vm2d, other=0.0)
            old_val = tl.load(old_values_ptr + voffs, mask=vm2d, other=0.0)
            ret = tl.load(returns_ptr + voffs, mask=vm2d, other=0.0)
            if CLIP_VALUE:
                delta = val - old_val
                val_clipped = old_val + tl.minimum(tl.maximum(delta, -e_clip), e_clip)
                l1 = (val - ret) * (val - ret)
                l2 = (val_clipped - ret) * (val_clipped - ret)
                c_elem = tl.maximum(l1, l2)
            else:
                c_elem = (ret - val) * (ret - val)
            c_row = tl.sum(tl.where(vm2d, c_elem, 0.0), axis=1)
        else:
            c_row = tl.zeros([BLOCK_N], dtype=tl.float32)

        # policy KL vs old (mu, sigma): torch_ext.policy_kl formula
        old_mu = tl.load(old_mu_ptr + offs, mask=m2d, other=0.0)
        old_sig = tl.load(old_sigma_ptr + offs, mask=m2d, other=1.0)
        kl_c1 = tl.log(old_sig / sigma + 1e-5)
        kl_c2 = (sigma * sigma + (old_mu - mu) * (old_mu - mu)) / (2.0 * (old_sig * old_sig + 1e-5))
        kl_row = tl.sum(tl.where(m2d, kl_c1 + kl_c2 - 0.5, 0.0), axis=1)

        if HAS_MASK:
            w = tl.load(mask_ptr + rows, mask=row_valid, other=0.0)
        else:
            w = tl.where(row_valid, 1.0, 0.0)
        w = tl.where(row_valid, w, 0.0)

        tl.store(partials_ptr + pid * 5 + 0, tl.sum(a_row * w))
        tl.store(partials_ptr + pid * 5 + 1, tl.sum(c_row * w))
        tl.store(partials_ptr + pid * 5 + 2, tl.sum(ent_row * w))
        tl.store(partials_ptr + pid * 5 + 3, tl.sum(b_row * w))
        tl.store(partials_ptr + pid * 5 + 4, tl.sum(kl_row * w))

    @triton.jit
    def _ppo_loss_bwd_kernel(
        grad_out_ptr,
        mu_ptr, sigma_ptr, actions_ptr, old_neglogp_ptr, adv_ptr,
        values_ptr, old_values_ptr, returns_ptr,
        mask_ptr,
        grad_mu_ptr, grad_sigma_ptr, grad_values_ptr,
        e_clip, critic_coef, entropy_coef, bounds_coef,
        inv_n, inv_nv, n_rows,
        A: tl.constexpr, A_PAD: tl.constexpr,
        V: tl.constexpr, V_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_PPO: tl.constexpr, USE_SMOOTH: tl.constexpr,
        CLIP_VALUE: tl.constexpr, BOUND_TYPE: tl.constexpr,
        HAS_VALUE_LOSS: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        row_valid = rows < n_rows
        cols = tl.arange(0, A_PAD)
        m2d = row_valid[:, None] & (cols[None, :] < A)
        offs = rows[:, None] * A + cols[None, :]

        g = tl.load(grad_out_ptr)

        if HAS_MASK:
            w = tl.load(mask_ptr + rows, mask=row_valid, other=0.0)
        else:
            w = tl.where(row_valid, 1.0, 0.0)
        w = tl.where(row_valid, w, 0.0)
        # per-row weights of the mean/mask reduction
        wa = w * inv_n
        if HAS_MASK:
            wc = w * inv_n
        else:
            wc = w * inv_nv

        mu = tl.load(mu_ptr + offs, mask=m2d, other=0.0)
        sigma = tl.load(sigma_ptr + offs, mask=m2d, other=1.0)
        act = tl.load(actions_ptr + offs, mask=m2d, other=0.0)

        d = (act - mu) / sigma

        # d a_loss / d neglogp
        if IS_PPO:
            LOG2PI = 1.8378770664093453
            neglogp = (0.5 * tl.sum(d * d, axis=1) + 0.5 * LOG2PI * A
                       + tl.sum(tl.where(m2d, tl.log(sigma), 0.0), axis=1))
            old_nlp = tl.load(old_neglogp_ptr + rows, mask=row_valid, other=0.0)
            adv = tl.load(adv_ptr + rows, mask=row_valid, other=0.0)
            ratio = tl.exp(old_nlp - neglogp)
            surr1 = adv * ratio
            lo = 1.0 - e_clip
            hi = 1.0 + e_clip
            if USE_SMOOTH:
                s = tl.sigmoid(((ratio - lo) / (hi - lo) - 0.5) * 4.0)
                clipped = s * (hi - lo) + lo
                dclipped_dr = 4.0 * s * (1.0 - s)
            else:
                clipped = tl.minimum(tl.maximum(ratio, lo), hi)
                dclipped_dr = tl.where((ratio >= lo) & (ratio <= hi), 1.0, 0.0)
            surr2 = adv * clipped
            g1, g2 = _tie_grads(-surr1, -surr2)
            da_dr = g1 * (-adv) + g2 * (-adv) * dclipped_dr
            da_dnlp = da_dr * (-ratio)
        else:
            adv = tl.load(adv_ptr + rows, mask=row_valid, other=0.0)
            da_dnlp = adv

        grad_nlp = g * wa * da_dnlp  # [BLOCK_N]

        # neglogp gradients: dnlp/dmu = -d/sigma, dnlp/dsigma = (1 - d^2)/sigma
        grad_mu = grad_nlp[:, None] * (-d / sigma)
        grad_sigma = grad_nlp[:, None] * ((1.0 - d * d) / sigma)

        # entropy: loss -= entropy_coef * mean(ent); d ent/d sigma = 1/sigma
        grad_sigma += (-entropy_coef) * (g * wa)[:, None] / sigma

        # bounds loss
        if BOUND_TYPE == 1:
            grad_mu += bounds_coef * (g * wa)[:, None] * 2.0 * mu
        elif BOUND_TYPE == 2:
            db_dmu = 2.0 * tl.maximum(mu - 1.1, 0.0) + 2.0 * tl.minimum(mu + 1.1, 0.0)
            grad_mu += bounds_coef * (g * wa)[:, None] * db_dmu

        tl.store(grad_mu_ptr + offs, grad_mu, mask=m2d)
        tl.store(grad_sigma_ptr + offs, grad_sigma, mask=m2d)

        # critic loss: loss += 0.5 * critic_coef * mean(c)
        vcols = tl.arange(0, V_PAD)
        vm2d = row_valid[:, None] & (vcols[None, :] < V)
        voffs = rows[:, None] * V + vcols[None, :]
        if HAS_VALUE_LOSS:
            val = tl.load(values_ptr + voffs, mask=vm2d, other=0.0)
            old_val = tl.load(old_values_ptr + voffs, mask=vm2d, other=0.0)
            ret = tl.load(returns_ptr + voffs, mask=vm2d, other=0.0)
            if CLIP_VALUE:
                delta = val - old_val
                inside = tl.where((delta >= -e_clip) & (delta <= e_clip), 1.0, 0.0)
                val_clipped = old_val + tl.minimum(tl.maximum(delta, -e_clip), e_clip)
                l1 = (val - ret) * (val - ret)
                l2 = (val_clipped - ret) * (val_clipped - ret)
                g1c, g2c = _tie_grads(l1, l2)
                dc_dv = g1c * 2.0 * (val - ret) + g2c * 2.0 * (val_clipped - ret) * inside
            else:
                dc_dv = 2.0 * (val - ret)
            grad_val = (0.5 * critic_coef) * (g * wc)[:, None] * dc_dv
        else:
            grad_val = tl.zeros([BLOCK_N, V_PAD], dtype=tl.float32)
        tl.store(grad_values_ptr + voffs, grad_val, mask=vm2d)


    @triton.jit
    def _ppo_loss_discrete_fwd_kernel(
        neglogp_ptr, entropy_ptr, old_neglogp_ptr, adv_ptr,
        values_ptr, old_values_ptr, returns_ptr,
        mask_ptr, partials_ptr,
        e_clip, n_rows,
        V: tl.constexpr, V_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_PPO: tl.constexpr, USE_SMOOTH: tl.constexpr,
        CLIP_VALUE: tl.constexpr,
        HAS_VALUE_LOSS: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        row_valid = rows < n_rows

        neglogp = tl.load(neglogp_ptr + rows, mask=row_valid, other=0.0)
        ent_row = tl.load(entropy_ptr + rows, mask=row_valid, other=0.0)
        old_nlp = tl.load(old_neglogp_ptr + rows, mask=row_valid, other=0.0)
        adv = tl.load(adv_ptr + rows, mask=row_valid, other=0.0)

        if IS_PPO:
            ratio = tl.exp(old_nlp - neglogp)
            surr1 = adv * ratio
            lo = 1.0 - e_clip
            hi = 1.0 + e_clip
            if USE_SMOOTH:
                s = tl.sigmoid(((ratio - lo) / (hi - lo) - 0.5) * 4.0)
                clipped = s * (hi - lo) + lo
            else:
                clipped = tl.minimum(tl.maximum(ratio, lo), hi)
            surr2 = adv * clipped
            a_row = tl.maximum(-surr1, -surr2)
        else:
            a_row = neglogp * adv

        if HAS_VALUE_LOSS:
            vcols = tl.arange(0, V_PAD)
            vm2d = row_valid[:, None] & (vcols[None, :] < V)
            voffs = rows[:, None] * V + vcols[None, :]
            val = tl.load(values_ptr + voffs, mask=vm2d, other=0.0)
            old_val = tl.load(old_values_ptr + voffs, mask=vm2d, other=0.0)
            ret = tl.load(returns_ptr + voffs, mask=vm2d, other=0.0)
            if CLIP_VALUE:
                delta = val - old_val
                val_clipped = old_val + tl.minimum(tl.maximum(delta, -e_clip), e_clip)
                l1 = (val - ret) * (val - ret)
                l2 = (val_clipped - ret) * (val_clipped - ret)
                c_elem = tl.maximum(l1, l2)
            else:
                c_elem = (ret - val) * (ret - val)
            c_row = tl.sum(tl.where(vm2d, c_elem, 0.0), axis=1)
        else:
            c_row = tl.zeros([BLOCK_N], dtype=tl.float32)

        # discrete kl proxy: 0.5 * (old_neglogp - neglogp)^2
        kl_row = 0.5 * (old_nlp - neglogp) * (old_nlp - neglogp)

        if HAS_MASK:
            w = tl.load(mask_ptr + rows, mask=row_valid, other=0.0)
        else:
            w = tl.where(row_valid, 1.0, 0.0)
        w = tl.where(row_valid, w, 0.0)

        tl.store(partials_ptr + pid * 4 + 0, tl.sum(a_row * w))
        tl.store(partials_ptr + pid * 4 + 1, tl.sum(c_row * w))
        tl.store(partials_ptr + pid * 4 + 2, tl.sum(ent_row * w))
        tl.store(partials_ptr + pid * 4 + 3, tl.sum(kl_row * w))

    @triton.jit
    def _ppo_loss_discrete_bwd_kernel(
        grad_out_ptr,
        neglogp_ptr, old_neglogp_ptr, adv_ptr,
        values_ptr, old_values_ptr, returns_ptr,
        mask_ptr,
        grad_neglogp_ptr, grad_entropy_ptr, grad_values_ptr,
        e_clip, critic_coef, entropy_coef,
        inv_n, inv_nv, n_rows,
        V: tl.constexpr, V_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_PPO: tl.constexpr, USE_SMOOTH: tl.constexpr,
        CLIP_VALUE: tl.constexpr,
        HAS_VALUE_LOSS: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        row_valid = rows < n_rows

        g = tl.load(grad_out_ptr)

        if HAS_MASK:
            w = tl.load(mask_ptr + rows, mask=row_valid, other=0.0)
        else:
            w = tl.where(row_valid, 1.0, 0.0)
        w = tl.where(row_valid, w, 0.0)
        wa = w * inv_n
        if HAS_MASK:
            wc = w * inv_n
        else:
            wc = w * inv_nv

        neglogp = tl.load(neglogp_ptr + rows, mask=row_valid, other=0.0)
        old_nlp = tl.load(old_neglogp_ptr + rows, mask=row_valid, other=0.0)
        adv = tl.load(adv_ptr + rows, mask=row_valid, other=0.0)

        if IS_PPO:
            ratio = tl.exp(old_nlp - neglogp)
            surr1 = adv * ratio
            lo = 1.0 - e_clip
            hi = 1.0 + e_clip
            if USE_SMOOTH:
                s = tl.sigmoid(((ratio - lo) / (hi - lo) - 0.5) * 4.0)
                clipped = s * (hi - lo) + lo
                dclipped_dr = 4.0 * s * (1.0 - s)
            else:
                clipped = tl.minimum(tl.maximum(ratio, lo), hi)
                dclipped_dr = tl.where((ratio >= lo) & (ratio <= hi), 1.0, 0.0)
            surr2 = adv * clipped
            g1, g2 = _tie_grads(-surr1, -surr2)
            da_dr = g1 * (-adv) + g2 * (-adv) * dclipped_dr
            da_dnlp = da_dr * (-ratio)
        else:
            da_dnlp = adv

        tl.store(grad_neglogp_ptr + rows, g * wa * da_dnlp, mask=row_valid)
        tl.store(grad_entropy_ptr + rows, (-entropy_coef) * g * wa, mask=row_valid)

        vcols = tl.arange(0, V_PAD)
        vm2d = row_valid[:, None] & (vcols[None, :] < V)
        voffs = rows[:, None] * V + vcols[None, :]
        if HAS_VALUE_LOSS:
            val = tl.load(values_ptr + voffs, mask=vm2d, other=0.0)
            old_val = tl.load(old_values_ptr + voffs, mask=vm2d, other=0.0)
            ret = tl.load(returns_ptr + voffs, mask=vm2d, other=0.0)
            if CLIP_VALUE:
                delta = val - old_val
                inside = tl.where((delta >= -e_clip) & (delta <= e_clip), 1.0, 0.0)
                val_clipped = old_val + tl.minimum(tl.maximum(delta, -e_clip), e_clip)
                l1 = (val - ret) * (val - ret)
                l2 = (val_clipped - ret) * (val_clipped - ret)
                g1c, g2c = _tie_grads(l1, l2)
                dc_dv = g1c * 2.0 * (val - ret) + g2c * 2.0 * (val_clipped - ret) * inside
            else:
                dc_dv = 2.0 * (val - ret)
            grad_val = (0.5 * critic_coef) * (g * wc)[:, None] * dc_dv
        else:
            grad_val = tl.zeros([BLOCK_N, V_PAD], dtype=tl.float32)
        tl.store(grad_values_ptr + voffs, grad_val, mask=vm2d)


# ---------------------------------------------------------------------------
# Reference PyTorch implementation (fallback + testing)
# ---------------------------------------------------------------------------

def _torch_ppo_loss(mu, sigma, values, actions, old_neglogp, advantage,
                    old_mu, old_sigma, old_values, returns, rnn_masks,
                    e_clip, critic_coef, entropy_coef, bounds_coef,
                    is_ppo, use_smooth_clamp, clip_value, bound_loss_type,
                    has_value_loss):
    """Numerically identical to A2CAgent.calc_losses + torch_ext.policy_kl."""
    from rl_games.common import common_losses
    from rl_games.algos_torch import torch_ext

    neglogp = (0.5 * (((actions - mu) / sigma) ** 2).sum(dim=-1)
               + 0.5 * _LOG2PI * actions.size(-1)
               + torch.log(sigma).sum(dim=-1))
    entropy = (0.5 + 0.5 * _LOG2PI + torch.log(sigma)).sum(dim=-1)

    loss_func = common_losses.smoothed_actor_loss if use_smooth_clamp else common_losses.actor_loss
    a_loss = loss_func(old_neglogp, neglogp, advantage, is_ppo, e_clip)
    if has_value_loss:
        c_loss = common_losses.default_critic_loss(old_values, values, e_clip, returns, clip_value)
    else:
        c_loss = torch.zeros(1, device=values.device)
    if bound_loss_type == 'regularisation':
        b_loss = (mu * mu).sum(dim=-1)
    elif bound_loss_type == 'bound':
        soft_bound = 1.1
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
        b_loss = (mu_loss_low + mu_loss_high).sum(dim=-1)
    else:
        b_loss = torch.zeros(mu.size(0), device=mu.device)

    losses, sum_mask = torch_ext.apply_masks(
        [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
    a_loss, c_loss, entropy, b_loss = losses

    loss = a_loss + 0.5 * c_loss * critic_coef - entropy * entropy_coef + b_loss * bounds_coef

    with torch.no_grad():
        kl = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma, rnn_masks is None)
        if rnn_masks is not None:
            kl = (kl * rnn_masks).sum() / rnn_masks.numel()

    return loss, a_loss.detach(), c_loss.detach(), entropy.detach(), b_loss.detach(), kl, sum_mask


def _torch_ppo_loss_discrete(neglogp, entropy, values, old_neglogp, advantage,
                             old_values, returns, rnn_masks,
                             e_clip, critic_coef, entropy_coef,
                             is_ppo, use_smooth_clamp, clip_value, has_value_loss):
    """Numerically identical to DiscreteA2CAgent.calc_gradients loss section."""
    from rl_games.common import common_losses
    from rl_games.algos_torch import torch_ext

    loss_func = common_losses.smoothed_actor_loss if use_smooth_clamp else common_losses.actor_loss
    a_loss = loss_func(old_neglogp, neglogp, advantage, is_ppo, e_clip)
    if has_value_loss:
        c_loss = common_losses.default_critic_loss(old_values, values, e_clip, returns, clip_value)
    else:
        c_loss = torch.zeros(1, device=values.device)

    losses, sum_mask = torch_ext.apply_masks(
        [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1)], rnn_masks)
    a_loss, c_loss, entropy_s = losses
    loss = a_loss + 0.5 * c_loss * critic_coef - entropy_s * entropy_coef

    with torch.no_grad():
        kl = 0.5 * ((old_neglogp - neglogp) ** 2)
        if rnn_masks is not None:
            kl = (kl * rnn_masks).sum() / rnn_masks.numel()
        else:
            kl = kl.mean()

    return loss, a_loss.detach(), c_loss.detach(), entropy_s.detach(), kl, sum_mask


# ---------------------------------------------------------------------------
# Autograd binding
# ---------------------------------------------------------------------------

class _FusedPPOLoss(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, mu, sigma, values, actions, old_neglogp, advantage,
                old_mu, old_sigma, old_values, returns, rnn_masks,
                e_clip, critic_coef, entropy_coef, bounds_coef,
                is_ppo, use_smooth_clamp, clip_value, bound_type_id,
                has_value_loss):
        n, num_actions = mu.shape
        value_size = values.shape[1]

        mu = mu.contiguous()
        sigma = sigma.contiguous()
        values = values.contiguous()
        actions = actions.contiguous()
        old_neglogp = old_neglogp.contiguous()
        advantage = advantage.contiguous()
        old_values = old_values.contiguous()
        returns = returns.contiguous()
        has_mask = rnn_masks is not None
        if has_mask:
            rnn_masks = rnn_masks.contiguous().float()

        a_pad = triton.next_power_of_2(num_actions)
        v_pad = triton.next_power_of_2(value_size)
        block_n = 128
        grid = (triton.cdiv(n, block_n),)

        partials = torch.empty((grid[0], 5), dtype=torch.float32, device=mu.device)

        _ppo_loss_fwd_kernel[grid](
            mu, sigma, actions, old_neglogp, advantage,
            old_mu.contiguous(), old_sigma.contiguous(),
            values, old_values, returns,
            rnn_masks if has_mask else mu,  # dummy ptr when unused
            partials,
            e_clip, n,
            num_actions, a_pad, value_size, v_pad, block_n,
            is_ppo, use_smooth_clamp, clip_value, bound_type_id,
            has_value_loss, has_mask,
        )

        sums = partials.sum(dim=0)
        inv_n = 1.0 / n
        inv_nv = 1.0 / (n * value_size)
        a_loss = sums[0] * inv_n
        c_loss = sums[1] * (inv_n if has_mask else inv_nv)
        if not has_value_loss and not has_mask:
            c_loss = sums[1] * 0.0  # torch path: mean(zeros(1)) == 0
        entropy = sums[2] * inv_n
        b_loss = sums[3] * inv_n
        kl = sums[4] * inv_n

        loss = a_loss + 0.5 * c_loss * critic_coef - entropy * entropy_coef + b_loss * bounds_coef

        ctx.save_for_backward(mu, sigma, values, actions, old_neglogp, advantage,
                              old_values, returns,
                              rnn_masks if has_mask else torch.empty(0))
        ctx.cfg = (e_clip, critic_coef, entropy_coef, bounds_coef,
                   is_ppo, use_smooth_clamp, clip_value, bound_type_id,
                   has_value_loss, has_mask, num_actions, a_pad,
                   value_size, v_pad, block_n, n)
        ctx.mark_non_differentiable(a_loss, c_loss, entropy, b_loss, kl)
        return loss, a_loss, c_loss, entropy, b_loss, kl

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_loss, *_):
        (mu, sigma, values, actions, old_neglogp, advantage,
         old_values, returns, rnn_masks) = ctx.saved_tensors
        (e_clip, critic_coef, entropy_coef, bounds_coef,
         is_ppo, use_smooth_clamp, clip_value, bound_type_id,
         has_value_loss, has_mask, num_actions, a_pad,
         value_size, v_pad, block_n, n) = ctx.cfg

        grad_mu = torch.empty_like(mu)
        grad_sigma = torch.empty_like(sigma)
        grad_values = torch.empty_like(values)

        grid = (triton.cdiv(n, block_n),)
        _ppo_loss_bwd_kernel[grid](
            grad_loss.contiguous(),
            mu, sigma, actions, old_neglogp, advantage,
            values, old_values, returns,
            rnn_masks if has_mask else mu,  # dummy ptr when unused
            grad_mu, grad_sigma, grad_values,
            e_clip, critic_coef, entropy_coef, bounds_coef,
            1.0 / n, 1.0 / (n * value_size), n,
            num_actions, a_pad, value_size, v_pad, block_n,
            is_ppo, use_smooth_clamp, clip_value, bound_type_id,
            has_value_loss, has_mask,
        )

        return (grad_mu, grad_sigma, grad_values) + (None,) * 17


class _FusedPPOLossDiscrete(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, neglogp, entropy, values, old_neglogp, advantage,
                old_values, returns, rnn_masks,
                e_clip, critic_coef, entropy_coef,
                is_ppo, use_smooth_clamp, clip_value, has_value_loss):
        n = neglogp.shape[0]
        value_size = values.shape[1]

        neglogp = neglogp.contiguous()
        entropy = entropy.contiguous()
        values = values.contiguous()
        old_neglogp = old_neglogp.contiguous()
        advantage = advantage.contiguous()
        old_values = old_values.contiguous()
        returns = returns.contiguous()
        has_mask = rnn_masks is not None
        if has_mask:
            rnn_masks = rnn_masks.contiguous().float()

        v_pad = triton.next_power_of_2(value_size)
        block_n = 256
        grid = (triton.cdiv(n, block_n),)
        partials = torch.empty((grid[0], 4), dtype=torch.float32, device=values.device)

        _ppo_loss_discrete_fwd_kernel[grid](
            neglogp, entropy, old_neglogp, advantage,
            values, old_values, returns,
            rnn_masks if has_mask else neglogp,  # dummy ptr when unused
            partials,
            e_clip, n,
            value_size, v_pad, block_n,
            is_ppo, use_smooth_clamp, clip_value,
            has_value_loss, has_mask,
        )

        sums = partials.sum(dim=0)
        inv_n = 1.0 / n
        inv_nv = 1.0 / (n * value_size)
        a_loss = sums[0] * inv_n
        c_loss = sums[1] * (inv_n if has_mask else inv_nv)
        entropy_s = sums[2] * inv_n
        kl = sums[3] * inv_n

        loss = a_loss + 0.5 * c_loss * critic_coef - entropy_s * entropy_coef

        ctx.save_for_backward(neglogp, values, old_neglogp, advantage,
                              old_values, returns,
                              rnn_masks if has_mask else torch.empty(0))
        ctx.cfg = (e_clip, critic_coef, entropy_coef,
                   is_ppo, use_smooth_clamp, clip_value,
                   has_value_loss, has_mask, value_size, v_pad, block_n, n)
        ctx.mark_non_differentiable(a_loss, c_loss, entropy_s, kl)
        return loss, a_loss, c_loss, entropy_s, kl

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_loss, *_):
        (neglogp, values, old_neglogp, advantage,
         old_values, returns, rnn_masks) = ctx.saved_tensors
        (e_clip, critic_coef, entropy_coef,
         is_ppo, use_smooth_clamp, clip_value,
         has_value_loss, has_mask, value_size, v_pad, block_n, n) = ctx.cfg

        grad_neglogp = torch.empty_like(neglogp)
        grad_entropy = torch.empty_like(neglogp)
        grad_values = torch.empty_like(values)

        grid = (triton.cdiv(n, block_n),)
        _ppo_loss_discrete_bwd_kernel[grid](
            grad_loss.contiguous(),
            neglogp, old_neglogp, advantage,
            values, old_values, returns,
            rnn_masks if has_mask else neglogp,  # dummy ptr when unused
            grad_neglogp, grad_entropy, grad_values,
            e_clip, critic_coef, entropy_coef,
            1.0 / n, 1.0 / (n * value_size), n,
            value_size, v_pad, block_n,
            is_ppo, use_smooth_clamp, clip_value,
            has_value_loss, has_mask,
        )

        return (grad_neglogp, grad_entropy, grad_values) + (None,) * 12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_ppo_loss(mu, sigma, values, actions, old_neglogp, advantage,
                   old_mu, old_sigma, old_values, returns, rnn_masks,
                   e_clip, critic_coef, entropy_coef, bounds_coef,
                   is_ppo=True, use_smooth_clamp=False, clip_value=True,
                   bound_loss_type='none', has_value_loss=True):
    """Fused PPO loss: total loss + logging scalars + policy KL.

    Args:
        mu, sigma: [N, A] current policy statistics (grad).
        values: [N, V] current value predictions (grad).
        actions: [N, A] taken actions.
        old_neglogp, advantage: [N].
        old_mu, old_sigma: [N, A] rollout policy statistics (for KL, no grad).
        old_values, returns: [N, V].
        rnn_masks: [N] float mask or None.
        bounds_coef: pass 0.0 when disabled (bound_loss_type 'none').

    Returns:
        (loss, a_loss, c_loss, entropy, b_loss, kl_dist, sum_mask) — loss
        carries gradients to mu/sigma/values; the rest are detached scalars.
        sum_mask mirrors torch_ext.apply_masks (mask.numel() or None).
    """
    if bounds_coef is None:
        bounds_coef = 0.0
        bound_loss_type = 'none'

    if not (USE_TRITON and mu.is_cuda):
        return _torch_ppo_loss(
            mu, sigma, values, actions, old_neglogp, advantage,
            old_mu, old_sigma, old_values, returns, rnn_masks,
            e_clip, critic_coef, entropy_coef, bounds_coef,
            is_ppo, use_smooth_clamp, clip_value, bound_loss_type,
            has_value_loss)

    loss, a_loss, c_loss, entropy, b_loss, kl = _FusedPPOLoss.apply(
        mu, sigma, values, actions.detach(), old_neglogp.detach(),
        advantage.detach(), old_mu.detach(), old_sigma.detach(),
        old_values.detach(), returns.detach(), rnn_masks,
        float(e_clip), float(critic_coef), float(entropy_coef), float(bounds_coef),
        bool(is_ppo), bool(use_smooth_clamp), bool(clip_value),
        _BOUND_TYPES[bound_loss_type], bool(has_value_loss))

    sum_mask = rnn_masks.numel() if rnn_masks is not None else None
    return loss, a_loss, c_loss, entropy, b_loss, kl, sum_mask


def fused_ppo_loss_discrete(neglogp, entropy, values, old_neglogp, advantage,
                            old_values, returns, rnn_masks,
                            e_clip, critic_coef, entropy_coef,
                            is_ppo=True, use_smooth_clamp=False, clip_value=True,
                            has_value_loss=True):
    """Fused PPO loss for discrete/multi-discrete policies.

    The categorical neglogp/entropy stay inside the model (they depend on the
    action-space structure and masking); this kernel fuses everything after
    them: clipped surrogate, clipped value loss, entropy weighting, masking,
    reductions and the discrete KL proxy 0.5*(old_nlp - nlp)^2.

    Args:
        neglogp, entropy: [N] current policy outputs (grad).
        values: [N, V] current value predictions (grad).
        old_neglogp, advantage: [N]; old_values, returns: [N, V].
        rnn_masks: [N] float mask or None.

    Returns:
        (loss, a_loss, c_loss, entropy, kl_dist, sum_mask) — loss carries
        gradients to neglogp/entropy/values; the rest are detached scalars.
    """
    if not (USE_TRITON and values.is_cuda):
        return _torch_ppo_loss_discrete(
            neglogp, entropy, values, old_neglogp, advantage,
            old_values, returns, rnn_masks,
            e_clip, critic_coef, entropy_coef,
            is_ppo, use_smooth_clamp, clip_value, has_value_loss)

    loss, a_loss, c_loss, entropy_s, kl = _FusedPPOLossDiscrete.apply(
        neglogp, entropy, values, old_neglogp.detach(), advantage.detach(),
        old_values.detach(), returns.detach(), rnn_masks,
        float(e_clip), float(critic_coef), float(entropy_coef),
        bool(is_ppo), bool(use_smooth_clamp), bool(clip_value),
        bool(has_value_loss))

    sum_mask = rnn_masks.numel() if rnn_masks is not None else None
    return loss, a_loss, c_loss, entropy_s, kl, sum_mask
