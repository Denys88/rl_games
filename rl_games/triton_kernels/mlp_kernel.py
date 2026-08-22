"""EXPERIMENTAL: fused actor-critic MLP (trunk + mu/value heads) in single
Triton forward / single backward kernels.

Not wired into the agents yet. Together with ppo_loss_kernel this makes the
whole PPO minibatch update (network fwd + loss fwd/bwd + network bwd,
optimizer excluded) a 4-kernel-launch pipeline. See docs/FUSED_PPO_KERNEL.md
for measured results and integration constraints (plain MLP nets, fixed
sigma, obs normalization / weight layout handling).

Network (matches rl_games A2CBuilder, separate=False, fixed_sigma=True):
    h1 = ELU(x @ W1 + b1)      # D_IN -> H1
    h2 = ELU(h1 @ W2 + b2)     # H1 -> H2
    h3 = ELU(h2 @ W3 + b3)     # H2 -> H3
    mu = h3 @ Wm + bm          # H3 -> A
    v  = h3 @ Wv + bv          # H3 -> V
    sigma = exp(logstd)        # [A] parameter, broadcast

Weights are stored [in, out] (transposed from nn.Linear). Activations h1..h3
are written to scratch in forward and reused in backward. Weight/bias grads
are accumulated with fp32 atomics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_p(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _elu(x):
    return tl.where(x > 0, x, tl.exp(x) - 1.0)


@triton.jit
def _elu_grad_from_h(h):
    # ELU output h: h>0 -> grad 1 ; else h = exp(z)-1 -> grad exp(z) = h+1
    return tl.where(h > 0, 1.0, h + 1.0)


@triton.jit
def _linear(x_ptr, w_ptr, b_ptr, rows, row_valid, acc_dtype,
            K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_K: tl.constexpr,
            N: tl.constexpr, BLOCK_M: tl.constexpr, IEEE: tl.constexpr):
    """y[rows, :N] = x[rows, :K] @ W[K, N] + b, K-chunked tl.dot."""
    acc = tl.zeros([BLOCK_M, N], dtype=tl.float32)
    for k0 in range(0, K_PAD, BLOCK_K):
        kcols = k0 + tl.arange(0, BLOCK_K)
        kmask = row_valid[:, None] & (kcols[None, :] < K)
        xk = tl.load(x_ptr + rows[:, None] * K + kcols[None, :], mask=kmask, other=0.0)
        wmask = kcols[:, None] < K
        ncols = tl.arange(0, N)
        wk = tl.load(w_ptr + kcols[:, None] * N + ncols[None, :], mask=wmask, other=0.0)
        if IEEE:
            acc = tl.dot(xk, wk, acc, input_precision="ieee")
        else:
            acc = tl.dot(xk, wk, acc)
    b = tl.load(b_ptr + tl.arange(0, N))
    return acc + b[None, :]


@triton.jit
def _ac_mlp_fwd_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
    wm_ptr, bm_ptr, wv_ptr, bv_ptr,
    h1_ptr, h2_ptr, h3_ptr, mu_ptr, v_ptr,
    n_rows,
    D: tl.constexpr, D_PAD: tl.constexpr,
    H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
    A: tl.constexpr, A_PAD: tl.constexpr, V: tl.constexpr, V_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_valid = rows < n_rows

    # layer 1: x [., D] -> h1 [., H1]
    z1 = _linear(x_ptr, w1_ptr, b1_ptr, rows, row_valid, tl.float32,
                 D, D_PAD, BLOCK_K, H1, BLOCK_M, IEEE)
    h1 = _elu(z1)
    c1 = tl.arange(0, H1)
    tl.store(h1_ptr + rows[:, None] * H1 + c1[None, :], h1, mask=row_valid[:, None])
    tl.debug_barrier()  # h1 scratch is re-read chunk-wise by all warps below

    # layer 2: h1 [., H1] -> h2 [., H2]
    z2 = _linear(h1_ptr, w2_ptr, b2_ptr, rows, row_valid, tl.float32,
                 H1, H1, BLOCK_K, H2, BLOCK_M, IEEE)
    h2 = _elu(z2)
    c2 = tl.arange(0, H2)
    tl.store(h2_ptr + rows[:, None] * H2 + c2[None, :], h2, mask=row_valid[:, None])
    tl.debug_barrier()  # h2 scratch is re-read chunk-wise by all warps below

    # layer 3
    z3 = _linear(h2_ptr, w3_ptr, b3_ptr, rows, row_valid, tl.float32,
                 H2, H2, BLOCK_K, H3, BLOCK_M, IEEE)
    h3 = _elu(z3)
    c3 = tl.arange(0, H3)
    tl.store(h3_ptr + rows[:, None] * H3 + c3[None, :], h3, mask=row_valid[:, None])

    # heads (K = H3 in one dot)
    am = tl.arange(0, A_PAD)
    wm = tl.load(wm_ptr + c3[:, None] * A_PAD + am[None, :])
    if IEEE:
        mu = tl.dot(h3, wm, input_precision="ieee")
    else:
        mu = tl.dot(h3, wm)
    bm = tl.load(bm_ptr + am)
    mu = mu + bm[None, :]
    tl.store(mu_ptr + rows[:, None] * A + am[None, :], mu,
             mask=row_valid[:, None] & (am[None, :] < A))

    vm = tl.arange(0, V_PAD)
    wv = tl.load(wv_ptr + c3[:, None] * V_PAD + vm[None, :])
    if IEEE:
        v = tl.dot(h3, wv, input_precision="ieee")
    else:
        v = tl.dot(h3, wv)
    bv = tl.load(bv_ptr + vm)
    v = v + bv[None, :]
    tl.store(v_ptr + rows[:, None] * V + vm[None, :], v,
             mask=row_valid[:, None] & (vm[None, :] < V))


@triton.jit
def _ac_mlp_bwd_kernel(
    x_ptr, h1_ptr, h2_ptr, h3_ptr,
    w2_ptr, w3_ptr, wm_ptr, wv_ptr,
    dmu_ptr, dv_ptr, dsigma_ptr,
    dw1_ptr, db1_ptr, dw2_ptr, db2_ptr, dw3_ptr, db3_ptr,
    dwm_ptr, dbm_ptr, dwv_ptr, dbv_ptr, dlogstd_ptr,
    n_rows,
    D: tl.constexpr, D_PAD: tl.constexpr,
    H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
    A: tl.constexpr, A_PAD: tl.constexpr, V: tl.constexpr, V_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, IEEE: tl.constexpr, HAS_SIGMA: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_valid = rows < n_rows

    am = tl.arange(0, A_PAD)
    vm = tl.arange(0, V_PAD)
    c1 = tl.arange(0, H1)
    c2 = tl.arange(0, H2)
    c3 = tl.arange(0, H3)

    dmu = tl.load(dmu_ptr + rows[:, None] * A + am[None, :],
                  mask=row_valid[:, None] & (am[None, :] < A), other=0.0)
    dv = tl.load(dv_ptr + rows[:, None] * V + vm[None, :],
                 mask=row_valid[:, None] & (vm[None, :] < V), other=0.0)

    # logstd grad (fixed sigma): sum over batch of dsigma * sigma is handled
    # outside; here dsigma is already d loss/d logstd per-row if provided.
    if HAS_SIGMA:
        dsig = tl.load(dsigma_ptr + rows[:, None] * A + am[None, :],
                       mask=row_valid[:, None] & (am[None, :] < A), other=0.0)
        tl.atomic_add(dlogstd_ptr + am, tl.sum(dsig, axis=0), mask=am < A)

    h3 = tl.load(h3_ptr + rows[:, None] * H3 + c3[None, :],
                 mask=row_valid[:, None], other=0.0)

    # head grads
    tl.atomic_add(dwm_ptr + c3[:, None] * A_PAD + am[None, :],
                  _dot_p(tl.trans(h3), dmu, IEEE), mask=(am[None, :] < A))
    tl.atomic_add(dbm_ptr + am, tl.sum(dmu, axis=0), mask=am < A)
    tl.atomic_add(dwv_ptr + c3[:, None] * V_PAD + vm[None, :],
                  _dot_p(tl.trans(h3), dv, IEEE), mask=(vm[None, :] < V))
    tl.atomic_add(dbv_ptr + vm, tl.sum(dv, axis=0), mask=vm < V)

    # dh3 = dmu @ Wm^T + dv @ Wv^T   (W stored [in, out] -> W^T via trans load)
    wmT = tl.load(wm_ptr + c3[None, :] * A_PAD + am[:, None])  # [A_PAD, H3]
    wvT = tl.load(wv_ptr + c3[None, :] * V_PAD + vm[:, None])  # [V_PAD, H3]
    if IEEE:
        dh3 = tl.dot(dmu, wmT, input_precision="ieee") + tl.dot(dv, wvT, input_precision="ieee")
    else:
        dh3 = tl.dot(dmu, wmT) + tl.dot(dv, wvT)
    dz3 = dh3 * _elu_grad_from_h(h3)

    # layer 3 grads
    h2 = tl.load(h2_ptr + rows[:, None] * H2 + c2[None, :],
                 mask=row_valid[:, None], other=0.0)
    tl.atomic_add(dw3_ptr + c2[:, None] * H3 + c3[None, :], _dot_p(tl.trans(h2), dz3, IEEE))
    tl.atomic_add(db3_ptr + c3, tl.sum(dz3, axis=0))

    w3T = tl.load(w3_ptr + c2[None, :] * H3 + c3[:, None])  # [H3, H2]
    if IEEE:
        dh2 = tl.dot(dz3, w3T, input_precision="ieee")
    else:
        dh2 = tl.dot(dz3, w3T)
    dz2 = dh2 * _elu_grad_from_h(h2)

    # layer 2 + layer 1 grads, chunked over H1 to bound tile sizes
    dcols = tl.arange(0, D_PAD)
    x = tl.load(x_ptr + rows[:, None] * D + dcols[None, :],
                mask=row_valid[:, None] & (dcols[None, :] < D), other=0.0)
    xT = tl.trans(x)
    hcols = tl.arange(0, BLOCK_H)
    for k0 in range(0, H1, BLOCK_H):
        kc = k0 + hcols
        h1_k = tl.load(h1_ptr + rows[:, None] * H1 + kc[None, :],
                       mask=row_valid[:, None], other=0.0)
        # dW2 rows [k0:k0+BLOCK_H] = h1_k^T @ dz2
        tl.atomic_add(dw2_ptr + kc[:, None] * H2 + c2[None, :],
                      _dot_p(tl.trans(h1_k), dz2, IEEE))
        # dh1[:, k0:k0+BLOCK_H] = dz2 @ (W2[k0:k0+BLOCK_H, :])^T
        w2_kT = tl.load(w2_ptr + kc[None, :] * H2 + c2[:, None])  # [H2, BLOCK_H]
        if IEEE:
            dh1_k = tl.dot(dz2, w2_kT, input_precision="ieee")
        else:
            dh1_k = tl.dot(dz2, w2_kT)
        dz1_k = dh1_k * _elu_grad_from_h(h1_k)
        tl.atomic_add(db1_ptr + kc, tl.sum(dz1_k, axis=0))
        # dW1 cols [k0:k0+BLOCK_H] = x^T @ dz1_k
        tl.atomic_add(dw1_ptr + dcols[:, None] * H1 + kc[None, :],
                      _dot_p(xT, dz1_k, IEEE), mask=(dcols[:, None] < D))
    tl.atomic_add(db2_ptr + c2, tl.sum(dz2, axis=0))


class FusedACMLP:
    """Owns params in [in, out] layout; fwd/bwd via single kernels each."""

    def __init__(self, d_in, units, actions, value_size=1, device='cuda',
                 ieee=False, block_m=64, seed=0):
        assert len(units) == 3, 'prototype: exactly 3 trunk layers'
        h1, h2, h3 = units
        g = torch.Generator(device='cpu').manual_seed(seed)

        def lin(i, o):
            w = torch.empty(o, i).uniform_(-(i ** -0.5), i ** -0.5, generator=g)
            b = torch.empty(o).uniform_(-(i ** -0.5), i ** -0.5, generator=g)
            return w.t().contiguous().to(device), b.to(device)

        self.d, self.units, self.a, self.v = d_in, units, actions, value_size
        self.a_pad = max(16, triton.next_power_of_2(actions))
        self.v_pad = max(16, triton.next_power_of_2(value_size))
        self.d_pad = max(16, triton.next_power_of_2(d_in))
        self.block_m = block_m
        self.bwd_block_m = 32
        self.ieee = ieee

        self.w1, self.b1 = lin(d_in, h1)
        self.w2, self.b2 = lin(h1, h2)
        self.w3, self.b3 = lin(h2, h3)
        wm, self.bm = lin(h3, actions)
        wv, self.bv = lin(h3, value_size)
        # heads padded on out-dim for tl.dot
        self.wm = torch.zeros(h3, self.a_pad, device=device)
        self.wm[:, :actions] = wm
        self.wv = torch.zeros(h3, self.v_pad, device=device)
        self.wv[:, :value_size] = wv
        self.bm = torch.cat([self.bm, torch.zeros(self.a_pad - actions, device=device)])
        self.bv = torch.cat([self.bv, torch.zeros(self.v_pad - value_size, device=device)])
        self.logstd = torch.zeros(actions, device=device)

        self._scratch_n = 0

    def _ensure_scratch(self, n, device):
        if self._scratch_n < n:
            h1, h2, h3 = self.units
            self.h1 = torch.empty(n, h1, device=device)
            self.h2 = torch.empty(n, h2, device=device)
            self.h3 = torch.empty(n, h3, device=device)
            self.mu = torch.empty(n, self.a, device=device)
            self.val = torch.empty(n, self.v, device=device)
            self._scratch_n = n

    def forward(self, x):
        n = x.shape[0]
        self._ensure_scratch(n, x.device)
        h1, h2, h3 = self.units
        grid = (triton.cdiv(n, self.block_m),)
        _ac_mlp_fwd_kernel[grid](
            x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
            self.wm, self.bm, self.wv, self.bv,
            self.h1, self.h2, self.h3, self.mu, self.val,
            n, self.d, self.d_pad, h1, h2, h3,
            self.a, self.a_pad, self.v, self.v_pad,
            self.block_m, 32, self.ieee,
            num_warps=8, num_stages=1,
        )
        return self.mu[:n], self.val[:n]

    def backward(self, x, dmu, dv, dsigma=None):
        n = x.shape[0]
        h1, h2, h3 = self.units
        dev = x.device
        g = {
            'w1': torch.zeros_like(self.w1), 'b1': torch.zeros_like(self.b1),
            'w2': torch.zeros_like(self.w2), 'b2': torch.zeros_like(self.b2),
            'w3': torch.zeros_like(self.w3), 'b3': torch.zeros_like(self.b3),
            'wm': torch.zeros_like(self.wm), 'bm': torch.zeros_like(self.bm),
            'wv': torch.zeros_like(self.wv), 'bv': torch.zeros_like(self.bv),
            'logstd': torch.zeros_like(self.logstd),
        }
        grid = (triton.cdiv(n, self.bwd_block_m),)
        _ac_mlp_bwd_kernel[grid](
            x, self.h1, self.h2, self.h3,
            self.w2, self.w3, self.wm, self.wv,
            dmu.contiguous(), dv.contiguous(),
            dsigma.contiguous() if dsigma is not None else dmu,
            g['w1'], g['b1'], g['w2'], g['b2'], g['w3'], g['b3'],
            g['wm'], g['bm'], g['wv'], g['bv'], g['logstd'],
            n, self.d, self.d_pad, h1, h2, h3,
            self.a, self.a_pad, self.v, self.v_pad,
            self.bwd_block_m, 64, self.ieee, dsigma is not None,
            num_warps=4, num_stages=1,
        )
        return g

    def torch_reference(self, device='cuda'):
        """nn.Module with identical weights for correctness/benchmarks."""
        import torch.nn as nn

        class Ref(nn.Module):
            def __init__(s):
                super().__init__()
                s.l1 = nn.Linear(self.d, self.units[0])
                s.l2 = nn.Linear(self.units[0], self.units[1])
                s.l3 = nn.Linear(self.units[1], self.units[2])
                s.mu = nn.Linear(self.units[2], self.a)
                s.value = nn.Linear(self.units[2], self.v)
                s.act = nn.ELU()

            def forward(s, x):
                h = s.act(s.l1(x))
                h = s.act(s.l2(h))
                h = s.act(s.l3(h))
                return s.mu(h), s.value(h)

        ref = Ref().to(device)
        with torch.no_grad():
            ref.l1.weight.copy_(self.w1.t()); ref.l1.bias.copy_(self.b1)
            ref.l2.weight.copy_(self.w2.t()); ref.l2.bias.copy_(self.b2)
            ref.l3.weight.copy_(self.w3.t()); ref.l3.bias.copy_(self.b3)
            ref.mu.weight.copy_(self.wm[:, :self.a].t()); ref.mu.bias.copy_(self.bm[:self.a])
            ref.value.weight.copy_(self.wv[:, :self.v].t()); ref.value.bias.copy_(self.bv[:self.v])
        return ref
