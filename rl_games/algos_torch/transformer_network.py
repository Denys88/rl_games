import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch import network_builder


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd), cos/sin: (T, hd/2)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.view(1, 1, cos.size(0), cos.size(1))
    sin = sin.view(1, 1, sin.size(0), sin.size(1))
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, q_in, kv_in, attn_mask, rope_q, rope_k):
        B, Tq, _ = q_in.shape
        Tk = kv_in.size(1)
        q = self.q_proj(q_in).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), *rope_q)
        k = apply_rope(self.k_norm(k), *rope_k)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, Tq, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_hidden)

    def forward(self, x, mem, attn_mask, rope_q, rope_k):
        # x: (B, Tq, d) current tokens; mem: (B, M, d) cached inputs to this
        # block from past steps (stop-gradient), or None
        kv = x if mem is None else torch.cat([mem, x], dim=1)
        kv_n = self.attn_norm(kv)
        q_n = kv_n[:, -x.size(1):]
        x = x + self.attn(q_n, kv_n, attn_mask, rope_q, rope_k)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerBuilder(network_builder.A2CBuilder):
    """Actor-critic with a small Qwen3-style causal transformer over recent steps.

    Each env step's observation is embedded by the mlp trunk into one token
    (d_model = last mlp layer). A stack of pre-RMSNorm blocks with grouped-query
    attention (QK-RMSNorm, RoPE) and SwiGLU MLPs attends causally over a short
    window of past steps. History rides through the standard rnn-state plumbing,
    so zero_rnn_on_done clears it on reset and no trainer changes are needed.

    Config, under `network`:

        name: actor_critic_transformer
        transformer:
          max_steps: 16     # context window
          n_layers: 2
          n_heads: 4
          n_kv_heads: 2
          ffn_hidden: 256
          rope_theta: 10000.0
          memory: False     # Transformer-XL-style cached context, see below

    Default mode (`memory: False`): the state is the raw token window; rollout
    recomputes attention over it each step, training runs causal attention with
    done-segment masking inside each seq_length chunk. Chunks start with empty
    context, so set seq_length == max_steps.

    TrXL mode (`memory: True`): the state caches every block's input hiddens
    for the last max_steps steps (stop-gradient, computed when the step was
    current). Both rollout and training attend over [cache | current], which
    makes them exactly consistent, extends the receptive field by max_steps per
    layer, and allows seq_length < max_steps (short gradient chunks, long
    context). Cached hiddens replay slightly stale activations during the PPO
    epoch, TrXL-style. Requires zero_rnn_on_done: True (the default).

    Not supported: separate critic, cnn, rnn block, multi-discrete actions.
    """

    def build(self, name, **kwargs):
        return TransformerBuilder.Network(self.params, **kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            if self.separate:
                raise NotImplementedError('actor_critic_transformer does not support separate critic')
            if self.has_cnn:
                raise NotImplementedError('actor_critic_transformer does not support cnn')
            if self.has_rnn:
                raise NotImplementedError('remove the rnn block; the transformer replaces it')
            if self.is_multi_discrete:
                raise NotImplementedError('actor_critic_transformer does not support multi-discrete action spaces')
            if len(self.units) == 0:
                raise ValueError('actor_critic_transformer requires a non-empty mlp (it is the token embedding)')

            cfg = params.get('transformer', {})
            self.d_model = self.units[-1]
            self.max_steps = cfg.get('max_steps', 16)
            self.n_layers = cfg.get('n_layers', 2)
            n_heads = cfg.get('n_heads', 4)
            n_kv_heads = cfg.get('n_kv_heads', n_heads)
            ffn_hidden = cfg.get('ffn_hidden', 2 * self.d_model)
            rope_theta = cfg.get('rope_theta', 10000.0)
            self.use_memory = cfg.get('memory', False)

            self.blocks = nn.ModuleList(
                [Block(self.d_model, n_heads, n_kv_heads, ffn_hidden) for _ in range(self.n_layers)])
            self.final_norm = RMSNorm(self.d_model)

            for m in self.blocks.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

            head_dim = self.d_model // n_heads
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            # memory mode attends over [cache | chunk], so allow 2x positions
            t = torch.arange(2 * self.max_steps + 1).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer('rope_cos', freqs.cos(), persistent=False)
            self.register_buffer('rope_sin', freqs.sin(), persistent=False)

        def is_rnn(self):
            return True

        def get_default_rnn_state(self):
            n_bufs = self.n_layers if self.use_memory else 1
            return tuple(
                [torch.zeros((self.max_steps, self.num_seqs, self.d_model)) for _ in range(n_bufs)]
                + [torch.zeros((self.max_steps, self.num_seqs, 1))])

        def _rope(self, start, end):
            return (self.rope_cos[start:end], self.rope_sin[start:end])

        def _heads(self, out, states):
            value = self.value_act(self.value(out))
            if self.central_value:
                return value, states
            if self.is_discrete:
                return self.logits(out), value, states
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, sigma, value, states

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            e = self.actor_mlp(obs)
            if obs_dict.get('is_train', False):
                seq_length = obs_dict.get('seq_length', 1)
                if seq_length > self.max_steps:
                    raise ValueError(f'seq_length {seq_length} exceeds transformer max_steps {self.max_steps}')
                if self.use_memory:
                    return self._train_memory(e, obs_dict, states, seq_length)
                return self._train_window(e, obs_dict, states, seq_length)
            if self.use_memory:
                return self._step_memory(e, states)
            return self._step_window(e, states)

        # ---- default mode: raw token window, recomputed each step ----

        def _train_window(self, e, obs_dict, states, S):
            B = e.size(0) // S
            x = e.view(B, S, -1)
            causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            dones = obs_dict.get('dones', None)
            if dones is not None:
                seg = dones.view(B, S).float().cumsum(dim=1)
                same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)
                attn_mask = (causal.unsqueeze(0) & same_seg).unsqueeze(1)
            else:
                attn_mask = causal.view(1, 1, S, S)
            rope = self._rope(0, S)
            for block in self.blocks:
                x = block(x, None, attn_mask, rope, rope)
            out = self.final_norm(x).reshape(B * S, -1)
            return self._heads(out, states)

        def _step_window(self, e, states):
            buf, valid = states
            e = e.to(buf.dtype)
            buf = torch.cat([buf[1:], e.unsqueeze(0)], dim=0)
            valid = torch.cat([valid[1:], torch.ones_like(valid[:1])], dim=0)

            x = buf.transpose(0, 1)                                   # (B, W, d)
            key_mask = valid.permute(1, 2, 0).unsqueeze(1) > 0        # (B, 1, 1, W)
            # causal within the window as well: without it, earlier tokens'
            # layer-1 features would attend to later tokens and leak future
            # information into deeper layers' keys/values
            w = buf.size(0)
            causal = torch.ones(w, w, dtype=torch.bool, device=x.device).tril()
            attn_mask = causal.view(1, 1, w, w) & key_mask
            rope = self._rope(0, w)
            for block in self.blocks:
                x = block(x, None, attn_mask, rope, rope)
            out = self.final_norm(x)[:, -1]
            return self._heads(out, (buf, valid))

        # ---- TrXL mode: per-block cached hiddens, exact rollout/train match ----

        def _train_memory(self, e, obs_dict, states, S):
            dones = obs_dict.get('dones', None)
            if dones is None:
                raise ValueError('transformer memory requires zero_rnn_on_done: True')
            mems, valid = states[:-1], states[-1]
            W = self.max_steps
            T = W + S
            B = e.size(0) // S
            x = e.view(B, S, -1)

            k_idx = torch.arange(T, device=x.device).view(1, -1)
            q_idx = torch.arange(S, device=x.device).view(-1, 1)
            is_mem = k_idx < W
            # cache slot i was evicted from the rollout ring buffer before chunk
            # step j ran unless i >= j — training must not see deeper history
            # than the rollout had
            causal = (is_mem & (k_idx >= q_idx)) | (~is_mem & (k_idx - W <= q_idx))
            key_valid = torch.cat(
                [valid.permute(1, 2, 0) > 0,
                 torch.ones(B, 1, S, dtype=torch.bool, device=x.device)], dim=2).unsqueeze(2)
            # cached tokens belong to segment 0: a chunk token may look into the
            # cache only while no done has occurred yet in its chunk
            seg_chunk = dones.view(B, S).float().cumsum(dim=1)
            seg_all = torch.cat([torch.zeros(B, W, device=x.device), seg_chunk], dim=1)
            same_seg = (seg_chunk.unsqueeze(2) == seg_all.unsqueeze(1)).unsqueeze(1)
            attn_mask = causal.view(1, 1, S, T) & key_valid & same_seg

            rope_q, rope_k = self._rope(W, T), self._rope(0, T)
            for block, mem in zip(self.blocks, mems):
                x = block(x, mem.to(x.dtype).transpose(0, 1), attn_mask, rope_q, rope_k)
            out = self.final_norm(x).reshape(B * S, -1)
            return self._heads(out, states)

        def _step_memory(self, e, states):
            mems, valid = states[:-1], states[-1]
            W = self.max_steps
            x = e.to(mems[0].dtype).unsqueeze(1)                      # (B, 1, d)
            key_mask = torch.cat(
                [valid.permute(1, 2, 0) > 0,
                 torch.ones(x.size(0), 1, 1, dtype=torch.bool, device=x.device)],
                dim=2).unsqueeze(2)                                   # (B, 1, 1, W+1)
            rope_q, rope_k = self._rope(W, W + 1), self._rope(0, W + 1)
            new_states = []
            for block, mem in zip(self.blocks, mems):
                new_states.append(torch.cat([mem[1:], x.transpose(0, 1)], dim=0))
                x = block(x, mem.transpose(0, 1), key_mask, rope_q, rope_k)
            new_states.append(torch.cat([valid[1:], torch.ones_like(valid[:1])], dim=0))
            out = self.final_norm(x)[:, 0]
            return self._heads(out, tuple(new_states))
