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

    def forward(self, x, attn_mask, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


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

    def forward(self, x, attn_mask, cos, sin):
        x = x + self.attn(self.attn_norm(x), attn_mask, cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerBuilder(network_builder.A2CBuilder):
    """Actor-critic with a small Qwen3-style causal transformer over recent steps.

    Each env step's observation is embedded by the mlp trunk into one token
    (d_model = last mlp layer). A stack of pre-RMSNorm blocks with grouped-query
    attention (QK-RMSNorm, RoPE) and SwiGLU MLPs attends causally over a short
    window of past steps. History rides through the standard rnn-state plumbing:
    state[0] is the token window, state[1] a validity mask; zero_rnn_on_done
    clears both on reset. During training, attention is masked at done
    boundaries within each seq_length chunk (requires zero_rnn_on_done: True
    for masking; seq_length must equal transformer.max_steps).

    Config, under `network`:

        name: actor_critic_transformer
        transformer:
          max_steps: 16     # context window; set algo seq_length to the same
          n_layers: 2
          n_heads: 4
          n_kv_heads: 2
          ffn_hidden: 256
          rope_theta: 10000.0

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
            n_layers = cfg.get('n_layers', 2)
            n_heads = cfg.get('n_heads', 4)
            n_kv_heads = cfg.get('n_kv_heads', n_heads)
            ffn_hidden = cfg.get('ffn_hidden', 2 * self.d_model)
            rope_theta = cfg.get('rope_theta', 10000.0)

            self.blocks = nn.ModuleList(
                [Block(self.d_model, n_heads, n_kv_heads, ffn_hidden) for _ in range(n_layers)])
            self.final_norm = RMSNorm(self.d_model)

            for m in self.blocks.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

            head_dim = self.d_model // n_heads
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            t = torch.arange(self.max_steps).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer('rope_cos', freqs.cos(), persistent=False)
            self.register_buffer('rope_sin', freqs.sin(), persistent=False)

        def is_rnn(self):
            return True

        def get_default_rnn_state(self):
            return (torch.zeros((self.max_steps, self.num_seqs, self.d_model)),
                    torch.zeros((self.max_steps, self.num_seqs, 1)))

        def _torso(self, x, attn_mask):
            T = x.size(1)
            cos, sin = self.rope_cos[:T].to(x.dtype), self.rope_sin[:T].to(x.dtype)
            for block in self.blocks:
                x = block(x, attn_mask, cos, sin)
            return self.final_norm(x)

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
                B = e.size(0) // seq_length
                x = e.view(B, seq_length, -1)

                causal = torch.ones(seq_length, seq_length, dtype=torch.bool, device=x.device).tril()
                dones = obs_dict.get('dones', None)
                if dones is not None:
                    seg = dones.view(B, seq_length).float().cumsum(dim=1)
                    same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)
                    attn_mask = (causal.unsqueeze(0) & same_seg).unsqueeze(1)
                else:
                    attn_mask = causal.view(1, 1, seq_length, seq_length)

                out = self._torso(x, attn_mask).reshape(B * seq_length, -1)
                return self._heads(out, states)

            # single env step: append the new token to the sliding window
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
            out = self._torso(x, causal.view(1, 1, w, w) & key_mask)[:, -1]
            return self._heads(out, (buf, valid))
