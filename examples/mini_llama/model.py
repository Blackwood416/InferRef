"""A minimal Llama-style transformer block, in plain PyTorch.

Deliberately self-contained: no Hugging Face download, no network, fast and
deterministic. It dispatches the same ATen operators a real Llama/Qwen block
does — RMSNorm, rotary embedding, grouped-query attention, SwiGLU — so a trace
of it exercises InferRef exactly like a trace of the real thing.

For the real-model path see ``examples/hf_causal_lm/`` (needs the ``hf`` extra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniLlamaConfig:
    hidden_size: int = 64
    intermediate_size: int = 128
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    num_layers: int = 2


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as used by Llama/Qwen."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    """Precomputes the rotary cos/sin tables."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension onto the first."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The RoPE reference region: slice, neg, cat, mul, mul, add (SPEC §51)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for grouped-query attention."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """Grouped-query self-attention with a causal mask."""

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        scores = scores + causal_mask
        weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(context)


class SwiGLUMLP(nn.Module):
    """The gated feed-forward network used by Llama-family models."""

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """One transformer block: norm -> attention -> norm -> MLP, with residuals."""

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cos, sin)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class MiniLlama(nn.Module):
    """A stack of decoder layers with a final norm."""

    def __init__(self, config: MiniLlamaConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniLlamaConfig()
        self.rotary = RotaryEmbedding(self.config.head_dim, self.config.rope_theta)
        self.layers = nn.ModuleList(
            [DecoderLayer(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary(seq_len)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


def build_model(seed: int = 0, config: MiniLlamaConfig | None = None) -> MiniLlama:
    """Build a deterministically initialised model in eval mode."""
    torch.manual_seed(seed)
    model = MiniLlama(config)
    model.eval()
    return model


def build_inputs(
    model: MiniLlama, batch: int = 1, seq_len: int = 8, seed: int = 1
) -> torch.Tensor:
    """Deterministic hidden-state input for one prefill step."""
    torch.manual_seed(seed)
    return torch.randn(batch, seq_len, model.config.hidden_size)
