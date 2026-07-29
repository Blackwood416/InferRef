"""A deterministic static KV-cache decode path for the mini Llama model.

The original :mod:`model` stays a stable prefill-only reference.  This module
adds a separate attention implementation whose cache writes are deliberately
ordinary dispatcher-visible ``copy_`` operations.  It therefore exercises the
same storage mutation contract an inference engine needs without relying on a
framework-specific cache object.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import MiniLlamaConfig, apply_rotary_pos_emb, repeat_kv


class StaticKVCache(nn.Module):
    """Fixed-capacity key/value storage for one attention layer.

    ``cache_position`` is explicit so a trace describes the exact write range.
    The cache never reallocates: prefill and every decode step mutate the same
    two storages, which is the important invariant for InferRef's versioned
    value model.
    """

    def __init__(
        self,
        config: MiniLlamaConfig,
        *,
        batch_size: int = 1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        shape = (
            batch_size,
            config.num_kv_heads,
            max_seq_len,
            config.head_dim,
        )
        self.max_seq_len = max_seq_len
        self.register_buffer("key_cache", torch.zeros(shape))
        self.register_buffer("value_cache", torch.zeros(shape))

    def update_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one prefill chunk or decode chunk and return the live prefix."""
        seq_len = key_states.shape[2]
        end = cache_position + seq_len
        if cache_position < 0 or end > self.max_seq_len:
            raise ValueError(
                f"cache write [{cache_position}:{end}] exceeds capacity "
                f"{self.max_seq_len}"
            )

        # Slice assignment lowers to view-producing slice ops followed by
        # schema-declared ``aten.copy_`` writes.  Both the target view and the
        # full cache share one storage, making this an adversarial test of
        # InferRef's storage-level versioning rather than object-level mutation.
        self.key_cache[:, :, cache_position:end, :].copy_(key_states)
        self.value_cache[:, :, cache_position:end, :].copy_(value_states)
        return (
            self.key_cache[:, :, :end, :],
            self.value_cache[:, :, :end, :],
        )

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.update_kv_cache(key_states, value_states, cache_position)


class CachedAttention(nn.Module):
    """Grouped-query attention with a dispatcher-visible static KV cache."""

    def __init__(
        self,
        config: MiniLlamaConfig,
        *,
        batch_size: int = 1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.cache = StaticKVCache(
            config, batch_size=batch_size, max_seq_len=max_seq_len
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key, value = self.cache(key, value, cache_position)

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        total_len = key.shape[2]
        query_positions = torch.arange(cache_position, cache_position + seq_len)
        key_positions = torch.arange(total_len)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        causal_mask = torch.zeros(
            (seq_len, total_len), dtype=scores.dtype, device=scores.device
        ).masked_fill(future, float("-inf"))
        weights = torch.softmax(scores + causal_mask, dim=-1)

        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(context)


def copy_attention_weights(source: nn.Module, target: CachedAttention) -> None:
    """Copy the four projection matrices from a prefill attention module."""
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(target, name).weight.copy_(getattr(source, name).weight)
