"""Trace prefill plus one decode step through a static KV cache.

This example is intentionally separate from ``run_trace.py`` so the original
206-op prefill baseline stays stable.  It validates the cached path against an
uncached causal-attention reference before writing the trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mini_llama.kv_cache import CachedAttention, copy_attention_weights  # noqa: E402
from examples.mini_llama.model import Attention, MiniLlamaConfig, RotaryEmbedding  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="trace-kv/")
    parser.add_argument("--prefill-len", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--capture-tensors", default="all")
    args = parser.parse_args(argv)

    total_len = args.prefill_len + 1
    if total_len > args.max_seq_len:
        parser.error("prefill plus decode token exceeds --max-seq-len")

    config = MiniLlamaConfig(num_layers=1)
    torch.manual_seed(0)
    reference = Attention(config).eval()
    cached = CachedAttention(config, max_seq_len=args.max_seq_len).eval()
    copy_attention_weights(reference, cached)

    torch.manual_seed(1)
    hidden = torch.randn(1, total_len, config.hidden_size)
    cos, sin = RotaryEmbedding(config.head_dim)(total_len)

    with torch.no_grad():
        expected = reference(hidden, cos, sin)

    import inferref

    with torch.no_grad(), inferref.trace(
        output=args.output,
        capture_tensors=args.capture_tensors,
        model_name="CachedAttention",
        seed=0,
        semantic_analysis=True,
    ) as session:
        prefill_hidden = hidden[:, : args.prefill_len]
        decode_hidden = hidden[:, args.prefill_len :]
        session.mark_input("prefill_hidden", prefill_hidden)
        prefill = cached(
            prefill_hidden,
            cos[: args.prefill_len],
            sin[: args.prefill_len],
            0,
        )
        session.mark_output("prefill_output", prefill)

        session.mark_input("decode_hidden", decode_hidden)
        decode = cached(
            decode_hidden,
            cos[args.prefill_len :],
            sin[args.prefill_len :],
            args.prefill_len,
        )
        session.mark_output("decode_output", decode)

    torch.testing.assert_close(prefill, expected[:, : args.prefill_len])
    torch.testing.assert_close(decode, expected[:, args.prefill_len :])

    package = session.package
    assert package is not None
    cache_regions = [
        region
        for region in package.regions
        if region.semantic is not None and region.semantic.name == "KVCacheUpdate"
    ]
    print(f"Wrote trace to {args.output}")
    print(f"  operators:        {len(package.graph.operators)}")
    print(f"  cache mutations:  {sum(bool(op.effects.mutated_storages) for op in package.graph.operators)}")
    print(f"  KV-cache regions: {len(cache_regions)}")
    print("  numerical check:  cached prefill/decode == uncached reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
