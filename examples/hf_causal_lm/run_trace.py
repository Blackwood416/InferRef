"""Trace a real Hugging Face causal LM (SPEC §65).

Optional example — requires the ``hf`` extra::

    uv pip install -e ".[hf]"

The mini-Llama example is the hermetic default used by the test suite; this one
exists to show that nothing about InferRef is specific to a hand-written model.
Nothing here is InferRef-aware: module paths, scope filtering and source
mapping all come from PyTorch's own hooks and stack.

By default it builds a *randomly initialised* tiny Llama from a config, so it
runs offline with no download. Pass ``--model`` to trace real weights::

    python examples/hf_causal_lm/run_trace.py --model Qwen/Qwen2.5-0.5B \\
        --scope model.layers.0 --output trace/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import inferref


def build_tiny_model():
    """A randomly initialised Llama small enough to trace comfortably."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def load_pretrained(name: str, revision: str | None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.float32
    )
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Hugging Face model id; omit for an offline tiny Llama"
    )
    parser.add_argument("--revision", help="model revision, recorded in the manifest")
    parser.add_argument("-o", "--output", default="trace/", help="output trace directory")
    parser.add_argument(
        "--scope",
        default="model.layers.0",
        help="module path to trace; the default keeps the trace small",
    )
    parser.add_argument(
        "--capture-tensors",
        default="all",
        help="tensor capture policy (SPEC §14)",
    )
    parser.add_argument("--seq-len", type=int, default=8, help="prefill length")
    args = parser.parse_args(argv)

    if args.model:
        model = load_pretrained(args.model, args.revision)
        model_name = args.model
    else:
        model = build_tiny_model()
        model_name = "tiny-llama-random"

    vocab_size = int(model.config.vocab_size)
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (1, args.seq_len))

    output = Path(args.output)
    with torch.no_grad(), inferref.trace(
        output=output,
        scope=args.scope,
        capture_tensors=args.capture_tensors,
        model_name=model_name,
        seed=0,
    ) as session:
        session.mark_input("input_ids", input_ids)
        result = model(input_ids=input_ids, use_cache=False)
        session.mark_output("logits", result.logits)

    package = session.package
    assert package is not None
    print(f"Wrote trace to {output}")
    print(f"  model:     {model_name}")
    print(f"  scope:     {args.scope}")
    print(f"  operators: {len(package.graph.operators)}")
    print(f"  values:    {len(package.graph.values)}")
    print()
    print("Next:")
    print(f"  inferref inspect {output}")
    print(f"  inferref analyze {output}")
    print(
        f"  inferref region create {output} --name RoPE "
        "--source-function apply_rotary_pos_emb"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
