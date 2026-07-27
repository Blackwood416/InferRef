"""Produce a reference trace of the mini Llama block.

::

    python examples/mini_llama/run_trace.py --output trace/

Also runnable through the CLI, which exercises the "trace a script you do not
control" path::

    inferref trace examples/mini_llama/run_trace.py --output trace/ \
        --capture-tensors all --scope layers.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MiniLlamaConfig, build_inputs, build_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="trace/", help="output trace directory")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--scope", help="restrict tracing to this module path")
    parser.add_argument("--capture-tensors", default="all")
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="just run the model (used when the CLI is doing the tracing)",
    )
    args = parser.parse_args(argv)

    model = build_model(seed=0, config=MiniLlamaConfig(num_layers=args.layers))
    hidden = build_inputs(model, batch=args.batch, seq_len=args.seq_len)

    if args.no_trace:
        with torch.no_grad():
            model(hidden)
        return 0

    import inferref

    with torch.no_grad():
        with inferref.trace(
            output=args.output,
            scope=args.scope,
            capture_tensors=args.capture_tensors,
            model_name="MiniLlama",
            seed=0,
        ) as session:
            session.mark_input("hidden_states", hidden)
            output = model(hidden)
            session.mark_output("output", output)

    package = session.package
    assert package is not None
    print(f"Wrote trace to {args.output}")
    print(f"  operators: {len(package.graph.operators)}")
    print(f"  values:    {len(package.graph.values)}")
    print(f"  modules:   {len(package.modules)}")
    print(f"  sources:   {len(package.sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
