#!/usr/bin/env python3
"""Print layer 0's RoPE operator range as shell assignments.

Used by CI so the end-to-end step does not hardcode operator ids, which would
break confusingly whenever the example model changes.

    eval "$(python .github/scripts/rope_range.py trace/)"
    inferref region create trace/ --name RoPE --from-op "$ROPE_FROM" --to-op "$ROPE_TO"

Selecting by source function alone is not enough: ``apply_rotary_pos_emb``
calls ``rotate_half``, whose operators carry the helper's source location. The
resulting node set has holes, and a region with holes picks up extra boundary
inputs. So the range is taken over *execution order* and includes everything
issued between the first and last RoPE operator of the first invocation.
"""

from __future__ import annotations

import sys

from inferref.ir.package import TracePackage

ROPE_FUNCTIONS = ("apply_rotary_pos_emb", "rotate_half")
#: Generous upper bound on one invocation's length; the real run is ~18 ops.
MAX_INVOCATION_SPAN = 30


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <trace-dir>", file=sys.stderr)
        return 2

    package = TracePackage.load(argv[1])
    rope_ops = [
        op
        for op in package.graph.ops_in_execution_order()
        if (source := package.source(op.source_id)) is not None
        and source.primary is not None
        and source.primary.function in ROPE_FUNCTIONS
    ]
    if not rope_ops:
        print("no RoPE operators found in this trace", file=sys.stderr)
        return 1

    start = rope_ops[0]
    last = next(
        op
        for op in reversed(rope_ops)
        if op.execution_index < start.execution_index + MAX_INVOCATION_SPAN
    )

    print(f"ROPE_FROM={start.id}")
    print(f"ROPE_TO={last.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
