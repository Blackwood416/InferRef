"""A numpy "inference engine" that consumes an InferRef testcase.

Stands in for a real CUDA/SYCL kernel so the full validation loop can be
demonstrated without building C++. It reads ``.irtensor`` inputs, computes
rotary embedding, and writes ``.irtensor`` outputs — the language-independent
engine output protocol of SPEC §22.

Note what it does *not* need: PyTorch, the model source, or the original trace.

::

    python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
    inferref compare repro/rope engine-out/ --first-failure

Pass ``--inject-bug`` to introduce a realistic error (rotating the wrong half)
and watch the comparator localise it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate the second half of the last dimension onto the first."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)


def rotate_half_buggy(x: np.ndarray) -> np.ndarray:
    """A plausible off-by-one-half bug: the negation lands on the wrong side."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((x2, -x1), axis=-1)


def apply_rope(
    q: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, buggy: bool = False
) -> np.ndarray:
    """q * cos + rotate_half(q) * sin, broadcasting cos/sin over batch and heads."""
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    rotate = rotate_half_buggy if buggy else rotate_half
    return (q * cos_b) + (rotate(q) * sin_b)


def load_inputs(testcase_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((testcase_dir / "testcase.json").read_text(encoding="utf-8"))
    tensors: dict[str, np.ndarray] = {}
    for entry in manifest["inputs"]:
        if not entry.get("payload"):
            raise SystemExit(
                f"input {entry['name']!r} has no payload; re-trace with "
                "--capture-tensors all"
            )
        view = codec.read(testcase_dir / entry["payload"])
        tensors[entry["name"]] = view.as_comparable().astype(np.float32)
    return tensors, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testcase", help="testcase directory produced by inferref")
    parser.add_argument("-o", "--output", default="engine-out/", help="engine output directory")
    parser.add_argument(
        "--inject-bug",
        action="store_true",
        help="compute a deliberately wrong result, to demonstrate --first-failure",
    )
    args = parser.parse_args(argv)

    testcase_dir = Path(args.testcase)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors, manifest = load_inputs(testcase_dir)

    # Identify the boundary tensors. Names come from `--input-names` when the
    # testcase was extracted; otherwise fall back to rank (an engine adapter
    # dispatching on shape is realistic).
    if {"query", "key", "cos", "sin"} <= tensors.keys():
        query, key = tensors["query"], tensors["key"]
        cos, sin = tensors["cos"], tensors["sin"]
    else:
        rank2 = [v for v in tensors.values() if v.ndim == 2]
        rank4 = [v for v in tensors.values() if v.ndim == 4]
        if len(rank2) != 2 or len(rank4) != 2:
            raise SystemExit(
                "cannot identify RoPE inputs; extract the testcase with "
                "--input-names query,key,cos,sin"
            )
        cos, sin = rank2
        query, key = rank4

    q_embed = apply_rope(query, cos, sin, buggy=args.inject_bug)
    # The bug only affects the query path, so the comparator has a passing
    # tensor and a failing one to order.
    k_embed = apply_rope(key, cos, sin, buggy=False)

    output_names = [entry["name"] for entry in manifest["outputs"]]
    results = [q_embed, k_embed]
    written: list[dict[str, str]] = []
    for name, array in zip(output_names, results):
        relative = f"{name}.irtensor"
        codec.write_array(output_dir / relative, array)
        written.append({"name": name, "payload": relative})

    (output_dir / "manifest.json").write_text(
        json.dumps({"engine": "rope_numpy", "outputs": written}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(written)} tensor(s) to {output_dir}")
    for entry in written:
        print(f"  {entry['payload']}")
    if args.inject_bug:
        print("(computed with --inject-bug: the query path is deliberately wrong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
