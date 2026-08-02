"""Candidate NumPy engine for the InferRef Agent repair evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def rotate_half(value: np.ndarray) -> np.ndarray:
    """Apply the engine's half-split rotation convention."""

    half = value.shape[-1] // 2
    first = value[..., :half]
    second = value[..., half:]
    return np.concatenate((second, -first), axis=-1)


def apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return value * cos + rotate_half(value) * sin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testcase")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    testcase = Path(args.testcase)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((testcase / "testcase.json").read_text(encoding="utf-8"))
    tensors = {
        entry["name"]: codec.read(testcase / entry["payload"]).as_comparable()
        for entry in manifest["inputs"]
    }
    results = {
        "q_embed": apply_rope(tensors["query"], tensors["cos"], tensors["sin"]),
        "k_embed": apply_rope(tensors["key"], tensors["cos"], tensors["sin"]),
    }
    outputs = []
    for name, value in results.items():
        relative = f"{name}.irtensor"
        codec.write_array(output / relative, np.asarray(value, dtype=np.float32))
        outputs.append({"name": name, "payload": relative})

    (output / "manifest.json").write_text(
        json.dumps({"engine": "agent-eval-rope", "outputs": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
