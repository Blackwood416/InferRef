"""Candidate NumPy engine for the InferRef indexed KV-cache evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def update_cache(
    cache: np.ndarray,
    update: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Overwrite one sequence position in a static KV cache."""

    batch, heads, capacity, dimension = cache.shape
    cache_out = cache.copy()
    start = int(index[0])
    sequence_view = cache_out.reshape(batch, capacity, heads, dimension)
    sequence_view[:, start : start + 1, :, :] = update.transpose(0, 2, 1, 3)
    return cache_out


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
        entry["name"]: codec.read(testcase / entry["payload"]).data
        for entry in manifest["inputs"]
    }
    cache_out = update_cache(tensors["cache"], tensors["update"], tensors["index"])
    relative = "cache_out.irtensor"
    codec.write_array(output / relative, np.asarray(cache_out, dtype=np.float32))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "engine": "agent-eval-kv-cache-index",
                "outputs": [{"name": "cache_out", "payload": relative}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
