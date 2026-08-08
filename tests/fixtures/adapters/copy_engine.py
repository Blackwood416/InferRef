"""Numpy-only fixture adapter for scenario tests (no torch).

The engine appends ``update`` to ``cache`` along the KV sequence axis and
optionally computes ``logits`` from an unbound ``scale`` input. It honors
``INFERREF_ENGINE_STATE_CORRUPTION`` so tests can exercise engine-state
validation without a real accelerator:

- ``shape``   truncates the appended cache by one sequence position;
- ``dtype``   writes ``cache_out`` as float16;
- ``value``   writes the correct shape but shifts every value by one;
- ``missing-state`` omits ``cache_out`` entirely (engine-state fail-fast case).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def main() -> int:
    testcase = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((testcase / "testcase.json").read_text(encoding="utf-8"))
    inputs = {
        entry["name"]: entry
        for entry in manifest.get("inputs", [])
        if isinstance(entry, dict)
    }
    cache = codec.read(testcase / inputs["cache"]["payload"]).data
    update = codec.read(testcase / inputs["update"]["payload"]).data
    cache_out = np.concatenate([cache, update], axis=2)
    corruption = os.environ.get("INFERREF_ENGINE_STATE_CORRUPTION")
    if corruption == "missing-state":
        if "scale" in inputs:
            scale = codec.read(testcase / inputs["scale"]["payload"]).data
            logits = (update * scale).sum(axis=2, keepdims=True)
            codec.write_array(output / "logits.irtensor", np.ascontiguousarray(logits))
        return 0
    if corruption == "shape":
        cache_out = cache_out[:, :, :-1, :]
        cache_out = np.ascontiguousarray(cache_out)
    elif corruption == "dtype":
        cache_out = cache_out.astype(np.float16)
    elif corruption == "value":
        cache_out = cache_out + 1.0
    codec.write_array(output / "cache_out.irtensor", np.ascontiguousarray(cache_out))
    if "scale" in inputs:
        scale = codec.read(testcase / inputs["scale"]["payload"]).data
        logits = (update * scale).sum(axis=2, keepdims=True)
        codec.write_array(output / "logits.irtensor", np.ascontiguousarray(logits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
