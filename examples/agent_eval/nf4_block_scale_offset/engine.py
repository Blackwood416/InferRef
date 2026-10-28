"""Candidate NumPy engine for the InferRef NF4 block-scale evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def dequantize_nf4(
    codes: np.ndarray,
    scales: np.ndarray,
    codebook: np.ndarray,
    group_size: np.ndarray,
) -> np.ndarray:
    """Dequantize NF4 codes with one scale per flat block."""

    rows, columns = codes.shape
    blocks_per_row = scales.shape[1]
    output = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            scale_index = row * blocks_per_row + column // int(group_size[0])
            output[row, column] = (
                codebook[int(codes[row, column])] * scales.reshape(-1)[scale_index]
            )
    return output


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
    weight = dequantize_nf4(
        tensors["codes"],
        tensors["scales"],
        tensors["codebook"],
        tensors["group_size"],
    )
    relative = "weight.irtensor"
    codec.write_array(output / relative, np.asarray(weight, dtype=np.float32))
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "engine": "agent-eval-nf4-block-scale",
                "outputs": [{"name": "weight", "payload": relative}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
