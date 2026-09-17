"""Generate golden .irtensor fixtures for the Windows C++ cross-language check.

The Windows C++ job deliberately does not install PyTorch: MSVC compilation,
the reader self-test, and the Python-writer -> C++-reader agreement can all be
verified with numpy alone. This script emits a small set of tensors that the
job compares with ``inferref_compare`` in both the positive and a deliberately
corrupted negative direction.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from inferref.ir.dtypes import FLOAT8_DTYPES
from inferref.tensor import codec
from inferref.tensor.dtype_numpy import float8_bytes_to_float32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="directory for the generated fixtures")
    parser.add_argument("--fp8-selftest", type=Path, help="run C++ FP8 reader and verify its writes")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    codec.write_array(output / "a.irtensor", a)
    codec.write_array(output / "a_equal.irtensor", a.copy())
    corrupted = a.copy()
    corrupted[1, 1] += 1.0
    codec.write_array(output / "a_corrupted.irtensor", corrupted)

    h = np.linspace(0.0, 1.0, 8, dtype=np.float16)
    codec.write_array(output / "h.irtensor", h)
    codec.write_array(output / "h_equal.irtensor", h.copy())

    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for dtype in sorted(FLOAT8_DTYPES):
        codec.write_array(output / f"{dtype}.irtensor", raw, dtype=dtype)
        codec.write_array(output / f"{dtype}.float32.irtensor", float8_bytes_to_float32(raw, dtype))
    if args.fp8_selftest:
        subprocess.run([str(args.fp8_selftest.resolve()), str(output.resolve())], check=True)
        for dtype in sorted(FLOAT8_DTYPES):
            original = (output / f"{dtype}.irtensor").read_bytes()
            rewritten = (output / f"{dtype}.roundtrip.irtensor").read_bytes()
            if original != rewritten:
                raise RuntimeError(f"C++ FP8 roundtrip changed wire bytes: {dtype}")

    print(f"wrote golden fixtures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
