"""Generate golden .irtensor fixtures for the Windows C++ cross-language check.

The Windows C++ job deliberately does not install PyTorch: MSVC compilation,
the reader self-test, and the Python-writer -> C++-reader agreement can all be
verified with numpy alone. This script emits a small set of tensors that the
job compares with ``inferref_compare`` in both the positive and a deliberately
corrupted negative direction.

``--quantization-goldens DIR`` additionally writes testcase format 0.4 fixtures
plus float32 bit-pattern expectations for ``inferref_quantization_selftest``.
The fixtures are deterministic and are checked into
``tests/fixtures/cpp_quantization_goldens`` so a mismatch is reviewable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inferref.ir.dtypes import FLOAT8_DTYPES
from inferref.quantization import QuantizationSpec, dequantize
from inferref.quantization.nf4 import NF4_CODEBOOK
from inferref.tensor import codec
from inferref.tensor.dtype_numpy import float8_bytes_to_float32
from inferref.testcase.requirements import derive_requirements
from inferref.testcase.validate import validate_testcase

QUANTIZATION_GOLDEN_FORMAT = "inferref-cpp-quantization-goldens"
QUANTIZATION_GOLDEN_VERSION = "1.0"

# Value IDs are arbitrary but fixed: they exercise the value_id key, not
# positional indices, and keep regenerated fixtures byte-identical.
_WEIGHT_ID = 100
_OUTPUT_ID = 101
_SCALE_ID = 10
_ZERO_ID = 11
_CODEBOOK_ID = 12

_AFFINE_STORAGE = {
    "affine-int8/v1": "int8",
    "affine-int4/v1": "int4",
    "affine-uint4/v1": "uint4",
}


@dataclass(frozen=True)
class QuantizationCase:
    """One standalone quantized input plus the Python reference result."""

    name: str
    scheme: str
    granularity: str
    shape: tuple[int, ...]
    codes: np.ndarray
    scale: np.ndarray
    zero: np.ndarray | None = None
    axis: int | None = None
    group_size: int | None = None
    codebook: np.ndarray | None = None


def _canonical_stride(shape: tuple[int, ...]) -> list[int]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(dim, 1)
    return list(reversed(stride))


def _tensor_metadata(
    dtype: str, shape: tuple[int, ...], value_id: int, quantization: dict | None = None
) -> dict:
    return {
        "dtype": dtype,
        "shape": list(shape),
        "stride": _canonical_stride(shape),
        "storage_offset": 0,
        "contiguous": True,
        "runtime_object_id": None,
        "storage_id": value_id,
        "storage_version": 0,
        **({"quantization": quantization} if quantization is not None else {}),
    }


def _cycle(values: list[float], shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    return np.asarray([values[i % len(values)] for i in range(count)], dtype=np.float32).reshape(shape)


def _int8_codes(shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.int64) * 37 + 11) % 256 - 128
    values[0] = -128
    values[-1] = 127
    return values.astype(np.int8).reshape(shape)


def _int4_codes(shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.int64) * 7 + 3) % 16 - 8
    values[0] = -8
    values[-1] = 7
    return values.astype(np.int8).reshape(shape)


def _uint4_codes(shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.int64) * 5 + 2) % 16
    values[0] = 0
    values[-1] = 15
    return values.astype(np.uint8).reshape(shape)


def _nf4_indices(shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    return (np.arange(count, dtype=np.int64) % 16).astype(np.uint8).reshape(shape)


def _quantization_cases() -> list[QuantizationCase]:
    """Deterministic coverage of storage dtypes, granularities, and tails."""

    return [
        QuantizationCase(
            "int8-per-tensor", "affine-int8/v1", "per_tensor", (2, 5),
            _int8_codes((2, 5)), np.float32(0.375), np.int8(3),
        ),
        QuantizationCase(
            "int8-per-axis", "affine-int8/v1", "per_axis", (3, 7),
            _int8_codes((3, 7)), _cycle([0.5, 0.25, 0.125], (3,)),
            np.asarray([-1, 3, 7], dtype=np.int8), axis=0,
        ),
        QuantizationCase(
            "int8-per-group", "affine-int8/v1", "per_group", (4, 10),
            _int8_codes((4, 10)),
            _cycle([0.5, 0.25, 0.125, 0.75], (4, 4)),
            np.asarray(
                [-3, 2, 5, -7, 4, -1, 6, 0, 7, -5, 3, 1, -8, 5, 2, 4], dtype=np.int8
            ).reshape(4, 4),
            axis=1, group_size=3,
        ),
        QuantizationCase(
            "int4-per-tensor-odd-tail", "affine-int4/v1", "per_tensor", (3, 5),
            _int4_codes((3, 5)), np.float32(0.25), np.int8(-2),
        ),
        QuantizationCase(
            "int4-per-axis", "affine-int4/v1", "per_axis", (5, 3),
            _int4_codes((5, 3)), _cycle([0.5, 0.25, 0.125, 0.75, 0.5], (5,)),
            np.asarray([-8, -4, 0, 3, 7], dtype=np.int8), axis=0,
        ),
        QuantizationCase(
            "int4-per-group", "affine-int4/v1", "per_group", (4, 7),
            _int4_codes((4, 7)),
            _cycle([0.5, 0.25, 0.125], (4, 3)),
            np.asarray(
                [-6, -2, 3, 1, 5, -4, 7, 0, -8, 4, 2, -1], dtype=np.int8
            ).reshape(4, 3),
            axis=1, group_size=3,
        ),
        QuantizationCase(
            "uint4-per-tensor-odd-tail", "affine-uint4/v1", "per_tensor", (3, 5),
            _uint4_codes((3, 5)), np.float32(0.75), np.uint8(7),
        ),
        QuantizationCase(
            "uint4-per-axis", "affine-uint4/v1", "per_axis", (4, 5),
            _uint4_codes((4, 5)), _cycle([0.5, 0.25, 0.125, 0.75, 1.5], (5,)),
            np.asarray([0, 4, 9, 12, 15], dtype=np.uint8), axis=1,
        ),
        QuantizationCase(
            "uint4-per-group", "affine-uint4/v1", "per_group", (5, 3),
            _uint4_codes((5, 3)),
            _cycle([0.5, 0.25, 0.125, 0.75, 1.5, 0.375, 2.0, 0.625, 0.875], (3, 3)),
            np.asarray(
                [1, 5, 9, 13, 2, 6, 10, 14, 3], dtype=np.uint8
            ).reshape(3, 3),
            axis=0, group_size=2,
        ),
        QuantizationCase(
            "nf4-per-block", "nf4/v1", "per_block", (5, 7),
            _nf4_indices((5, 7)),
            np.asarray([0.5, 0.0, 1.25], dtype=np.float32),
            group_size=16, codebook=np.asarray(NF4_CODEBOOK, dtype=np.float32),
        ),
        QuantizationCase(
            "nf4-per-block-tail", "nf4/v1", "per_block", (3, 3),
            _nf4_indices((3, 3)),
            np.asarray([0.25, 0.5, 1.5], dtype=np.float32),
            group_size=4, codebook=np.asarray(NF4_CODEBOOK, dtype=np.float32),
        ),
        QuantizationCase(
            "nf4-blocks-cross-rows", "nf4/v1", "per_block", (4, 8),
            _nf4_indices((4, 8)),
            np.asarray([0.125, 0.5, 0.0, 0.25, 0.75, 0.5, 0.25], dtype=np.float32),
            group_size=5, codebook=np.asarray(NF4_CODEBOOK, dtype=np.float32),
        ),
    ]


def _write_quantization_case(root: Path, case: QuantizationCase) -> dict:
    nf4 = case.scheme == "nf4/v1"
    storage = _AFFINE_STORAGE.get(case.scheme, "uint4")
    zero_id = _ZERO_ID if case.zero is not None else None
    spec = QuantizationSpec(
        logical_shape=case.shape,
        scale_value_id=_SCALE_ID,
        zero_point_value_id=zero_id,
        granularity=case.granularity,
        axis=case.axis,
        group_size=case.group_size,
        scheme=case.scheme,
        storage_dtype=storage,
        rounding="none" if nf4 else "nearest_even",
        saturation="none" if nf4 else "clamp",
        codebook_value_id=_CODEBOOK_ID if nf4 else None,
        codebook="bitsandbytes-nf4/v1" if nf4 else None,
    )
    descriptor = spec.to_dict()
    directory = root / "cases" / case.name
    weight_path = directory / "inputs" / "weight.irtensor"
    codec.write_array(weight_path, case.codes, dtype=storage)

    parameter_values = {}
    scale_path = directory / "quantization_parameters" / f"v{_SCALE_ID}.irtensor"
    codec.write_array(scale_path, np.asarray(case.scale, dtype=np.float32))
    parameter_values[_SCALE_ID] = codec.read(scale_path)
    parameter_records = [
        (_SCALE_ID, scale_path, "float32", case.scale.shape),
    ]
    if case.zero is not None:
        zero_dtype = spec.zero_point_dtype
        zero_path = directory / "quantization_parameters" / f"v{_ZERO_ID}.irtensor"
        codec.write_array(zero_path, np.asarray(case.zero, dtype=zero_dtype))
        parameter_values[_ZERO_ID] = codec.read(zero_path)
        parameter_records.append((_ZERO_ID, zero_path, zero_dtype, case.zero.shape))
    if nf4:
        assert case.codebook is not None
        codebook_path = directory / "quantization_parameters" / f"v{_CODEBOOK_ID}.irtensor"
        codec.write_array(codebook_path, np.asarray(case.codebook, dtype=np.float32))
        parameter_values[_CODEBOOK_ID] = codec.read(codebook_path)
        parameter_records.append((_CODEBOOK_ID, codebook_path, "float32", (16,)))

    expected = np.asarray(
        dequantize(codec.read(weight_path), spec, values=parameter_values),
        dtype=np.float32,
    )
    assert expected.shape == case.shape, (case.name, expected.shape)
    reference_path = directory / "reference" / "y.irtensor"
    codec.write_array(reference_path, expected)

    input_entry = {
        "name": "weight",
        "value_id": _WEIGHT_ID,
        "payload": "inputs/weight.irtensor",
        "role": "parameter",
        "qualified_name": "weight",
        **_tensor_metadata(storage, case.shape, _WEIGHT_ID, descriptor),
    }
    output_metadata = _tensor_metadata("float32", case.shape, _OUTPUT_ID)
    output_entry = {
        "name": "y",
        "value_id": _OUTPUT_ID,
        "payload": "reference/y.irtensor",
        **output_metadata,
    }
    value_records = [
        {
            "id": _WEIGHT_ID,
            **_tensor_metadata(storage, case.shape, _WEIGHT_ID, descriptor),
            "role": "parameter",
            "qualified_name": "weight",
            "producer": None,
        },
        {
            "id": _OUTPUT_ID,
            **output_metadata,
            "role": "output",
            "qualified_name": "y",
            "producer": None,
        },
    ]
    parameters = []
    for value_id, path, dtype, shape in parameter_records:
        relative = path.relative_to(directory).as_posix()
        parameters.append(
            {
                "value_id": value_id,
                "name": f"v{value_id}",
                "payload": relative,
                **_tensor_metadata(dtype, tuple(shape), value_id),
            }
        )
        value_records.append(
            {
                "id": value_id,
                **_tensor_metadata(dtype, tuple(shape), value_id),
                "role": "parameter",
                "qualified_name": f"v{value_id}",
                "producer": None,
            }
        )

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.4",
        "name": case.name,
        "origin": {"region": "quantization/reference/v1"},
        "reproducible": True,
        "inputs": [input_entry],
        "outputs": [output_entry],
        "nodes": [],
        "values": value_records,
        "quantization_parameters": parameters,
    }
    manifest["requirements"] = derive_requirements(manifest)
    (directory / "testcase.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validate_testcase(directory).raise_for_errors()

    bits = [
        f"0x{int(value):08x}"
        for value in np.ascontiguousarray(expected).reshape(-1).view(np.uint32)
    ]
    return {
        "name": case.name,
        "testcase": f"cases/{case.name}",
        "input": "weight",
        "scheme": case.scheme,
        "granularity": case.granularity,
        "shape": list(case.shape),
        "bits": bits,
    }


def generate_quantization_goldens(root: Path) -> Path:
    """Write deterministic format-0.4 testcases and their float32 bit results."""

    root.mkdir(parents=True, exist_ok=True)
    cases = [_write_quantization_case(root, case) for case in _quantization_cases()]
    manifest = {
        "format": QUANTIZATION_GOLDEN_FORMAT,
        "format_version": QUANTIZATION_GOLDEN_VERSION,
        "cases": cases,
    }
    (root / "goldens.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="directory for the generated fixtures")
    parser.add_argument("--fp8-selftest", type=Path, help="run C++ FP8 reader and verify its writes")
    parser.add_argument("--packed-selftest", type=Path, help="run C++ packed reader and verify its writes")
    parser.add_argument(
        "--quantization-goldens",
        type=Path,
        help="write testcase 0.4 quantization goldens to this directory",
    )
    parser.add_argument(
        "--quantization-selftest",
        type=Path,
        help="run the C++ quantization self-test against the generated goldens",
    )
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

    for dtype, carrier in (("int4", "int8"), ("uint4", "uint8")):
        values = np.array([value for byte in range(256) for value in (byte % 16, byte // 16)], dtype=np.uint8)
        if dtype == "int4":
            values = values.astype(np.int8)
            values[values >= 8] -= 16
        for label, codes in (("pairs", values), ("odd", values[:31]),
                             ("scalar", np.array(7, dtype=carrier)),
                             ("empty", np.empty((2, 0), dtype=carrier))):
            source = codec.write_array(output / f"{dtype}.{label}.irtensor", codes, dtype=dtype)
            if args.packed_selftest:
                target = output / f"{dtype}.{label}.roundtrip.irtensor"
                subprocess.run([str(args.packed_selftest.resolve()), str(source.resolve()), str(target.resolve())], check=True)
                if source.read_bytes() != target.read_bytes():
                    raise RuntimeError(f"C++ packed roundtrip changed wire bytes: {dtype}/{label}")

    if args.quantization_goldens or args.quantization_selftest:
        quantization_root = args.quantization_goldens or (output / "quantization")
        generate_quantization_goldens(quantization_root)
        if args.quantization_selftest:
            subprocess.run(
                [str(args.quantization_selftest.resolve()), str(quantization_root.resolve())],
                check=True,
            )

    print(f"wrote golden fixtures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
