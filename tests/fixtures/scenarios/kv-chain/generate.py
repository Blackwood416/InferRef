"""Generate the committed kv-chain scenario fixture (numpy only, no torch).

The chain is a prefill followed by two decode steps:

    state0 = prefill_kv                 (1, 2, 4, 8)
    state1 = concat(state0, prefill_tokens)          (1, 2, 6, 8)
    state2 = concat(state1, decode_tokens_0)         (1, 2, 7, 8)
    state3 = concat(state2, decode_tokens_1)         (1, 2, 8, 8)

Every testcase embeds the exact tensor its step will receive as ``cache``,
so the effective testcase after input binding remains self-consistent and the
reference outputs are the same tensors the scenario's state chain produces.
``scale`` on the prefill testcase is intentionally left unbound so tests can
verify unbound input fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements

ROOT = Path(__file__).resolve().parent


def _kv(batch: int, heads: int, seq: int, width: int, offset: int) -> np.ndarray:
    count = batch * heads * seq * width
    return (
        np.arange(count, dtype=np.float32).reshape(batch, heads, seq, width) + offset
    )


def _logits(update: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (update * scale).sum(axis=2, keepdims=True)


def _write_testcase(
    name: str,
    cache: np.ndarray,
    update: np.ndarray,
    cache_out: np.ndarray,
    *,
    scale: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> None:
    case = ROOT / "cases" / name
    (case / "inputs").mkdir(parents=True, exist_ok=True)
    (case / "reference").mkdir(parents=True, exist_ok=True)
    cache_path = codec.write_array(case / "inputs" / "cache.irtensor", cache)
    update_path = codec.write_array(case / "inputs" / "update.irtensor", update)
    out_path = codec.write_array(case / "reference" / "cache_out.irtensor", cache_out)
    metadata = lambda path: codec.read(path).to_metadata()
    inputs = [
        {
            "name": "cache",
            "value_id": 1,
            "payload": "inputs/cache.irtensor",
            **metadata(cache_path),
        },
        {
            "name": "update",
            "value_id": 2,
            "payload": "inputs/update.irtensor",
            **metadata(update_path),
        },
    ]
    values = [
        {"id": 1, **metadata(cache_path)},
        {"id": 2, **metadata(update_path)},
    ]
    outputs = [
        {
            "name": "cache_out",
            "value_id": 3,
            "payload": "reference/cache_out.irtensor",
            **metadata(out_path),
        }
    ]
    values.append({"id": 3, **metadata(out_path)})
    if scale is not None and logits is not None:
        scale_path = codec.write_array(case / "inputs" / "scale.irtensor", scale)
        logits_path = codec.write_array(case / "reference" / "logits.irtensor", logits)
        inputs.append(
            {
                "name": "scale",
                "value_id": 4,
                "payload": "inputs/scale.irtensor",
                **metadata(scale_path),
            }
        )
        outputs.append(
            {
                "name": "logits",
                "value_id": 5,
                "payload": "reference/logits.irtensor",
                **metadata(logits_path),
            }
        )
        values.extend(
            [
                {"id": 4, **metadata(scale_path)},
                {"id": 5, **metadata(logits_path)},
            ]
        )
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": f"kv-chain-{name}",
        "origin": {"generator": "tests/fixtures/scenarios/kv-chain/generate.py"},
        "reproducible": True,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": [],
        "values": values,
    }
    manifest["requirements"] = derive_requirements(manifest)
    (case / "testcase.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    batch, heads, width = 1, 2, 8
    prefill_kv = _kv(batch, heads, 4, width, 0)
    prefill_tokens = _kv(batch, heads, 2, width, 100)
    decode_tokens_0 = _kv(batch, heads, 1, width, 200)
    decode_tokens_1 = _kv(batch, heads, 1, width, 300)
    scale = np.array([1.0], dtype=np.float32)

    state1 = np.concatenate([prefill_kv, prefill_tokens], axis=2)
    state2 = np.concatenate([state1, decode_tokens_0], axis=2)
    state3 = np.concatenate([state2, decode_tokens_1], axis=2)

    inputs = {
        "prefill_kv": prefill_kv,
        "prefill_tokens": prefill_tokens,
        "decode_tokens_0": decode_tokens_0,
        "decode_tokens_1": decode_tokens_1,
    }
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    for name, tensor in inputs.items():
        codec.write_array(ROOT / "inputs" / f"{name}.irtensor", tensor)

    _write_testcase(
        "prefill",
        prefill_kv,
        prefill_tokens,
        state1,
        scale=scale,
        logits=_logits(prefill_tokens, scale),
    )
    _write_testcase("decode-0", state1, decode_tokens_0, state2)
    _write_testcase("decode-1", state2, decode_tokens_1, state3)

    scenario = {
        "format": "inferref-scenario",
        "format_version": "0.1",
        "id": "kv-chain",
        "description": "KV cache prefill followed by two decode steps",
        "inputs": {name: {"kind": "tensor"} for name in inputs},
        "state": {"kv": {"kind": "tensor", "init": "scenario.inputs.prefill_kv"}},
        "outputs": {"logits": {"kind": "tensor"}},
        "steps": [
            {
                "id": "prefill",
                "testcase": "cases/prefill",
                "bindings": {
                    "inputs": {
                        "cache": "scenario.inputs.prefill_kv",
                        "update": "scenario.inputs.prefill_tokens",
                    },
                    "outputs": {
                        "cache_out": "state.kv",
                        "logits": "scenario.outputs.logits",
                    },
                },
            },
            {
                "id": "decode-0",
                "testcase": "cases/decode-0",
                "bindings": {
                    "inputs": {
                        "cache": "state.kv",
                        "update": "scenario.inputs.decode_tokens_0",
                    },
                    "outputs": {"cache_out": "state.kv"},
                },
            },
            {
                "id": "decode-1",
                "testcase": "cases/decode-1",
                "bindings": {
                    "inputs": {
                        "cache": "state.kv",
                        "update": "scenario.inputs.decode_tokens_1",
                    },
                    "outputs": {"cache_out": "state.kv"},
                },
            },
        ],
    }
    (ROOT / "scenario.json").write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote kv-chain fixture under {ROOT}")


if __name__ == "__main__":
    main()
