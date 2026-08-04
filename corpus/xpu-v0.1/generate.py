"""Generate InferRef's tiny, weight-free XPU/SYCL validation corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements

ROOT = Path(__file__).resolve().parent


def _quantize(value: np.ndarray, dtype: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if dtype == "float32":
        return value
    if dtype == "float16":
        return value.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        bits = value.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        return ((rounded >> 16) << 16).astype(np.uint32).view(np.float32)
    raise ValueError(dtype)


def _write(path: Path, value: np.ndarray, dtype: str) -> tuple[Path, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    if dtype == "float32":
        written = codec.write_array(path, value.astype(np.float32))
    elif dtype == "float16":
        written = codec.write_array(path, value.astype(np.float16))
    elif dtype == "bfloat16":
        bits = value.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        payload = (rounded >> 16).astype("<u2").tobytes()
        shape = value.shape
        stride = tuple(np.cumprod((1,) + shape[:0:-1], dtype=np.int64)[::-1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(codec.encode(dtype="bfloat16", shape=shape, stride=stride, payload=payload))
        written = path
    else:
        raise ValueError(dtype)
    return written, codec.read(written).as_comparable().astype(np.float32)


def _case(name: str, inputs: dict[str, tuple[np.ndarray, str]], outputs: dict[str, tuple[np.ndarray, str]], tags: list[str]) -> dict:
    root = ROOT / "cases" / name
    records_in, records_out, values = [], [], []
    value_id = 1
    for label, (array, dtype) in inputs.items():
        path, _ = _write(root / "inputs" / f"{label}.irtensor", array, dtype)
        meta = codec.read(path).to_metadata()
        records_in.append({"name": label, "value_id": value_id, "payload": f"inputs/{label}.irtensor", **meta})
        values.append({"id": value_id, **meta}); value_id += 1
    for label, (array, dtype) in outputs.items():
        path, _ = _write(root / "reference" / f"{label}.irtensor", array, dtype)
        meta = codec.read(path).to_metadata()
        records_out.append({"name": label, "value_id": value_id, "payload": f"reference/{label}.irtensor", **meta})
        values.append({"id": value_id, **meta}); value_id += 1
    manifest = {"format": "inferref-testcase", "format_version": "0.2", "name": name, "origin": {"generator": "corpus/xpu-v0.1/generate.py", "seed": 20260804}, "reproducible": True, "inputs": records_in, "outputs": records_out, "nodes": [], "values": values}
    manifest["requirements"] = derive_requirements(manifest)
    (root / "testcase.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"id": name, "testcase": f"cases/{name}", "tags": tags}


def generate() -> None:
    rng = np.random.default_rng(20260804)
    cases = []
    for dtype in ("float32", "float16", "bfloat16"):
        x = _quantize(rng.normal(size=(3, 16)), dtype)
        w = _quantize(rng.normal(size=(16,)), dtype)
        eps = np.array(1e-5, dtype=np.float32)
        y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w
        name = f"rmsnorm-{dtype}"
        cases.append(_case(name, {"x": (x, dtype), "weight": (w, dtype), "epsilon": (eps, "float32")}, {"y": (y, dtype)}, ["rmsnorm", dtype, "prefill"]))
    for dim in (4, 12):
        query = rng.normal(size=(2, 3, 5, dim)).astype(np.float32)
        key = rng.normal(size=(2, 2, 5, dim)).astype(np.float32)
        angle = rng.normal(size=(5, dim)).astype(np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        def rope(x):
            half = dim // 2
            rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
            return x * cos[None, None] + rotated * sin[None, None]
        name = f"rope-dim{dim}"
        cases.append(_case(name, {"query": (query, "float32"), "key": (key, "float32"), "cos": (cos, "float32"), "sin": (sin, "float32")}, {"q_embed": (rope(query), "float32"), "k_embed": (rope(key), "float32")}, ["rope", "decode", f"head-dim-{dim}"]))
    cache = rng.normal(size=(1, 2, 3, 8)).astype(np.float32)
    update = rng.normal(size=(1, 2, 1, 8)).astype(np.float32)
    cases.append(_case("kv-append", {"cache": (cache, "float32"), "update": (update, "float32")}, {"cache_out": (np.concatenate((cache, update), axis=-2), "float32")}, ["kv-cache", "decode", "append"]))
    indexed = cache.copy(); indexed[:, :, 1:2, :] = update
    cases.append(_case("kv-index", {"cache": (cache, "float32"), "update": (update, "float32"), "index": (np.array(1, dtype=np.float32), "float32")}, {"cache_out": (indexed, "float32")}, ["kv-cache", "decode", "indexed-update"]))
    packed = rng.normal(size=(1, 2, 2, 3, 8)).astype(np.float32)
    packed_update = rng.normal(size=(1, 2, 2, 1, 8)).astype(np.float32)
    cases.append(_case("kv-packed", {"cache": (packed, "float32"), "update": (packed_update, "float32")}, {"cache_out": (np.concatenate((packed, packed_update), axis=-2), "float32")}, ["kv-cache", "prefill", "packed"]))

    # Weight-free model-derived TraceSet slices.  Shapes mirror tiny Llama and
    # Qwen3.5 prefill/decode contracts without distributing model parameters.
    llama_x = _quantize(rng.normal(size=(1, 7, 32)), "float16")
    llama_w = _quantize(rng.normal(size=(32,)), "float16")
    llama_eps = np.array(1e-5, dtype=np.float32)
    llama_y = llama_x / np.sqrt(np.mean(llama_x * llama_x, axis=-1, keepdims=True) + llama_eps) * llama_w
    cases.append(_case(
        "rmsnorm-tiny-llama-prefill",
        {"x": (llama_x, "float16"), "weight": (llama_w, "float16"), "epsilon": (llama_eps, "float32")},
        {"y": (llama_y, "float16")},
        ["rmsnorm", "tiny-llama", "prefill", "model-derived"],
    ))

    llama_dim = 16
    llama_q = _quantize(rng.normal(size=(1, 4, 1, llama_dim)), "float16")
    llama_k = _quantize(rng.normal(size=(1, 2, 1, llama_dim)), "float16")
    llama_angle = _quantize(rng.normal(size=(1, llama_dim)), "float16")
    llama_cos = _quantize(np.cos(llama_angle), "float16")
    llama_sin = _quantize(np.sin(llama_angle), "float16")

    def llama_rope(value: np.ndarray) -> np.ndarray:
        half = llama_dim // 2
        rotated = np.concatenate((-value[..., half:], value[..., :half]), axis=-1)
        return value * llama_cos[None, None] + rotated * llama_sin[None, None]

    cases.append(_case(
        "rope-tiny-llama-decode",
        {"query": (llama_q, "float16"), "key": (llama_k, "float16"), "cos": (llama_cos, "float16"), "sin": (llama_sin, "float16")},
        {"q_embed": (llama_rope(llama_q), "float16"), "k_embed": (llama_rope(llama_k), "float16")},
        ["rope", "tiny-llama", "decode", "model-derived"],
    ))

    qwen_prefill = rng.normal(size=(1, 2, 2, 4, 8)).astype(np.float32)
    qwen_prefill_update = rng.normal(size=(1, 2, 2, 2, 8)).astype(np.float32)
    cases.append(_case(
        "kv-qwen35-prefill",
        {"cache": (qwen_prefill, "float32"), "update": (qwen_prefill_update, "float32")},
        {"cache_out": (np.concatenate((qwen_prefill, qwen_prefill_update), axis=-2), "float32")},
        ["kv-cache", "tiny-qwen3.5", "prefill", "hybrid", "model-derived"],
    ))
    qwen_decode = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    for steps, suffix in ((1, "step"), (3, "multistep")):
        qwen_update = rng.normal(size=(1, 2, steps, 8)).astype(np.float32)
        cases.append(_case(
            f"kv-qwen35-decode-{suffix}",
            {"cache": (qwen_decode, "float32"), "update": (qwen_update, "float32")},
            {"cache_out": (np.concatenate((qwen_decode, qwen_update), axis=-2), "float32")},
            ["kv-cache", "tiny-qwen3.5", "decode", suffix, "model-derived"],
        ))
    suite = {"format": "inferref-suite", "format_version": "0.1", "name": "inferref-xpu-v0.1", "cases": cases}
    (ROOT / "suite.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
