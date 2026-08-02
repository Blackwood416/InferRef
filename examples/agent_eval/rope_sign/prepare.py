"""Build the trusted, deterministic testcase for the RoPE Agent evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from inferref.tensor import codec


def _rotate_half(value: np.ndarray) -> np.ndarray:
    half = value.shape[-1] // 2
    first = value[..., :half]
    second = value[..., half:]
    return np.concatenate((-second, first), axis=-1)


def _apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return value * cos[None, None, :, :] + _rotate_half(value) * sin[None, None, :, :]


def build_testcase(output: str | Path) -> Path:
    """Create one small reproducible testcase and refuse to overwrite artifacts."""

    root = Path(output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"evaluation testcase directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260802)
    query = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    key = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    angles = np.linspace(0.0, 1.25, num=16, dtype=np.float32).reshape(4, 4)
    cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1).astype(np.float32)
    sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1).astype(np.float32)

    inputs = {"query": query, "key": key, "cos": cos, "sin": sin}
    references = {
        "q_embed": _apply_rope(query, cos, sin),
        "k_embed": _apply_rope(key, cos, sin),
    }
    manifest_inputs = []
    for value_id, (name, value) in enumerate(inputs.items(), start=1):
        relative = f"inputs/{name}.irtensor"
        codec.write_array(root / relative, value)
        manifest_inputs.append(
            {"name": name, "value_id": value_id, "payload": relative}
        )

    producer = {
        "op_id": 17,
        "execution_index": 17,
        "canonical_name": "aten.add.Tensor",
        "module_path": "layers.0.self_attn",
        "source": "reference_model.py:42 in apply_rotary_pos_emb",
        "region": "RoPE@agent-eval",
    }
    manifest_outputs = []
    for value_id, (name, value) in enumerate(references.items(), start=101):
        relative = f"reference/{name}.irtensor"
        codec.write_array(root / relative, value.astype(np.float32))
        manifest_outputs.append(
            {
                "name": name,
                "value_id": value_id,
                "payload": relative,
                "producer": producer,
            }
        )

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.1",
        "name": "agent-eval-rope-half-rotation",
        "reproducible": True,
        "origin": {
            "kind": "agent_evaluation",
            "benchmark": "rope-half-rotation-sign",
        },
        "inputs": manifest_inputs,
        "outputs": manifest_outputs,
        "nodes": [],
        "values": [],
    }
    (root / "testcase.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root


def setup_workspace(output: str | Path) -> Path:
    """Create the Agent-visible workspace without copying the trusted generator."""

    root = Path(output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"evaluation workspace is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    for name in ("engine.py", "adapter.json", "TASK.md"):
        shutil.copy2(source / name, root / name)
    build_testcase(root / "testcase")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", help="write only the trusted testcase")
    destination.add_argument(
        "--workspace",
        help="copy the candidate workspace and generate its testcase",
    )
    args = parser.parse_args(argv)
    if args.workspace:
        root = setup_workspace(args.workspace)
        print(f"Wrote Agent evaluation workspace to {root}")
    else:
        root = build_testcase(args.output)
        print(f"Wrote Agent evaluation testcase to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
