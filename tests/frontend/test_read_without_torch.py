"""Trace IR v0.1 acceptance criterion 10 (IR §57).

    a trace readable without PyTorch installed

This is the criterion that makes InferRef useful to an engine team, so it is
tested for real: a subprocess installs an import hook that makes ``import
torch`` raise, then loads a trace, decodes payloads, and runs a comparison.

A plain ``import`` check would not be enough — the risk is a lazily imported
torch somewhere in the read path, which only a hard block catches.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import inferref

# Executed in a subprocess with torch blocked. Kept as a script rather than a
# fixture because the block must be installed before InferRef is imported.
_BLOCKER = '''
import sys


class TorchBlocker:
    """Make any attempt to import torch fail loudly."""

    FORBIDDEN = ("torch", "torchvision", "functorch")

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in self.FORBIDDEN:
            raise ImportError(f"torch is blocked for this test (tried to import {name!r})")
        return None


for name in list(sys.modules):
    if name.split(".")[0] in TorchBlocker.FORBIDDEN:
        del sys.modules[name]
sys.meta_path.insert(0, TorchBlocker())

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("blocker failed: torch was importable")
'''


def _run_without_torch(body: str, trace_dir: Path) -> str:
    script = _BLOCKER + textwrap.dedent(body).replace("__TRACE__", trace_dir.as_posix())
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed ({result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "torch" not in result.stdout.lower() or "blocked" not in result.stdout
    return result.stdout


@pytest.fixture
def traced_dir(trace_dir: Path, mini_llama, mini_llama_input) -> Path:
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="all", model_name="MiniLlama"
    ) as session:
        session.mark_input("hidden_states", mini_llama_input)
        session.mark_output("out", mini_llama(mini_llama_input))
    return trace_dir


def test_import_inferref_without_torch(traced_dir: Path) -> None:
    out = _run_without_torch(
        """
        import inferref
        print("version", inferref.__version__)
        """,
        traced_dir,
    )
    from inferref.ir.version import INFERREF_VERSION

    assert f"version {INFERREF_VERSION}" in out


def test_load_trace_without_torch(traced_dir: Path) -> None:
    out = _run_without_torch(
        """
        from inferref.ir.package import TracePackage

        pkg = TracePackage.load("__TRACE__")
        print("operators", len(pkg.graph.operators))
        print("values", len(pkg.graph.values))
        print("modules", len(pkg.modules))
        print("sources", len(pkg.sources))
        assert pkg.manifest.format == "inferref-trace"
        assert pkg.manifest.format_version == "0.1"
        # Module paths and source locations survive without the framework.
        op = pkg.graph.ops_in_execution_order()[10]
        print("module", pkg.module_path(op.module_stack))
        print("source", pkg.source(op.source_id))
        """,
        traced_dir,
    )
    assert "operators" in out
    assert "layers.0" in out
    assert "model.py" in out


def test_validate_without_torch(traced_dir: Path) -> None:
    out = _run_without_torch(
        """
        from inferref.ir.package import TracePackage
        from inferref.ir.validate import validate_package

        pkg = TracePackage.load("__TRACE__")
        issues = [i for i in validate_package(pkg) if i.severity == "error"]
        assert not issues, issues
        print("validated ok")
        """,
        traced_dir,
    )
    assert "validated ok" in out


def test_decode_payload_without_torch(traced_dir: Path) -> None:
    out = _run_without_torch(
        """
        import numpy as np
        from inferref.ir.package import TracePackage
        from inferref.tensor import codec

        pkg = TracePackage.load("__TRACE__")
        decoded = 0
        nonfinite = 0
        for value in pkg.graph.values:
            if value.capture.mode == "full" and value.capture.payload:
                view = codec.read(pkg.tensor_payload_path(value.capture.payload))
                assert tuple(view.shape) == tuple(value.shape)
                assert view.dtype == value.dtype
                assert view.data.size == value.logical_numel
                array = view.as_comparable()
                if not np.isfinite(array).all():
                    nonfinite += 1
                decoded += 1
        print("decoded", decoded)
        print("tensors with non-finite values", nonfinite)
        assert decoded > 0
        # The causal attention mask is built from -inf, so it must survive the
        # binary round-trip rather than being clamped or corrupted.
        assert nonfinite > 0
        """,
        traced_dir,
    )
    assert "decoded" in out
    assert "tensors with non-finite values" in out


def test_extract_and_compare_without_torch(traced_dir: Path) -> None:
    """The full engine-side loop: extract, recompute in numpy, compare."""
    out = _run_without_torch(
        """
        import json
        import numpy as np
        from inferref.compare.compare import compare_testcase
        from inferref.ir.package import TracePackage
        from inferref.region.manager import create_region_from_ops
        from inferref.tensor import codec
        from inferref.testcase.extract import extract_operator

        pkg = TracePackage.load("__TRACE__")
        mm = next(
            op for op in pkg.graph.ops_in_execution_order()
            if op.canonical_name == "aten.mm.default"
        )
        extract_operator(pkg, mm.id, "__TRACE__/../tc", input_names=["a", "b"])

        manifest = json.load(open("__TRACE__/../tc/testcase.json"))
        base = "__TRACE__/../tc/"
        a = codec.read(base + manifest["inputs"][0]["payload"]).as_comparable()
        b = codec.read(base + manifest["inputs"][1]["payload"]).as_comparable()

        out_name = manifest["outputs"][0]["name"]
        codec.write_array(
            "__TRACE__/../engine/" + out_name + ".irtensor", (a @ b).astype(np.float32)
        )

        report = compare_testcase("__TRACE__/../tc", "__TRACE__/../engine")
        print("status", report.status)
        print("passed", report.passed_count)
        assert report.status == "pass", report.to_dict()

        # And a region works too.
        region = create_region_from_ops(pkg, "Head", 1, 8)
        print("region inputs", len(region.inputs), "outputs", len(region.outputs))
        """,
        traced_dir,
    )
    assert "status pass" in out
    assert "region inputs" in out


def test_cli_inspect_without_torch(traced_dir: Path) -> None:
    """The CLI itself must not import torch for read-only commands."""
    out = _run_without_torch(
        """
        from inferref.cli.main import main

        code = main(["inspect", "__TRACE__", "--limit", "5"])
        assert code == 0, code
        code = main(["analyze", "__TRACE__"])
        assert code == 0, code
        code = main(["validate", "__TRACE__"])
        assert code == 0, code
        print("cli ok")
        """,
        traced_dir,
    )
    assert "cli ok" in out
    assert "MiniLlama" in out
