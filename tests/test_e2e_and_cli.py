"""End-to-end validation loop (SPEC §65).

    reference trace
       -> extract testcase
       -> external implementation
       -> inferref compare
       -> PASS / FAIL

Plus CLI coverage: every command in SPEC §32 runs, and every one supports
``--json`` for agent and CI consumption (SPEC §42).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import inferref
from inferref.cli.main import main
from inferref.ir.package import TracePackage

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def traced_dir(trace_dir: Path, mini_llama, mini_llama_input) -> Path:
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="all", model_name="MiniLlama", seed=0
    ) as session:
        session.mark_input("hidden_states", mini_llama_input)
        session.mark_output("out", mini_llama(mini_llama_input))
    return trace_dir


def _run_cli(*args: str) -> int:
    return main([str(a) for a in args])


def _json_cli(capsys, *args: str, expect: int = 0):
    """Run a CLI command with --json and parse its output.

    ``expect`` is the exit code the command should return; ``compare`` uses a
    non-zero exit to signal FAIL while still emitting a full JSON report.
    """
    assert _run_cli(*args, "--json") == expect
    return json.loads(capsys.readouterr().out)


# -- the SPEC §65 loop -----------------------------------------------------


def test_full_rope_validation_loop(traced_dir: Path, tmp_path: Path, capsys) -> None:
    # 1. Find the RoPE operators and carve out layer 0's invocation.
    package = TracePackage.load(traced_dir)
    rope_ops = [
        op
        for op in package.graph.ops_in_execution_order()
        if (src := package.source(op.source_id)) is not None
        and src.primary is not None
        and src.primary.function in ("apply_rotary_pos_emb", "rotate_half")
    ]
    assert rope_ops
    start = rope_ops[0]
    end = next(op for op in reversed(rope_ops)
               if op.execution_index < start.execution_index + 30)

    assert _run_cli(
        "region", "create", traced_dir,
        "--name", "RotaryEmbedding",
        "--from-op", start.id, "--to-op", end.id,
        "--semantic", "RoPE",
        "--engine-op", "SYCLRotaryEmbeddingKernel",
    ) == 0
    capsys.readouterr()

    # 2. Extract a standalone testcase with meaningful boundary names.
    repro = tmp_path / "repro"
    assert _run_cli(
        "testcase", "extract", traced_dir,
        "--region", "RotaryEmbedding",
        "-o", repro,
        "--input-names", "cos,sin,query,key",
        "--output-names", "q_embed,k_embed",
    ) == 0
    capsys.readouterr()

    assert (repro / "testcase.json").is_file()
    assert (repro / "README.md").is_file()
    for name in ("cos", "sin", "query", "key"):
        assert (repro / "inputs" / f"{name}.irtensor").is_file()
    for name in ("q_embed", "k_embed"):
        assert (repro / "reference" / f"{name}.irtensor").is_file()

    # 3. Run the "engine" — numpy only, no torch, no model source.
    engine_out = tmp_path / "engine-out"
    assert _engine(repro, engine_out) == 0

    # 4. Compare: expect PASS.
    assert _run_cli("compare", repro, engine_out, "--first-failure") == 0
    assert "Result: PASS" in capsys.readouterr().out

    # 5. Now a buggy engine: expect FAIL, localised.
    engine_bad = tmp_path / "engine-bad"
    assert _engine(repro, engine_bad, inject_bug=True) == 0

    assert _run_cli("compare", repro, engine_bad, "--first-failure") == 1
    text = capsys.readouterr().out
    assert "Result: FAIL" in text
    assert "First divergence:" in text
    assert "q_embed" in text
    assert "First mismatching element:" in text


def _engine(testcase: Path, output: Path, *, inject_bug: bool = False) -> int:
    args = [
        sys.executable,
        str(REPO_ROOT / "examples" / "engine_sim" / "rope_numpy.py"),
        str(testcase),
        "--output",
        str(output),
    ]
    if inject_bug:
        args.append("--inject-bug")
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(result.stdout, result.stderr)
    return result.returncode


def test_first_failure_localises_to_the_wrong_output(
    traced_dir: Path, tmp_path: Path, capsys
) -> None:
    """The bug only corrupts q_embed, so k_embed must still pass."""
    package = TracePackage.load(traced_dir)
    rope_ops = [
        op
        for op in package.graph.ops_in_execution_order()
        if (src := package.source(op.source_id)) is not None
        and src.primary is not None
        and src.primary.function in ("apply_rotary_pos_emb", "rotate_half")
    ]
    start = rope_ops[0]
    end = next(op for op in reversed(rope_ops)
               if op.execution_index < start.execution_index + 30)
    _run_cli("region", "create", traced_dir, "--name", "R",
             "--from-op", start.id, "--to-op", end.id)
    repro = tmp_path / "repro"
    _run_cli("testcase", "extract", traced_dir, "--region", "R", "-o", repro,
             "--input-names", "cos,sin,query,key",
             "--output-names", "q_embed,k_embed")
    capsys.readouterr()

    engine_bad = tmp_path / "bad"
    _engine(repro, engine_bad, inject_bug=True)

    payload = _json_cli(capsys, "compare", repro, engine_bad, expect=1)
    assert payload["status"] == "fail"
    by_name = {c["name"]: c for c in payload["comparisons"]}
    assert by_name["q_embed"]["status"] == "fail"
    assert by_name["k_embed"]["status"] == "pass"
    assert payload["first_failure"]["name"] == "q_embed"
    assert payload["first_failure"]["metrics"]["max_abs_error"] > 0.1


def test_testcase_comparison_reports_provenance(
    traced_dir: Path, tmp_path: Path, capsys
) -> None:
    """The engine never saw the model, but the report still names the source.

    Producer context is recorded into testcase.json at extraction time so a
    failure stays actionable without the trace or the model (SPEC §2.6).
    """
    package = TracePackage.load(traced_dir)
    rope_ops = [
        op
        for op in package.graph.ops_in_execution_order()
        if (src := package.source(op.source_id)) is not None
        and src.primary is not None
        and src.primary.function in ("apply_rotary_pos_emb", "rotate_half")
    ]
    start = rope_ops[0]
    end = next(op for op in reversed(rope_ops)
               if op.execution_index < start.execution_index + 30)
    _run_cli("region", "create", traced_dir, "--name", "RotaryEmbedding",
             "--from-op", start.id, "--to-op", end.id, "--semantic", "RoPE")
    repro = tmp_path / "repro"
    _run_cli("testcase", "extract", traced_dir, "--region", "RotaryEmbedding",
             "-o", repro, "--input-names", "cos,sin,query,key",
             "--output-names", "q_embed,k_embed")
    capsys.readouterr()

    # The provenance is in the testcase itself, not only in the trace.
    manifest = json.loads((repro / "testcase.json").read_text(encoding="utf-8"))
    producer = manifest["outputs"][0]["producer"]
    assert producer["canonical_name"].startswith("aten.")
    assert producer["module_path"] == "layers.0.self_attn"
    assert producer["region"] == "RotaryEmbedding"
    assert "apply_rotary_pos_emb" in producer["source"]

    engine_bad = tmp_path / "bad"
    _engine(repro, engine_bad, inject_bug=True)

    assert _run_cli("compare", repro, engine_bad, "--first-failure") == 1
    text = capsys.readouterr().out
    assert "Module:    layers.0.self_attn" in text
    assert "Region:    RotaryEmbedding" in text
    assert "apply_rotary_pos_emb" in text
    assert "Producer:  #" in text


# -- CLI surface -----------------------------------------------------------


def test_cli_inspect(traced_dir: Path, capsys) -> None:
    assert _run_cli("inspect", traced_dir, "--limit", "10") == 0
    text = capsys.readouterr().out
    assert "Model:  MiniLlama" in text
    assert "aten." in text
    assert "layers.0" in text


def test_cli_inspect_json(traced_dir: Path, capsys) -> None:
    payload = _json_cli(capsys, "inspect", traced_dir, "--limit", "5")
    assert payload["model"] == "MiniLlama"
    assert payload["counts"]["operators"] > 0
    assert len(payload["operators"]) == 5
    assert payload["operators"][0]["execution_index"] == 0


def test_cli_inspect_filters(traced_dir: Path, capsys) -> None:
    assert _run_cli("inspect", traced_dir, "--module", "layers.0.mlp") == 0
    text = capsys.readouterr().out
    assert "layers.0.mlp" in text
    assert "layers.1" not in text

    assert _run_cli("inspect", traced_dir, "--operator", "aten.mm.default") == 0
    assert "aten.mm.default" in capsys.readouterr().out


def test_cli_inspect_tensor_detail(traced_dir: Path, capsys) -> None:
    assert _run_cli("inspect", traced_dir, "--tensor", "1") == 0
    text = capsys.readouterr().out
    assert "Tensor t1" in text
    assert "storage:" in text
    assert "consumers:" in text


def test_cli_analyze(traced_dir: Path, capsys) -> None:
    payload = _json_cli(capsys, "analyze", traced_dir)
    assert payload["model"] == "MiniLlama"
    assert payload["totals"]["operators"] > 0
    assert payload["coverage"]["source"] > 0.9
    assert payload["coverage"]["payload"] == pytest.approx(1.0)
    assert payload["signatures"]["total_signatures"] > 0
    # Real models repeat operators; dedup must actually compress.
    assert payload["signatures"]["total_signatures"] < payload["totals"]["operators"]


def test_cli_validate(traced_dir: Path, capsys) -> None:
    payload = _json_cli(capsys, "validate", traced_dir)
    assert payload["status"] == "pass"
    assert payload["errors"] == 0


def test_cli_testcase_dedup(traced_dir: Path, capsys) -> None:
    payload = _json_cli(capsys, "testcase", "dedup", traced_dir,
                        "--operator", "aten.mm.default")
    summary = payload["summary"]
    # 14 mm executions across 2 layers collapse to a handful of signatures.
    assert summary["total_executions"] > summary["total_signatures"]
    assert all(s["operator"] == "aten.mm.default" for s in payload["signatures"])


def test_cli_region_lifecycle(traced_dir: Path, capsys) -> None:
    assert _run_cli("region", "list", traced_dir) == 0
    assert "No regions defined." in capsys.readouterr().out

    assert _run_cli("region", "create", traced_dir, "--name", "Head",
                    "--from-op", "1", "--to-op", "6") == 0
    capsys.readouterr()

    payload = _json_cli(capsys, "region", "list", traced_dir)
    assert [r["name"] for r in payload["regions"]] == ["Head"]

    # Regions persist to regions.json and keep the package valid.
    assert _run_cli("validate", traced_dir) == 0
    capsys.readouterr()

    assert _run_cli("region", "delete", traced_dir, "Head") == 0
    capsys.readouterr()
    payload = _json_cli(capsys, "region", "list", traced_dir)
    assert payload["regions"] == []


def test_cli_region_from_module(traced_dir: Path, capsys) -> None:
    assert _run_cli("region", "create", traced_dir, "--name", "MLP0",
                    "--module", "layers.0.mlp") == 0
    text = capsys.readouterr().out
    assert "Created region" in text

    payload = _json_cli(capsys, "region", "list", traced_dir)
    region = payload["regions"][0]
    assert region["creation"]["method"] == "module"
    assert region["node_ids"]


def test_cli_export(traced_dir: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "export.json"
    assert _run_cli("export", traced_dir, "-o", out) == 0
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["manifest"]["format"] == "inferref-trace"
    assert len(payload["graph"]["operators"]) > 0


def test_cli_reports_usage_errors(traced_dir: Path, tmp_path: Path) -> None:
    assert _run_cli("testcase", "extract", traced_dir, "-o", tmp_path / "x") == 2
    assert _run_cli("testcase", "extract", traced_dir, "--region", "nope",
                    "-o", tmp_path / "x") == 2
    assert _run_cli("inspect", tmp_path / "does-not-exist") == 2
    assert _run_cli("compare", tmp_path, tmp_path) == 2


def test_cli_trace_subcommand(tmp_path: Path, capsys) -> None:
    """`inferref trace script.py` works on a script InferRef does not control."""
    script = tmp_path / "run.py"
    script.write_text(
        "import torch\n"
        "m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()).eval()\n"
        "with torch.no_grad():\n"
        "    m(torch.randn(2, 4))\n",
        encoding="utf-8",
    )
    out = tmp_path / "scripted"
    assert _run_cli("trace", script, "-o", out, "--capture-tensors", "all",
                    "--model-name", "Scripted") == 0
    capsys.readouterr()

    package = TracePackage.load(out)
    assert package.manifest.model.name == "Scripted"
    names = [op.canonical_name for op in package.graph.ops_in_execution_order()]
    assert "aten.addmm.default" in names
    assert "aten.relu.default" in names
    # Module paths were recovered without the script cooperating.
    paths = {package.module_path(op.module_stack) for op in package.graph.operators}
    assert "0" in paths     # Sequential's first child


def test_manifest_records_environment(traced_dir: Path) -> None:
    manifest = TracePackage.load(traced_dir).manifest
    assert manifest.reference_framework.name == "pytorch"
    assert manifest.reference_framework.version == torch.__version__
    assert manifest.environment.python
    assert manifest.environment.packages["torch"] == torch.__version__
    assert manifest.determinism.seed == 0
    assert manifest.determinism.training is False
    assert manifest.capture.tensor_policy == "full"
