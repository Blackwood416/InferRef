"""Semantic detection against a real traced model (SPEC §17, §25, §33).

The core suite covers the detection rules on synthetic packages. This one
checks the result on an actual PyTorch trace, where the thing that matters is
whether the automatic region is the same one a human would have drawn by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import inferref
from inferref.cli.main import main
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.region.manager import create_region_from_ops
from inferref.semantic import apply_detections, detect
from inferref.semantic.invocations import is_contiguous


@pytest.fixture
def traced(trace_dir: Path, mini_llama, mini_llama_input) -> TracePackage:
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="all", model_name="MiniLlama", seed=0
    ) as session:
        session.mark_input("hidden_states", mini_llama_input)
        session.mark_output("out", mini_llama(mini_llama_input))
    return TracePackage.load(trace_dir)


def _by_name(package: TracePackage) -> dict[str, object]:
    return {r.name: r for r in package.regions}


# -- the headline claim ----------------------------------------------------


def test_detected_rope_matches_the_hand_built_region(traced: TracePackage) -> None:
    """Automatic detection reproduces the region a human drew with --from-op.

    This is the whole point of the feature: the operator ids 31..48 that the
    workflow used to require are neither stable nor explicable, and detection
    has to arrive at exactly the same node set without them.
    """
    manual = TracePackage.load(traced.root)
    hand_built = create_region_from_ops(manual, "Manual", 31, 48)

    apply_detections(traced, detect(traced))
    auto = _by_name(traced)["RoPE@layers.0.self_attn"]

    assert set(auto.node_ids) == set(hand_built.node_ids)
    assert len(auto.node_ids) == 18
    assert set(auto.inputs) == set(hand_built.inputs)
    assert set(auto.outputs) == set(hand_built.outputs)
    # Four inputs (query, key, cos, sin) and two outputs (SPEC §4.3).
    assert len(auto.inputs) == 4
    assert len(auto.outputs) == 2


def test_rope_region_is_contiguous(traced: TracePackage) -> None:
    """The stack-matching fix means the region has no holes."""
    apply_detections(traced, detect(traced))
    region = _by_name(traced)["RoPE@layers.0.self_attn"]
    assert is_contiguous(traced.graph, region.node_ids)

    # It really does contain the inlined rotate_half operators (SPEC §51).
    names = {traced.graph.op(n).canonical_name for n in region.node_ids}
    assert {"aten.neg.default", "aten.cat.default", "aten.slice.Tensor"} <= names


# -- structure -------------------------------------------------------------


def test_every_construct_is_found_once_per_invocation(traced: TracePackage) -> None:
    detections = detect(traced)
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.name] = counts.get(detection.name, 0) + 1

    # 2 layers x (q, k, v, o) + 2 layers x (gate, up, down)
    assert counts["Linear"] == 14
    # 2 per layer, plus the final norm
    assert counts["RMSNorm"] == 5
    assert counts["Attention"] == 2
    assert counts["SwiGLU"] == 2
    assert counts["TransformerBlock"] == 2
    assert counts["RepeatKV"] == 2
    # One per layer from apply_rotary_pos_emb, plus the RotaryEmbedding module
    # that builds the cos/sin tables.
    assert counts["RoPE"] == 3


def test_confidence_reflects_the_evidence(traced: TracePackage) -> None:
    """IR §32: a torch built-in is deterministic, a class name is inference."""
    by_name: dict[str, list[float]] = {}
    for detection in detect(traced):
        by_name.setdefault(detection.name, []).append(detection.confidence)

    assert set(by_name["Linear"]) == {1.0}          # torch.nn.Linear
    assert set(by_name["RMSNorm"]) == {0.90}        # custom class name
    assert 0.95 in by_name["RoPE"]                  # source function


def test_regions_nest(traced: TracePackage) -> None:
    """A Linear inside an Attention inside a TransformerBlock (IR §36)."""
    apply_detections(traced, detect(traced))
    regions = _by_name(traced)
    linear = set(regions["Linear@layers.0.self_attn.q_proj"].node_ids)
    attention = set(regions["Attention@layers.0.self_attn"].node_ids)
    block = set(regions["TransformerBlock@layers.0"].node_ids)

    assert linear < attention < block


def test_layer_regions_are_disjoint(traced: TracePackage) -> None:
    apply_detections(traced, detect(traced))
    regions = _by_name(traced)
    first = set(regions["TransformerBlock@layers.0"].node_ids)
    second = set(regions["TransformerBlock@layers.1"].node_ids)
    assert not (first & second)


# -- invariants ------------------------------------------------------------


def test_detection_leaves_the_physical_trace_untouched(traced: TracePackage) -> None:
    """SPEC §68.3: never replace execution truth with inferred semantics."""
    before = [
        (o.id, o.execution_index, o.canonical_name, o.positional_args, o.result)
        for o in traced.graph.ops_in_execution_order()
    ]
    values_before = [(v.id, v.shape, v.stride, v.storage_version) for v in traced.graph.values]

    apply_detections(traced, detect(traced))

    after = [
        (o.id, o.execution_index, o.canonical_name, o.positional_args, o.result)
        for o in traced.graph.ops_in_execution_order()
    ]
    assert before == after
    assert [(v.id, v.shape, v.stride, v.storage_version) for v in traced.graph.values] == (
        values_before
    )


def test_detected_regions_validate(traced: TracePackage) -> None:
    """Derived boundaries must satisfy IR §48 invariant 9."""
    apply_detections(traced, detect(traced))
    traced.save(traced.root)
    reloaded = TracePackage.load(traced.root)
    errors = [i for i in validate_package(reloaded) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_full_semantic_coverage(traced: TracePackage) -> None:
    """Every operator in this model gets at least one label (SPEC §25)."""
    from inferref.inspect.analyze import analyze

    apply_detections(traced, detect(traced))
    result = analyze(traced)
    assert result.semantic_coverage == pytest.approx(1.0)
    assert result.region_coverage == pytest.approx(1.0)
    assert not result.unlabelled_modules
    assert result.semantic_counts["Linear"] > 0


# -- CLI -------------------------------------------------------------------


def test_cli_region_detect(traced: TracePackage, capsys) -> None:
    assert main(["region", "detect", str(traced.root)]) == 0
    text = capsys.readouterr().out
    assert "Detected 30 semantic region(s)" in text
    assert "14x  Linear" in text

    reloaded = TracePackage.load(traced.root)
    assert len(reloaded.regions) == 30
    assert any(r.name == "RoPE@layers.0.self_attn" for r in reloaded.regions)


def test_cli_region_detect_dry_run_writes_nothing(traced: TracePackage, capsys) -> None:
    assert main(["region", "detect", str(traced.root), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "would be created" in text
    assert TracePackage.load(traced.root).regions == []


def test_cli_region_detect_json(traced: TracePackage, capsys) -> None:
    assert main(["region", "detect", str(traced.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["regions"] == 30
    assert payload["counts"]["skipped"] == 0
    names = {d["region_name"] for d in payload["detections"]}
    assert "RoPE@layers.0.self_attn" in names


def test_cli_min_confidence_filters(traced: TracePackage, capsys) -> None:
    assert main(
        ["region", "detect", str(traced.root), "--min-confidence", "1.0", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {d["name"] for d in payload["detections"]} == {"Linear"}


def test_cli_detector_selection(traced: TracePackage, capsys) -> None:
    assert main(
        ["region", "detect", str(traced.root), "--detector", "source_function", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {d["method"] for d in payload["detections"]} == {"source_function"}


def test_cli_rejects_unknown_detector(traced: TracePackage) -> None:
    assert main(["region", "detect", str(traced.root), "--detector", "nope"]) == 2


def test_cli_replace_avoids_duplicates(traced: TracePackage, capsys) -> None:
    main(["region", "detect", str(traced.root)])
    main(["region", "detect", str(traced.root), "--replace"])
    capsys.readouterr()

    reloaded = TracePackage.load(traced.root)
    assert len(reloaded.regions) == 30
    assert len({r.name for r in reloaded.regions}) == 30
    # Annotations were cleared first, so no operator accumulated duplicates.
    for op in reloaded.graph.operators:
        names = [a.name for a in op.annotations]
        assert len(names) == len(set(names))


def test_trace_semantic_analysis_flag(
    tmp_path: Path, mini_llama, mini_llama_input, capsys
) -> None:
    """SPEC §33 --semantic-analysis, and SPEC §17: it stays optional."""
    script = tmp_path / "run.py"
    script.write_text(
        "import sys, torch\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2]).replace(chr(92), '/')!r})\n"
        "from examples.mini_llama.model import build_model, build_inputs\n"
        "m = build_model(seed=0)\n"
        "with torch.no_grad():\n"
        "    m(build_inputs(m, seq_len=8, seed=1))\n",
        encoding="utf-8",
    )

    plain = tmp_path / "plain"
    assert main(["trace", str(script), "-o", str(plain)]) == 0
    capsys.readouterr()
    assert TracePackage.load(plain).regions == []

    labelled = tmp_path / "labelled"
    assert main(["trace", str(script), "-o", str(labelled), "--semantic-analysis"]) == 0
    text = capsys.readouterr().out
    assert "regions:" in text

    package = TracePackage.load(labelled)
    assert package.regions
    assert any(r.name.startswith("RoPE@") for r in package.regions)
    # Physical trace is identical with and without semantic analysis.
    assert [o.canonical_name for o in package.graph.ops_in_execution_order()] == [
        o.canonical_name for o in TracePackage.load(plain).graph.ops_in_execution_order()
    ]


def test_cli_inspect_shows_semantic_labels(traced: TracePackage, capsys) -> None:
    main(["region", "detect", str(traced.root)])
    capsys.readouterr()
    assert main(["inspect", str(traced.root), "--operator", "aten.neg.default"]) == 0
    text = capsys.readouterr().out
    assert "semantic:" in text
    assert "RoPE" in text


# -- the workflow this replaces -------------------------------------------


def test_detected_region_extracts_a_working_testcase(
    traced: TracePackage, tmp_path: Path, capsys
) -> None:
    """The detected region drives the SPEC §65 loop without any operator ids."""
    assert main(["region", "detect", str(traced.root)]) == 0
    capsys.readouterr()

    repro = tmp_path / "repro"
    assert main(
        [
            "testcase", "extract", str(traced.root),
            "--region", "RoPE@layers.0.self_attn",
            "-o", str(repro),
            "--input-names", "cos,sin,query,key",
            "--output-names", "q_embed,k_embed",
        ]
    ) == 0
    capsys.readouterr()

    manifest = json.loads((repro / "testcase.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in manifest["inputs"]] == ["cos", "sin", "query", "key"]
    assert [e["name"] for e in manifest["outputs"]] == ["q_embed", "k_embed"]
    assert manifest["reproducible"]
    # Provenance now names the region a human never had to identify.
    assert manifest["outputs"][0]["producer"]["region"] == "RoPE@layers.0.self_attn"
