"""Semantic detection tests (SPEC §17, §56; IR §31, §32, §35).

These live in the core suite because detection operates on a loaded trace
package and never touches PyTorch — the packages here are built by hand, which
also lets each case isolate one rule.
"""

from __future__ import annotations

import pytest

from inferref.ir.graph import Graph, GraphIO
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.source import SourceFrame, SourceRecord
from inferref.ir.tensor_value import TensorValueRecord
from inferref.ir.validate import validate_package
from inferref.ir.values import TensorRef
from inferref.semantic import (
    CONFIDENCE_DETERMINISTIC,
    CONFIDENCE_STRONG,
    CONFIDENCE_VERY_STRONG,
    Detection,
    ModuleTypeDetector,
    SourceFunctionDetector,
    apply_detections,
    clear_semantic_annotations,
    detect,
    detector_names,
    is_contiguous,
    select,
    semantic_for_function,
    semantic_for_type,
    split_invocations,
)


# -- fixtures --------------------------------------------------------------


def _build(
    *,
    ops: list[tuple[int, str, list[int], int, tuple[int, ...], int | None]],
    modules: list[ModuleRecord] | None = None,
    sources: list[SourceRecord] | None = None,
) -> TracePackage:
    """Build a package from ``(id, name, inputs, output, module_stack, source_id)``."""
    value_ids = {v for _, _, ins, out, _, _ in ops for v in [*ins, out]}
    values = [
        TensorValueRecord(id=v, dtype="float32", shape=(2, 2), stride=(2, 1), storage_id=v)
        for v in sorted(value_ids)
    ]
    records = [
        OperatorRecord(
            id=op_id,
            execution_index=index,
            namespace="aten",
            op=name,
            overload="default",
            positional_args=tuple(TensorRef(v) for v in ins),
            result=TensorRef(out),
            module_stack=stack,
            source_id=source_id,
        )
        for index, (op_id, name, ins, out, stack, source_id) in enumerate(ops)
    ]
    graph = Graph(operators=records, values=values)
    graph.recompute_links()
    produced = {out for _, _, _, out, _, _ in ops}
    graph.inputs = [
        GraphIO(f"in_{v}", TensorRef(v)) for v in sorted(value_ids - produced)
    ]
    return TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=modules or [],
        sources=sources or [],
    )


def _linear_package() -> TracePackage:
    """Two Linear modules under one Attention, plus an unlabelled module."""
    modules = [
        ModuleRecord(id=1, path="", type="mymodel.Net"),
        ModuleRecord(id=2, path="attn", type="mymodel.Attention", parent_id=1),
        ModuleRecord(
            id=3, path="attn.q_proj", type="torch.nn.modules.linear.Linear", parent_id=2
        ),
        ModuleRecord(
            id=4, path="attn.k_proj", type="torch.nn.modules.linear.Linear", parent_id=2
        ),
        ModuleRecord(id=5, path="mystery", type="mymodel.Frobnicator", parent_id=1),
    ]
    ops = [
        (1, "t", [1], 2, (1, 2, 3), None),
        (2, "mm", [2, 3], 4, (1, 2, 3), None),
        (3, "t", [5], 6, (1, 2, 4), None),
        (4, "mm", [6, 7], 8, (1, 2, 4), None),
        (5, "frobnicate", [8], 9, (1, 5), None),
    ]
    return _build(ops=ops, modules=modules)


# -- module type mapping ---------------------------------------------------


@pytest.mark.parametrize(
    "type_path,expected,confidence",
    [
        ("torch.nn.modules.linear.Linear", "Linear", CONFIDENCE_DETERMINISTIC),
        ("torch.nn.modules.normalization.LayerNorm", "LayerNorm", CONFIDENCE_DETERMINISTIC),
        ("torch.nn.modules.sparse.Embedding", "Embedding", CONFIDENCE_DETERMINISTIC),
        ("mymodel.RMSNorm", "RMSNorm", CONFIDENCE_STRONG),
        ("transformers.models.qwen3.Qwen3RMSNorm", "RMSNorm", CONFIDENCE_STRONG),
        ("mymodel.LlamaRotaryEmbedding", "RoPE", CONFIDENCE_STRONG),
        ("mymodel.SwiGLUMLP", "SwiGLU", CONFIDENCE_STRONG),
        ("mymodel.CausalSelfAttention", "Attention", CONFIDENCE_STRONG),
        ("mymodel.DecoderLayer", "TransformerBlock", CONFIDENCE_STRONG),
        ("mymodel.StaticKVCache", "KVCacheUpdate", CONFIDENCE_STRONG),
        ("mymodel.KeyValueCache", "KVCacheUpdate", CONFIDENCE_STRONG),
    ],
)
def test_module_type_mapping(type_path: str, expected: str, confidence: float) -> None:
    resolved = semantic_for_type(type_path)
    assert resolved is not None, type_path
    assert resolved == (expected, confidence)


@pytest.mark.parametrize(
    "type_path",
    [
        "",
        "mymodel.Frobnicator",
        # An unlisted torch built-in is left unlabelled rather than guessed at.
        "torch.nn.modules.rnn.LSTM",
    ],
)
def test_unrecognised_types_are_not_guessed(type_path: str) -> None:
    assert semantic_for_type(type_path) is None


def test_specific_patterns_win_over_general() -> None:
    """`RMSNorm` must not be swallowed by the `layernorm` pattern."""
    assert semantic_for_type("m.RMSNorm")[0] == "RMSNorm"
    assert semantic_for_type("m.LayerNorm")[0] == "LayerNorm"
    # `SwiGLUMLP` contains both `swiglu` and `mlp`; the specific one wins.
    assert semantic_for_type("m.SwiGLUMLP")[0] == "SwiGLU"
    assert semantic_for_type("m.AttentionKVCache")[0] == "KVCacheUpdate"


def test_builtin_beats_name_heuristic() -> None:
    """A real torch class scores 1.0, not the 0.9 its name would earn."""
    name, confidence = semantic_for_type("torch.nn.modules.normalization.LayerNorm")
    assert (name, confidence) == ("LayerNorm", CONFIDENCE_DETERMINISTIC)


# -- module type detector --------------------------------------------------


def test_module_detector_labels_each_module() -> None:
    package = _linear_package()
    found = ModuleTypeDetector().detect(package)
    by_name = {d.region_name: d for d in found}

    assert by_name["Linear@attn.q_proj"].confidence == CONFIDENCE_DETERMINISTIC
    assert by_name["Linear@attn.q_proj"].node_ids == (1, 2)
    assert by_name["Linear@attn.k_proj"].node_ids == (3, 4)
    # Attention contains its projections (IR §36 permits the nesting).
    assert by_name["Attention@attn"].node_ids == (1, 2, 3, 4)
    assert "Frobnicator" not in " ".join(by_name)
    assert all(d.method == "module" for d in found)


def test_module_detector_skips_containers() -> None:
    """A Sequential would produce a region identical to its children's."""
    modules = [
        ModuleRecord(id=1, path="seq", type="torch.nn.modules.container.Sequential"),
        ModuleRecord(
            id=2, path="seq.0", type="torch.nn.modules.linear.Linear", parent_id=1
        ),
    ]
    package = _build(ops=[(1, "mm", [1, 2], 3, (1, 2), None)], modules=modules)
    names = {d.name for d in ModuleTypeDetector().detect(package)}
    assert names == {"Linear"}


def test_module_detector_splits_repeated_invocations() -> None:
    """A module called twice yields two regions, not one spanning both."""
    modules = [
        ModuleRecord(id=1, path="", type="mymodel.Net"),
        ModuleRecord(id=2, path="fc", type="torch.nn.modules.linear.Linear", parent_id=1),
        ModuleRecord(id=3, path="act", type="torch.nn.modules.activation.SiLU", parent_id=1),
    ]
    ops = [
        (1, "mm", [1, 2], 3, (1, 2), None),      # fc, first call
        (2, "silu", [3], 4, (1, 3), None),       # act, in between
        (3, "mm", [4, 2], 5, (1, 2), None),      # fc, second call
    ]
    package = _build(ops=ops, modules=modules)
    linear = [d for d in ModuleTypeDetector().detect(package) if d.name == "Linear"]

    assert len(linear) == 2
    assert linear[0].node_ids == (1,)
    assert linear[1].node_ids == (3,)
    assert {d.scope for d in linear} == {"fc#0", "fc#1"}


# -- source function detector ----------------------------------------------


def _rope_package() -> TracePackage:
    """RoPE calling an inlined `rotate_half`, twice, as a real trace does."""
    modules = [
        ModuleRecord(id=1, path="", type="mymodel.Net"),
        ModuleRecord(id=2, path="attn", type="mymodel.Attention", parent_id=1),
    ]
    outer = SourceFrame("model.py", 10, "apply_rotary_pos_emb")
    inner = SourceFrame("model.py", 20, "rotate_half")
    caller = SourceFrame("model.py", 99, "forward")
    sources = [
        SourceRecord(id=1, primary=outer, stack=(outer, caller)),
        # The helper's own frame is innermost; the target function is its caller.
        SourceRecord(id=2, primary=inner, stack=(inner, outer, caller)),
        SourceRecord(id=3, primary=caller, stack=(caller,)),
    ]
    ops = [
        (1, "mul", [1, 2], 3, (1, 2), 1),     # apply_rotary_pos_emb
        (2, "neg", [3], 4, (1, 2), 2),        # rotate_half
        (3, "cat", [4], 5, (1, 2), 2),        # rotate_half
        (4, "add", [5], 6, (1, 2), 1),        # apply_rotary_pos_emb
        (5, "mm", [6, 7], 8, (1, 2), 3),      # unrelated
        (6, "mul", [8, 2], 9, (1, 2), 1),     # second invocation
        (7, "add", [9], 10, (1, 2), 1),
    ]
    return _build(ops=ops, modules=modules, sources=sources)


def test_source_detector_matches_the_whole_stack() -> None:
    """Operators from an inlined helper belong to their caller's region.

    Matching only the innermost frame would drop the `rotate_half` operators,
    leaving a node set with holes and therefore spurious region inputs.
    """
    package = _rope_package()
    found = [d for d in SourceFunctionDetector().detect(package) if d.name == "RoPE"]

    assert len(found) == 2
    first = found[0]
    assert first.node_ids == (1, 2, 3, 4)     # includes the rotate_half ops
    assert first.confidence == CONFIDENCE_VERY_STRONG
    assert first.method == "source_function"
    # Repeated invocations at one module path get stable ordinals.  Distinct
    # layer paths do not need them, but prefill/decode calls do.
    assert first.scope == "attn#0"
    assert found[1].scope == "attn#1"
    assert is_contiguous(package.graph, first.node_ids)

    # The unrelated `mm` at index 4 splits the two invocations.
    assert found[1].node_ids == (6, 7)


def test_nested_helper_is_not_its_own_region() -> None:
    """`rotate_half` only ever runs inside RoPE, so it is not detected alone."""
    assert semantic_for_function("rotate_half") is None
    names = {d.name for d in SourceFunctionDetector().detect(_rope_package())}
    assert names == {"RoPE"}


@pytest.mark.parametrize(
    "function,expected",
    [
        ("apply_rotary_pos_emb", "RoPE"),
        ("apply_rope", "RoPE"),
        ("repeat_kv", "RepeatKV"),
        ("update_kv_cache", "KVCacheUpdate"),
        ("append_kv_cache", "KVCacheUpdate"),
        ("scaled_dot_product_attention", "Attention"),
        ("forward", None),
        ("some_random_helper", None),
    ],
)
def test_source_function_mapping(function: str, expected: str | None) -> None:
    assert semantic_for_function(function) == expected


def test_source_detector_honours_instance_patterns() -> None:
    """Custom detector configuration must not fall back to global patterns."""
    frame = SourceFrame("extension.py", 7, "project_fused_gate")
    package = _build(
        ops=[(1, "mul", [1, 2], 3, (), 1)],
        sources=[SourceRecord(id=1, primary=frame, stack=(frame,))],
    )
    detector = SourceFunctionDetector(
        patterns=((r"^project_fused_gate$", "ProjectGate"),)
    )

    found = detector.detect(package)

    assert [d.name for d in found] == ["ProjectGate"]
    assert found[0].node_ids == (1,)


# -- invocation splitting --------------------------------------------------


def test_split_is_strict_by_default() -> None:
    """A hole splits the run unless the caller opts into tolerating it."""
    package = _rope_package()
    assert split_invocations(package.graph, [1, 4]) == [(1,), (4,)]


def test_split_fills_a_tolerated_gap() -> None:
    """With a larger max_gap the in-between operators are absorbed.

    This is what keeps a region a contiguous slice of execution, so its derived
    boundary stays clean (IR §34).
    """
    package = _rope_package()
    assert split_invocations(package.graph, [1, 4], max_gap=3) == [(1, 2, 3, 4)]


def test_split_keeps_contiguous_runs_intact() -> None:
    package = _rope_package()
    assert split_invocations(package.graph, [1, 2, 3, 4]) == [(1, 2, 3, 4)]


def test_split_separates_distant_runs() -> None:
    package = _rope_package()
    runs = split_invocations(package.graph, [1, 2, 3, 4, 6, 7])
    assert runs == [(1, 2, 3, 4), (6, 7)]


def test_split_ignores_unknown_ids() -> None:
    package = _rope_package()
    assert split_invocations(package.graph, [999]) == []
    assert split_invocations(package.graph, []) == []


def test_is_contiguous() -> None:
    package = _rope_package()
    assert is_contiguous(package.graph, [1, 2, 3])
    assert not is_contiguous(package.graph, [1, 4])


# -- orchestration ---------------------------------------------------------


def test_detect_filters_by_confidence() -> None:
    package = _linear_package()
    assert any(d.confidence == CONFIDENCE_STRONG for d in detect(package))
    # 1.0 keeps only the deterministic torch built-ins.
    strict = detect(package, min_confidence=1.0)
    assert strict
    assert {d.name for d in strict} == {"Linear"}


def test_detect_deduplicates_identical_node_sets() -> None:
    """One construct recognised twice yields one region, the more confident."""
    modules = [ModuleRecord(id=1, path="rope", type="mymodel.RotaryEmbedding")]
    frame = SourceFrame("model.py", 10, "apply_rotary_pos_emb")
    sources = [SourceRecord(id=1, primary=frame, stack=(frame,))]
    package = _build(
        ops=[(1, "mul", [1, 2], 3, (1,), 1)], modules=modules, sources=sources
    )

    found = detect(package)
    assert len(found) == 1
    # source_function (0.95) beats module name heuristic (0.90).
    assert found[0].confidence == CONFIDENCE_VERY_STRONG


def test_detect_retains_different_semantics_on_identical_node_sets() -> None:
    """One physical boundary may carry multiple valid semantic readings."""
    package = _build(ops=[(1, "mul", [1, 2], 3, (), None)])

    class FixedDetector:
        def __init__(self, semantic: str) -> None:
            self.semantic = semantic

        def detect(self, _package: TracePackage) -> list[Detection]:
            return [
                Detection(
                    name=self.semantic,
                    node_ids=(1,),
                    confidence=CONFIDENCE_STRONG,
                    detector=f"test.{self.semantic}",
                    method="module",
                )
            ]

    found = detect(
        package,
        detectors=[FixedDetector("MLP"), FixedDetector("SwiGLU")],
    )

    assert {d.name for d in found} == {"MLP", "SwiGLU"}
    applied = apply_detections(package, found)
    assert {region.semantic.name for region in applied.regions} == {"MLP", "SwiGLU"}
    assert {annotation.name for annotation in package.graph.op(1).annotations} == {
        "MLP",
        "SwiGLU",
    }


def test_detect_orders_outermost_first() -> None:
    found = detect(_linear_package())
    sizes = [len(d.node_ids) for d in found]
    assert sizes == sorted(sizes, reverse=True)


def test_apply_writes_annotations_and_regions() -> None:
    package = _linear_package()
    result = apply_detections(package, detect(package))

    assert result.regions
    assert result.annotated_operators == 4      # the Frobnicator op stays unlabelled
    assert {r.name for r in package.regions} >= {
        "Linear@attn.q_proj",
        "Linear@attn.k_proj",
        "Attention@attn",
    }
    region = next(r for r in package.regions if r.name == "Linear@attn.q_proj")
    assert region.semantic is not None
    assert region.semantic.name == "Linear"
    assert region.semantic.confidence == CONFIDENCE_DETERMINISTIC
    assert region.creation_method == "module"


def test_annotations_nest_innermost_last() -> None:
    package = _linear_package()
    apply_detections(package, detect(package))

    op = package.graph.op(1)   # inside Linear, inside Attention
    names = [a.name for a in op.annotations if a.type == "semantic"]
    assert names == ["Attention", "Linear"]


def test_apply_does_not_touch_physical_truth() -> None:
    """SPEC §68.3: semantics annotate, they never rewrite the trace."""
    package = _linear_package()
    before = [
        (o.id, o.canonical_name, o.execution_index, o.positional_args, o.result)
        for o in package.graph.operators
    ]
    apply_detections(package, detect(package))
    after = [
        (o.id, o.canonical_name, o.execution_index, o.positional_args, o.result)
        for o in package.graph.operators
    ]
    assert before == after


def test_detected_regions_validate() -> None:
    """Derived boundaries must satisfy IR §48 invariant 9."""
    package = _linear_package()
    apply_detections(package, detect(package))
    errors = [i for i in validate_package(package) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_annotate_only() -> None:
    package = _linear_package()
    result = apply_detections(package, detect(package), create_regions=False)
    assert result.annotated_operators > 0
    assert not package.regions


def test_regions_only() -> None:
    package = _linear_package()
    apply_detections(package, detect(package), annotate=False)
    assert package.regions
    assert all(not op.annotations for op in package.graph.operators)


def test_clear_semantic_annotations() -> None:
    package = _linear_package()
    apply_detections(package, detect(package))
    removed = clear_semantic_annotations(package)
    assert removed > 0
    assert all(not op.annotations for op in package.graph.operators)


def test_name_collisions_are_suffixed() -> None:
    package = _linear_package()
    detections = detect(package)
    apply_detections(package, detections)
    apply_detections(package, detections)   # again, into the same package
    names = [r.name for r in package.regions]
    assert len(names) == len(set(names))
    assert any(n.endswith("#1") for n in names)


def test_annotations_are_not_duplicated_on_rerun() -> None:
    package = _linear_package()
    detections = detect(package)
    apply_detections(package, detections, create_regions=False)
    first = [len(op.annotations) for op in package.graph.operators]
    apply_detections(package, detections, create_regions=False)
    assert [len(op.annotations) for op in package.graph.operators] == first


def test_detections_survive_a_roundtrip(tmp_path) -> None:
    package = _linear_package()
    apply_detections(package, detect(package))
    package.save(tmp_path / "trace")

    reloaded = TracePackage.load(tmp_path / "trace")
    op = reloaded.graph.op(1)
    semantic = [a for a in op.annotations if a.type == "semantic"]
    assert [a.name for a in semantic] == ["Attention", "Linear"]
    assert semantic[-1].detector.startswith("inferref.semantic.")
    assert any(r.name == "Attention@attn" for r in reloaded.regions)


# -- registry --------------------------------------------------------------


def test_registry() -> None:
    assert detector_names() == ["module_type", "source_function"]
    assert len(select(None)) == 2
    assert [d.name for d in select(["source_function"])] == ["source_function"]
    with pytest.raises(ValueError, match="unknown detector"):
        select(["nope"])


def test_selecting_one_detector_excludes_the_other() -> None:
    package = _rope_package()
    found = detect(package, detector_names=["module_type"])
    assert all(d.method == "module" for d in found)


# -- Detection record ------------------------------------------------------


def test_detection_rejects_empty_node_set() -> None:
    with pytest.raises(ValueError, match="no nodes"):
        Detection(name="X", node_ids=(), confidence=1.0, detector="d", method="module")


def test_detection_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection(name="X", node_ids=(1,), confidence=1.5, detector="d", method="module")


def test_detection_region_name() -> None:
    scoped = Detection(
        name="RoPE", node_ids=(1,), confidence=1.0, detector="d",
        method="source_function", scope="layers.0.self_attn",
    )
    assert scoped.region_name == "RoPE@layers.0.self_attn"
    bare = Detection(name="RoPE", node_ids=(1,), confidence=1.0, detector="d", method="module")
    assert bare.region_name == "RoPE"
