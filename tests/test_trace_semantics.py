"""Trace IR v0.1 acceptance criteria 1-7 (IR §57).

Each test names the criterion it covers:

1. a normal ``Linear``
2. a metadata-only ``transpose``
3. two distinct views sharing one storage
4. an in-place tensor mutation
5. a Transformer submodule call stack
6. operator scalar/keyword arguments
7. source mapping
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package


def _trace(model_fn, trace_dir: Path, **kwargs) -> TracePackage:
    """Run ``model_fn`` under a trace and return the loaded package."""
    kwargs.setdefault("capture_tensors", "all")
    with torch.no_grad(), inferref.trace(output=trace_dir, **kwargs) as session:
        model_fn(session)
    return TracePackage.load(trace_dir)


def _assert_valid(package: TracePackage) -> None:
    errors = [i for i in validate_package(package) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


# -- criterion 1: a normal Linear -----------------------------------------


def test_linear_is_recorded(trace_dir: Path) -> None:
    linear = torch.nn.Linear(4, 6).eval()
    x = torch.randn(2, 4)

    package = _trace(lambda s: s.mark_output("y", linear(x)), trace_dir)
    _assert_valid(package)

    names = [op.canonical_name for op in package.graph.ops_in_execution_order()]
    # Runtime truth: nn.Linear dispatches as t + addmm, not as a single mm.
    assert names == ["aten.t.default", "aten.addmm.default"]

    addmm = package.graph.ops_in_execution_order()[-1]
    assert package.module_path(addmm.module_stack) == ""
    outputs = package.graph.op_output_value_ids(addmm)
    assert package.graph.value(outputs[0]).shape == (2, 6)


def test_linear_parameters_are_classified(trace_dir: Path) -> None:
    linear = torch.nn.Linear(4, 6).eval()
    x = torch.randn(2, 4)
    package = _trace(lambda s: s.mark_output("y", linear(x)), trace_dir)

    named = {
        v.qualified_name: v.role for v in package.graph.values if v.qualified_name
    }
    assert named.get("weight") == "parameter"
    assert named.get("bias") == "parameter"


# -- criterion 2: a metadata-only transpose --------------------------------


def test_transpose_is_metadata_only(trace_dir: Path) -> None:
    base = torch.randn(2, 3)

    def run(session):
        session.mark_input("x", base)
        session.mark_output("y", base.transpose(0, 1))

    package = _trace(run, trace_dir)
    _assert_valid(package)

    op = package.graph.ops_in_execution_order()[0]
    assert op.canonical_name == "aten.transpose.int"

    src = package.graph.value(package.graph.op_input_value_ids(op)[0])
    dst = package.graph.value(package.graph.op_output_value_ids(op)[0])

    assert src.storage_id == dst.storage_id           # same storage
    assert src.storage_version == dst.storage_version  # no mutation
    assert src.shape == (2, 3) and dst.shape == (3, 2)
    assert src.stride == (3, 1) and dst.stride == (1, 3)
    assert src.runtime_object_id != dst.runtime_object_id  # different objects
    assert dst.contiguous is False

    # The alias relationship is recorded explicitly (IR §25).
    assert [a.relationship for a in op.effects.aliases] == ["view"]
    assert not op.effects.mutated_storages


# -- criterion 3: two views sharing one storage ----------------------------


def test_two_views_share_one_storage(trace_dir: Path) -> None:
    base = torch.randn(2, 3)

    def run(session):
        session.mark_input("x", base)
        session.mark_output("t", base.transpose(0, 1))
        session.mark_output("r", base.reshape(6))

    package = _trace(run, trace_dir)
    _assert_valid(package)

    by_layout = {v.shape: v for v in package.graph.values}
    original = by_layout[(2, 3)]
    transposed = by_layout[(3, 2)]
    flat = by_layout[(6,)]

    assert original.storage_id == transposed.storage_id == flat.storage_id
    # Distinct object identities, one shared storage.
    assert len({original.runtime_object_id, transposed.runtime_object_id,
                flat.runtime_object_id}) == 3
    assert transposed.stride == (1, 3)
    assert flat.stride == (1,)


# -- criterion 4: an in-place tensor mutation ------------------------------


def test_inplace_mutation_creates_two_values(trace_dir: Path) -> None:
    cache = torch.zeros(2, 3)

    def run(session):
        session.mark_input("cache", cache)
        cache.add_(1.0)
        session.mark_output("cache_after", cache)

    package = _trace(run, trace_dir)
    _assert_valid(package)

    op = package.graph.ops_in_execution_order()[0]
    assert op.canonical_name == "aten.add_.Tensor"

    # The schema declares the write, so InferRef advances its own version.
    assert len(op.effects.mutated_storages) == 1
    mutation = op.effects.mutated_storages[0]
    assert (mutation.version_before, mutation.version_after) == (0, 1)

    before = package.graph.value(package.graph.op_input_value_ids(op)[0])
    after = package.graph.value(package.graph.op_output_value_ids(op)[0])

    # IR §2.4: input and output are distinct immutable values...
    assert before.id != after.id
    assert before.storage_version == 0 and after.storage_version == 1
    # ...even though they are the same runtime object and storage.
    assert before.runtime_object_id == after.runtime_object_id
    assert before.storage_id == after.storage_id


def test_kv_cache_style_slice_write(trace_dir: Path) -> None:
    """The mutation pattern that actually matters: a cache write (IR §50)."""
    cache = torch.zeros(2, 4)
    value = torch.ones(2, 2)

    def run(session):
        session.mark_input("cache", cache)
        cache[:, 0:2] = value
        session.mark_output("cache_after", cache)

    package = _trace(run, trace_dir)
    _assert_valid(package)

    mutating = [
        op for op in package.graph.ops_in_execution_order() if op.effects.mutated_storages
    ]
    assert [op.canonical_name for op in mutating] == ["aten.copy_.default"]

    storage_id = mutating[0].effects.mutated_storages[0].storage_id
    versions = sorted(
        {v.storage_version for v in package.graph.values if v.storage_id == storage_id}
    )
    assert versions == [0, 1]


# -- criterion 5: a Transformer submodule call stack -----------------------


def test_module_stack_is_recorded(trace_dir: Path, mini_llama, mini_llama_input) -> None:
    package = _trace(
        lambda s: s.mark_output("out", mini_llama(mini_llama_input)), trace_dir
    )
    _assert_valid(package)

    paths = {package.module_path(op.module_stack) for op in package.graph.operators}
    assert "layers.0.self_attn.q_proj" in paths
    assert "layers.0.mlp.gate_proj" in paths
    assert "layers.1.self_attn.o_proj" in paths

    # The stack is outermost -> innermost and every id resolves.
    q_proj_ops = [
        op
        for op in package.graph.operators
        if package.module_path(op.module_stack) == "layers.0.self_attn.q_proj"
    ]
    assert q_proj_ops
    stack = q_proj_ops[0].module_stack
    resolved = [package.module(mid).path for mid in stack]
    assert resolved == ["", "layers.0", "layers.0.self_attn", "layers.0.self_attn.q_proj"]


def test_scope_filter_limits_capture(trace_dir: Path, mini_llama, mini_llama_input) -> None:
    package = _trace(
        lambda s: s.mark_output("out", mini_llama(mini_llama_input)),
        trace_dir,
        scope="layers.0.self_attn",
    )

    paths = {package.module_path(op.module_stack) for op in package.graph.operators}
    assert paths
    assert all(path.startswith("layers.0.self_attn") for path in paths), paths


# -- criterion 6: operator scalar/keyword arguments ------------------------


def test_scalar_and_keyword_arguments(trace_dir: Path) -> None:
    x = torch.randn(2, 3, 4)

    def run(session):
        session.mark_input("x", x)
        session.mark_output("s", torch.softmax(x, dim=-1))
        session.mark_output("c", torch.cat([x, x], dim=1))
        session.mark_output("m", x.mean(dim=1, keepdim=True))

    package = _trace(run, trace_dir)
    _assert_valid(package)

    ops = {op.canonical_name: op for op in package.graph.ops_in_execution_order()}

    softmax = ops["aten._softmax.default"]
    # dim is carried as a typed scalar, not a stringified blob.
    assert softmax.positional_args[1].kind == "scalar"
    assert softmax.positional_args[1].value == -1
    assert softmax.positional_args[1].dtype == "int64"
    assert softmax.positional_args[2].value is False
    assert softmax.positional_args[2].dtype == "bool"

    cat = ops["aten.cat.default"]
    assert cat.positional_args[0].kind == "list"
    assert len(cat.positional_args[0].items) == 2
    assert cat.positional_args[1].value == 1

    mean = ops["aten.mean.dim"]
    assert mean.positional_args[1].kind == "list"       # dim=[1]
    assert mean.positional_args[2].value is True        # keepdim


def test_special_float_scalars_are_json_safe(trace_dir: Path) -> None:
    """IR §11: nan/inf must survive JSON round-tripping."""
    x = torch.randn(2, 3)

    def run(session):
        session.mark_input("x", x)
        session.mark_output("y", torch.clamp(x, min=float("-inf")))

    package = _trace(run, trace_dir)
    reloaded = TracePackage.load(trace_dir)

    def specials(pkg):
        found = []
        for op in pkg.graph.operators:
            for arg in list(op.positional_args) + list(op.keyword_args.values()):
                if getattr(arg, "encoding", None) == "special":
                    found.append(arg.value)
        return found

    assert "-inf" in specials(package)
    assert specials(package) == specials(reloaded)


# -- criterion 7: source mapping -------------------------------------------


def test_source_mapping_points_at_model_code(
    trace_dir: Path, mini_llama, mini_llama_input
) -> None:
    package = _trace(
        lambda s: s.mark_output("out", mini_llama(mini_llama_input)), trace_dir
    )

    assert package.sources, "no source records were captured"

    rope_ops = [
        op
        for op in package.graph.operators
        if (src := package.source(op.source_id)) is not None
        and src.primary is not None
        and src.primary.function == "apply_rotary_pos_emb"
    ]
    assert rope_ops, "no operator was attributed to apply_rotary_pos_emb"

    primary = package.source(rope_ops[0].source_id).primary
    assert primary.file.endswith("examples/mini_llama/model.py")
    assert primary.line > 0
    # Framework frames are filtered out; the stack is model code only.
    stack = package.source(rope_ops[0].source_id).stack
    assert all("site-packages/torch" not in frame.file for frame in stack)


def test_source_text_is_not_embedded_by_default(
    trace_dir: Path, mini_llama, mini_llama_input
) -> None:
    """SPEC §58: embedding model source is opt-in."""
    package = _trace(
        lambda s: s.mark_output("out", mini_llama(mini_llama_input)), trace_dir
    )

    assert package.manifest.source_policy.embed_source_text is False
    for source in package.sources:
        for frame in source.stack:
            assert frame.source_text is None


def test_source_text_can_be_embedded(
    trace_dir: Path, mini_llama, mini_llama_input
) -> None:
    package = _trace(
        lambda s: s.mark_output("out", mini_llama(mini_llama_input)),
        trace_dir,
        embed_source_text=True,
    )
    assert package.manifest.source_policy.embed_source_text is True
    assert any(
        frame.source_text for source in package.sources for frame in source.stack
    )
