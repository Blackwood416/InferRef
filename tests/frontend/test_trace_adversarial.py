"""Adversarial tracing cases.

Edge cases where the obvious implementation is subtly wrong. Each test here
corresponds to a bug that was present and fixed, so they exist to stop it
coming back rather than to describe intended design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import inferref
from inferref.frontend.pytorch.dispatch import iter_tensors, iter_written_tensors
from inferref.frontend.pytorch.params import ParameterIndex
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.ir.values import TensorRef


def _trace(fn, trace_dir: Path, **kwargs) -> TracePackage:
    kwargs.setdefault("capture_tensors", "metadata")
    with torch.no_grad(), inferref.trace(output=trace_dir, **kwargs) as session:
        fn(session)
    return TracePackage.load(trace_dir)


def _mutations(op) -> list[tuple[int, int, int]]:
    return [
        (m.storage_id, m.version_before, m.version_after)
        for m in op.effects.mutated_storages
    ]


def _find(package: TracePackage, needle: str):
    return [
        op for op in package.graph.ops_in_execution_order() if needle in op.canonical_name
    ]


def _io_value_id(package: TracePackage, name: str, *, output: bool = False) -> int:
    entries = package.graph.outputs if output else package.graph.inputs
    value = next(entry.value for entry in entries if entry.name == name)
    assert isinstance(value, TensorRef)
    return value.value_id


# -- mutation binding ------------------------------------------------------


def test_keyword_only_write_is_detected(trace_dir: Path) -> None:
    """``aten.add.out`` declares its write on a kwarg-only ``out`` argument.

    Its schema index is past the end of ``args``, so binding writes by position
    alone loses the mutation entirely. ``out=`` variants are common enough that
    this is not an exotic case.
    """
    a, b, c = torch.zeros(3), torch.ones(3), torch.empty(3)

    package = _trace(lambda s: torch.add(a, b, out=c), trace_dir)
    ops = _find(package, "aten.add.out")
    assert ops, [o.canonical_name for o in package.graph.operators]

    mutations = _mutations(ops[0])
    assert len(mutations) == 1
    assert mutations[0][1:] == (0, 1)

    # The mutated storage is c's, not a's or b's.
    written = mutations[0][0]
    outputs = package.graph.op_output_value_ids(ops[0])
    assert package.graph.value(outputs[0]).storage_id == written


def test_out_reallocation_is_rebind_not_old_storage_mutation(trace_dir: Path) -> None:
    """An empty ``out=`` tensor may be rebound before receiving the result."""
    a = torch.ones(1024)
    b = torch.ones(1024)
    out = torch.empty(0)
    assert out.data_ptr() == 0

    package = _trace(lambda session: torch.add(a, b, out=out), trace_dir)
    op = _find(package, "aten.add.out")[0]
    before_ref = op.keyword_args["out"]
    assert isinstance(before_ref, TensorRef)
    before = package.graph.value(before_ref.value_id)
    output = package.graph.value(package.graph.op_output_value_ids(op)[0])

    assert out.data_ptr() != 0
    assert not op.effects.mutated_storages
    assert before.storage_id != output.storage_id
    assert output.storage_version == 0
    assert [item.relationship for item in op.effects.aliases] == ["same_object"]


def test_container_write_is_detected(trace_dir: Path) -> None:
    """``aten._foreach_add_`` declares its write on a ``List[Tensor]``.

    The written tensors are nested inside the argument, so an ``isinstance``
    check against the argument itself skips them.
    """
    tensors = [torch.zeros(2), torch.zeros(3)]

    package = _trace(lambda s: torch._foreach_add_(tensors, 1.0), trace_dir)
    ops = _find(package, "_foreach_add_")
    assert ops

    mutations = _mutations(ops[0])
    assert len(mutations) == 2, mutations
    assert all(before == 0 and after == 1 for _, before, after in mutations)
    # Two distinct storages, one per list element.
    assert len({storage for storage, _, _ in mutations}) == 2


def test_aliased_writes_bump_a_storage_once(trace_dir: Path) -> None:
    """Two writable arguments over one storage advance it a single generation."""
    tensor = torch.zeros(4)

    package = _trace(
        lambda s: torch._foreach_add_([tensor, tensor.view(2, 2)], 1.0), trace_dir
    )
    ops = _find(package, "_foreach_add_")
    mutations = _mutations(ops[0])

    assert len(mutations) == 1
    assert mutations[0][1:] == (0, 1)


def test_iter_written_tensors_binds_positional_arg_given_by_name() -> None:
    """A non-kwarg-only argument may still be supplied as a keyword."""
    target = torch.zeros(3)
    func = torch.ops.aten.add_.Tensor

    by_position = list(iter_written_tensors(func, (target, torch.ones(3)), {}))
    by_name = list(iter_written_tensors(func, (), {"self": target, "other": torch.ones(3)}))

    assert [t.data_ptr() for t in by_position] == [target.data_ptr()]
    assert [t.data_ptr() for t in by_name] == [target.data_ptr()]


def test_iter_written_tensors_skips_defaulted_arguments() -> None:
    """An argument left at its default is absent, not a write."""
    func = torch.ops.aten.add.out
    # `out` not supplied at all: nothing is written.
    assert list(iter_written_tensors(func, (torch.zeros(2), torch.ones(2)), {})) == []


def test_iter_tensors_walks_nested_containers() -> None:
    a, b, c = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)
    nested = [a, (b, {"k": c}), "not a tensor", 5, None]
    found = [t.item() for t in iter_tensors(nested)]
    assert found == [0.0, 1.0, 2.0]


def test_pure_operator_records_no_mutation(trace_dir: Path) -> None:
    """Guard against over-eager detection: `add` must not claim a write."""
    a, b = torch.zeros(3), torch.ones(3)
    package = _trace(lambda s: a + b, trace_dir)
    for op in package.graph.operators:
        assert not op.effects.mutated_storages, op.canonical_name


def test_independent_empty_tensors_have_distinct_storage_identity(
    trace_dir: Path,
) -> None:
    """Pointer zero is not a storage identity: every empty allocation has it."""
    x = torch.empty(0)
    y = torch.empty(0)
    assert x.data_ptr() == y.data_ptr() == 0
    assert x.untyped_storage()._cdata != y.untyped_storage()._cdata

    def run(session) -> None:
        session.mark_input("x", x)
        session.mark_input("y", y)
        x.add_(1.0)
        session.mark_output("y_after", y)

    package = _trace(run, trace_dir)
    x_value = package.graph.value(_io_value_id(package, "x"))
    y_value = package.graph.value(_io_value_id(package, "y"))
    y_after = package.graph.value(_io_value_id(package, "y_after", output=True))

    assert x_value.storage_id != y_value.storage_id
    assert y_after.storage_id == y_value.storage_id
    assert y_after.storage_version == 0
    mutation = _find(package, "aten.add_.Tensor")[0].effects.mutated_storages
    assert [
        (item.storage_id, item.version_before, item.version_after) for item in mutation
    ] == [(x_value.storage_id, 0, 1)]


def test_empty_parameter_views_do_not_cross_classify() -> None:
    module = torch.nn.Module()
    module.register_parameter("left", torch.nn.Parameter(torch.empty(0)))
    module.register_parameter("right", torch.nn.Parameter(torch.empty(0)))
    index = ParameterIndex()
    index.index(module)

    left_view = module.left.detach().view(0)
    right_view = module.right.detach().view(0)

    assert index.classify(left_view) == ("parameter", "left")
    assert index.classify(right_view) == ("parameter", "right")


def test_resize_reallocation_is_storage_rebind_not_old_storage_mutation(
    trace_dir: Path,
) -> None:
    tensor = torch.empty(1)
    pointer_before = tensor.data_ptr()

    def run(session) -> None:
        session.mark_input("before", tensor)
        tensor.resize_(1_000_000)
        session.mark_output("after", tensor)

    package = _trace(run, trace_dir)
    assert tensor.data_ptr() != pointer_before
    op = _find(package, "aten.resize_")[0]
    before = package.graph.value(_io_value_id(package, "before"))
    after = package.graph.value(_io_value_id(package, "after", output=True))

    assert not op.effects.mutated_storages
    assert before.storage_id != after.storage_id
    assert after.storage_version == 0
    assert [item.relationship for item in op.effects.aliases] == ["same_object"]


def test_resize_within_allocation_changes_object_metadata_only(trace_dir: Path) -> None:
    tensor = torch.empty(100)
    pointer_before = tensor.data_ptr()

    package = _trace(lambda session: tensor.resize_(50), trace_dir)
    assert tensor.data_ptr() == pointer_before
    op = _find(package, "aten.resize_")[0]
    input_id = next(
        ref.value_id for ref in op.positional_args if isinstance(ref, TensorRef)
    )
    output_id = package.graph.op_output_value_ids(op)[0]
    before = package.graph.value(input_id)
    after = package.graph.value(output_id)

    assert _mutations(op) == []
    assert after.storage_id == before.storage_id
    assert after.storage_version == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda tensor: tensor.transpose_(0, 1),
        lambda tensor: tensor.squeeze_(0),
        lambda tensor: tensor.unsqueeze_(0),
    ],
)
def test_metadata_only_inplace_does_not_version_shared_storage(
    trace_dir: Path, operation
) -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    alias = base.detach()

    def run(session) -> None:
        session.mark_input("base", base)
        session.mark_input("alias", alias)
        operation(base)
        session.mark_output("alias_after", alias)

    package = _trace(run, trace_dir)
    metadata_op = next(
        op
        for op in package.graph.ops_in_execution_order()
        if op.canonical_name in {
            "aten.transpose_.default",
            "aten.squeeze_.dim",
            "aten.unsqueeze_.default",
        }
    )
    alias_before = package.graph.value(_io_value_id(package, "alias"))
    alias_after = package.graph.value(_io_value_id(package, "alias_after", output=True))

    assert not metadata_op.effects.mutated_storages
    assert alias_after.storage_id == alias_before.storage_id
    assert alias_after.storage_version == alias_before.storage_version == 0
    assert alias_after.producer is None


def test_set_within_same_storage_is_metadata_only(trace_dir: Path) -> None:
    base = torch.arange(8, dtype=torch.float32)
    target = base[:4]
    alias = base.detach()

    def run(session) -> None:
        session.mark_input("alias", alias)
        target.set_(base.untyped_storage(), 2, (3,), (1,))
        session.mark_output("alias_after", alias)

    package = _trace(run, trace_dir)
    op = _find(package, "aten.set_")[0]
    alias_before = package.graph.value(_io_value_id(package, "alias"))
    alias_after = package.graph.value(_io_value_id(package, "alias_after", output=True))

    assert not op.effects.mutated_storages
    assert alias_after.storage_id == alias_before.storage_id
    assert alias_after.storage_version == alias_before.storage_version == 0


def test_set_records_object_rebind_and_new_storage_alias(trace_dir: Path) -> None:
    target = torch.zeros(2)
    source = torch.ones(3)

    package = _trace(lambda session: target.set_(source), trace_dir)
    op = _find(package, "aten.set_")[0]
    input_ids = [
        ref.value_id for ref in op.positional_args if isinstance(ref, TensorRef)
    ]
    output_id = package.graph.op_output_value_ids(op)[0]
    target_before, source_before = map(package.graph.value, input_ids)
    output = package.graph.value(output_id)

    assert not op.effects.mutated_storages
    assert output.storage_id != target_before.storage_id
    assert output.storage_id == source_before.storage_id
    assert output.storage_version == source_before.storage_version == 0
    assert {item.relationship for item in op.effects.aliases} == {
        "same_object",
        "shared_storage",
    }


# -- value identity vs payload identity ------------------------------------


def test_lineage_is_not_rewritten_by_an_alias_only_operator(trace_dir: Path) -> None:
    """``detach`` must not steal ``x``'s identity from later consumers.

    ``y = x.detach()`` yields a new object over the same storage, version and
    layout. If value identity ignored object identity, interning ``y`` would
    rebind that key and the later ``x + 1`` would resolve to ``y``, recording a
    serial ``x -> detach -> add`` chain where the truth is two independent uses
    of ``x``.
    """
    x = torch.randn(2, 3)

    def run(session):
        session.mark_input("x", x)
        x.detach()
        session.mark_output("z", x + 1)

    package = _trace(run, trace_dir)
    graph = package.graph

    # How many aten.detach calls one `.detach()` dispatches is a PyTorch
    # implementation detail that has changed across versions, so locate the
    # operators by name rather than by position.
    detaches = _find(package, "aten.detach.default")
    assert detaches
    add = next(op for op in graph.ops_in_execution_order() if op.op == "add")

    x_value = graph.op_input_value_ids(detaches[0])[0]
    add_input = graph.op_input_value_ids(add)[0]
    detach_outputs = {
        vid for d in detaches for vid in graph.op_output_value_ids(d)
    }

    assert add_input == x_value, "add should consume x, not a detach output"
    assert add_input not in detach_outputs
    # Each detach output is its own value with its own object identity.
    for vid in detach_outputs:
        assert graph.value(vid).runtime_object_id != graph.value(x_value).runtime_object_id
        assert graph.value(vid).storage_id == graph.value(x_value).storage_id


def test_distinct_objects_over_one_storage_are_distinct_values(trace_dir: Path) -> None:
    base = torch.randn(2, 3)

    def run(session):
        session.mark_input("base", base)
        # Two separate view calls: same storage, version and layout, new objects.
        session.mark_output("a", base.detach())
        session.mark_output("b", base.detach())

    package = _trace(run, trace_dir)
    detaches = _find(package, "aten.detach.default")
    assert len(detaches) >= 2

    outputs = [
        package.graph.value(package.graph.op_output_value_ids(d)[0]) for d in detaches
    ]
    # Every detach output is a distinct value over the one shared storage.
    assert len({v.id for v in outputs}) == len(outputs)
    assert len({v.runtime_object_id for v in outputs}) == len(outputs)
    assert len({v.storage_id for v in outputs}) == 1
    assert len({v.storage_version for v in outputs}) == 1


def test_distinct_values_share_one_payload(trace_dir: Path) -> None:
    """Value identity and payload identity are separate concerns (IR §39)."""

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.fc(x))

    model = Net().eval()
    package = _trace(
        lambda s: s.mark_output("y", model(torch.randn(2, 4))),
        trace_dir,
        capture_tensors="all",
    )

    weights = [v for v in package.graph.values if v.qualified_name == "fc.weight"]
    # The parameter plus one transposed view per fc() call.
    assert len(weights) == 3
    # ...but only two files: the parameter's layout and the transposed layout.
    assert len({v.capture.payload for v in weights}) == 2

    transposed = [v for v in weights if v.stride == (1, 4)]
    assert len(transposed) == 2
    assert transposed[0].id != transposed[1].id
    assert transposed[0].runtime_object_id != transposed[1].runtime_object_id
    assert transposed[0].capture.payload == transposed[1].capture.payload


def test_repeated_use_of_one_object_interns_once(trace_dir: Path) -> None:
    """The same object used twice is one value: dedup must survive the split.

    Adding ``runtime_object_id`` to the value key must not defeat parameter
    deduplication (IR §39) — a parameter is the *same* object on every use, so
    it still collapses onto a single value record.
    """

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.fc(x))

    model = Net().eval()
    package = _trace(lambda s: s.mark_output("y", model(torch.randn(2, 4))), trace_dir)
    graph = package.graph

    # Both fc() calls transpose the same weight object.
    transposes = _find(package, "aten.t.default")
    assert len(transposes) == 2

    first = graph.op_input_value_ids(transposes[0])[0]
    second = graph.op_input_value_ids(transposes[1])[0]
    assert first == second, "the same parameter object must intern to one value"

    weight = graph.value(first)
    assert weight.role == "parameter"
    assert weight.qualified_name == "fc.weight"


# -- tied weights ----------------------------------------------------------


class _TiedLM(torch.nn.Module):
    """An embedding whose weight is reused as the output projection."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed(ids))


def test_tied_weights_record_every_name(trace_dir: Path) -> None:
    """One storage carrying two parameter names must report both (IR §38).

    ``named_parameters()`` deduplicates shared tensors by default, so a tied
    ``lm_head`` would otherwise never report its own name at all.
    """
    model = _TiedLM().eval()
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr()

    package = _trace(
        lambda s: s.mark_output("logits", model(torch.tensor([[1, 2, 3]]))),
        trace_dir,
        capture_tensors="all",
    )
    tied = [v for v in package.graph.values if v.qualified_names]
    assert tied, "tied weights were not detected"

    for value in tied:
        assert set(value.qualified_names) == {"embed.weight", "lm_head.weight"}
        # The canonical name is stable regardless of how the tensor was reached.
        assert value.qualified_name == "embed.weight"
        assert value.role == "parameter"


def test_untied_parameters_have_no_alias_list(trace_dir: Path) -> None:
    """``qualified_names`` stays empty for the ordinary case."""
    model = torch.nn.Linear(4, 4).eval()
    package = _trace(
        lambda s: s.mark_output("y", model(torch.randn(2, 4))),
        trace_dir,
        capture_tensors="all",
    )
    parameters = [v for v in package.graph.values if v.role == "parameter"]
    assert parameters
    assert all(not v.qualified_names for v in parameters)


def test_tied_weight_names_survive_a_roundtrip(trace_dir: Path) -> None:
    model = _TiedLM().eval()
    _trace(
        lambda s: s.mark_output("logits", model(torch.tensor([[1, 2]]))),
        trace_dir,
        capture_tensors="all",
    )
    reloaded = TracePackage.load(trace_dir)
    tied = [v for v in reloaded.graph.values if v.qualified_names]
    assert tied
    assert set(tied[0].qualified_names) == {"embed.weight", "lm_head.weight"}


# -- tensor capture --------------------------------------------------------


def test_capture_uses_no_numpy_bridge(trace_dir: Path) -> None:
    """Byte extraction must not go through ``Tensor.numpy()``.

    That call crosses PyTorch's NumPy ABI bridge and fails outright when the
    installed NumPy major version differs from the one PyTorch was built
    against, which would silently cost every payload in the trace.
    """
    from inferref.frontend.pytorch import capture

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    called = False

    original = torch.Tensor.numpy

    def tripwire(self, *args, **kwargs):  # pragma: no cover - must not run
        nonlocal called
        called = True
        return original(self, *args, **kwargs)

    torch.Tensor.numpy = tripwire
    try:
        raw = capture.logical_bytes(x)
    finally:
        torch.Tensor.numpy = original

    assert not called, "logical_bytes must not call Tensor.numpy()"
    assert len(raw) == 24


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1),
        torch.arange(10, dtype=torch.float32)[3:7],
        torch.arange(10, dtype=torch.float32)[::2],
        torch.tensor([1.5, 2.5, -0.75], dtype=torch.bfloat16),
        torch.tensor([True, False, True]),
        torch.zeros(0, dtype=torch.float32),
        torch.tensor(2.5),
    ],
    ids=["contiguous", "transposed", "offset", "strided", "bfloat16", "bool", "empty", "scalar"],
)
def test_logical_bytes_length(tensor: torch.Tensor) -> None:
    """Views, offsets and odd dtypes all yield exactly numel x itemsize bytes."""
    from inferref.frontend.pytorch import capture

    raw = capture.logical_bytes(tensor)
    assert len(raw) == tensor.numel() * tensor.element_size()


def test_capture_failure_is_reported_not_swallowed(
    trace_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace whose payloads failed must say so (SPEC §59).

    Degrading `--capture-tensors all` to metadata silently would hand back a
    trace that looks complete and yields testcases that cannot run.
    """
    from inferref.frontend.pytorch import capture

    def boom(tensor):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(capture, "logical_bytes", boom)

    x = torch.randn(2, 3)
    package = _trace(
        lambda s: s.mark_output("y", x * 2), trace_dir, capture_tensors="all"
    )

    warnings = package.manifest.determinism.warnings
    assert any("tensor capture failed" in w for w in warnings), warnings
    assert any("simulated capture failure" in w for w in warnings)
    assert any("cannot be reproduced" in w for w in warnings)
    # And every value really did fall back to metadata.
    assert all(v.capture.mode == "metadata" for v in package.graph.values)
    assert all(v.capture.requested_mode == "full" for v in package.graph.values)
    assert all(v.capture.degraded_reason == "capture_error" for v in package.graph.values)


def test_capture_limit_records_structured_degradation(trace_dir: Path) -> None:
    """Hash-only fallback must retain why a full payload was not written."""
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    package = _trace(
        lambda session: session.mark_output("y", x * 2),
        trace_dir,
        capture_tensors="all",
        max_capture_elements=4,
    )

    degraded = [
        value.capture
        for value in package.graph.values
        if value.capture.degraded_reason == "max_capture_elements"
    ]
    assert degraded
    assert all(capture.mode == "hash" for capture in degraded)
    assert all(capture.requested_mode == "full" for capture in degraded)
    assert all(capture.limit == 4 for capture in degraded)
    assert all(capture.logical_numel == 6 for capture in degraded)


# -- validity --------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["out_kwarg", "foreach", "detach_then_use", "inplace_on_view"],
)
def test_adversarial_traces_stay_valid(trace_dir: Path, scenario: str) -> None:
    """Every case above must still satisfy all ten IR §48 invariants."""

    def run(session):
        if scenario == "out_kwarg":
            a, b, c = torch.zeros(3), torch.ones(3), torch.empty(3)
            torch.add(a, b, out=c)
            session.mark_output("c", c)
        elif scenario == "foreach":
            tensors = [torch.zeros(2), torch.zeros(2)]
            torch._foreach_add_(tensors, 1.0)
            session.mark_output("t0", tensors[0])
        elif scenario == "detach_then_use":
            x = torch.randn(2, 2)
            session.mark_input("x", x)
            x.detach()
            session.mark_output("z", x * 2)
        else:  # inplace_on_view
            base = torch.zeros(4, 4)
            session.mark_input("base", base)
            view = base[1:3]
            view.add_(1.0)
            session.mark_output("base_after", base)

    package = _trace(run, trace_dir)
    errors = [i for i in validate_package(package) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_inplace_on_a_view_versions_the_shared_storage(trace_dir: Path) -> None:
    """Writing through a view must advance the base tensor's storage."""
    base = torch.zeros(4, 4)

    def run(session):
        session.mark_input("base", base)
        base[1:3].add_(1.0)
        session.mark_output("after", base)

    package = _trace(run, trace_dir)
    mutating = [op for op in package.graph.operators if op.effects.mutated_storages]
    assert mutating

    storage_id = mutating[0].effects.mutated_storages[0].storage_id
    versions = sorted(
        {v.storage_version for v in package.graph.values if v.storage_id == storage_id}
    )
    assert versions == [0, 1]
