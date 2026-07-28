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
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package


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
    detach, add = graph.ops_in_execution_order()[:2]
    assert detach.canonical_name == "aten.detach.default"
    assert add.canonical_name == "aten.add.Tensor"

    detach_input = graph.op_input_value_ids(detach)[0]
    detach_output = graph.op_output_value_ids(detach)[0]
    add_input = graph.op_input_value_ids(add)[0]

    assert add_input == detach_input, "add should consume x, not detach's output"
    assert add_input != detach_output
    # detach's output is still its own value with its own object identity.
    assert (
        graph.value(detach_output).runtime_object_id
        != graph.value(detach_input).runtime_object_id
    )
    assert graph.value(detach_output).storage_id == graph.value(detach_input).storage_id


def test_distinct_objects_over_one_storage_are_distinct_values(trace_dir: Path) -> None:
    base = torch.randn(2, 3)

    def run(session):
        session.mark_input("base", base)
        # Two separate view calls: same storage, version and layout, new objects.
        session.mark_output("a", base.detach())
        session.mark_output("b", base.detach())

    package = _trace(run, trace_dir)
    detaches = _find(package, "aten.detach.default")
    assert len(detaches) == 2

    first = package.graph.value(package.graph.op_output_value_ids(detaches[0])[0])
    second = package.graph.value(package.graph.op_output_value_ids(detaches[1])[0])

    assert first.id != second.id
    assert first.runtime_object_id != second.runtime_object_id
    assert first.storage_id == second.storage_id
    assert first.storage_version == second.storage_version


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
