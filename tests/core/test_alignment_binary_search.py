"""Coarse-to-fine binary-search narrowing on a synthetic hierarchical trace.

The fixture is a hand-built 28-layer decoder trace: no checkpoint, no PyTorch,
and no accelerator.  It has the same nested region shape as
:func:`inferref.frontend.pytorch.hierarchical.compile_step_trace`:

    WholeModel@prefill
      TransformerBlock@layers.<i>
        RMSNorm@layers.<i>.input_layernorm
        Attention@layers.<i>.self_attn
          Linear@layers.<i>.self_attn.q_proj
          ...
        RMSNorm@layers.<i>.post_attention_layernorm
        SwiGLU@layers.<i>.mlp
          Linear@layers.<i>.mlp.down_proj

The fake runner models a serial prefix probe: a boundary that ends at or after
the injected ``layers.19.mlp.down_proj`` defect fails, and one that ends before
it passes.  That is the monotone predicate a coarse-to-fine midpoint search
needs; the test counts how many boundaries must actually be executed.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferref.alignment.boundaries import BoundaryInventory, inventory_boundaries
from inferref.alignment.narrowing import (
    BINARY_SEARCH_REASON,
    compute_narrowing,
    serial_chain_children,
)
from inferref.alignment.planner import plan_alignment
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.region.boundary import derive_boundary

NUM_LAYERS = 28
FAULT_LAYER = 19
FAULT_NAME = "Linear@layers.19.mlp.down_proj"
WHOLE_MODEL = "WholeModel@prefill"

#: Execution order of the sub-blocks and leaf projections the fake runner
#: observes below the injected layer.
LAYER_BLOCKS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass
class _Comparison:
    result: str
    observations: dict


def _build_hierarchical_package(num_layers: int = NUM_LAYERS) -> TracePackage:
    """Build one deterministic prefill trace with nested decoder regions."""
    graph = Graph()
    values: list[TensorValueRecord] = [
        TensorValueRecord(
            id=1,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            capture=CaptureInfo(mode="full", payload="payloads/v1.irtensor"),
        )
    ]
    operators: list[OperatorRecord] = []
    layer_ops: list[dict[str, list[int]]] = []
    state = {"value": 2, "op": 1000, "execution": 0}

    def add_op(name: str, inputs: tuple[int, ...]) -> int:
        op_id = state["op"]
        value_id = state["value"]
        operators.append(
            OperatorRecord(
                id=op_id,
                execution_index=state["execution"],
                op=name,
                result=TensorRef(value_id),
                positional_args=tuple(TensorRef(value) for value in inputs),
            )
        )
        values.append(
            TensorValueRecord(
                id=value_id,
                shape=(2,),
                stride=(1,),
                dtype="float32",
                producer=op_id,
                capture=CaptureInfo(mode="full", payload=f"payloads/v{value_id}.irtensor"),
            )
        )
        state["op"] += 1
        state["value"] += 1
        state["execution"] += 1
        return op_id

    previous = 1
    for _ in range(num_layers):
        ops: dict[str, list[int]] = {}
        ops["input_layernorm"] = [add_op("rms_norm", (previous,))]
        attention_input = ops["input_layernorm"][0]
        ops["self_attn"] = [
            add_op("q_proj", (attention_input,)),
            add_op("k_proj", (attention_input,)),
            add_op("v_proj", (attention_input,)),
        ]
        ops["self_attn"].append(add_op("sdpa", tuple(ops["self_attn"])))
        ops["self_attn"].append(add_op("o_proj", (ops["self_attn"][-1],)))
        residual = add_op("add", (previous, ops["self_attn"][-1]))
        ops["post_attention_layernorm"] = [add_op("rms_norm", (residual,))]
        mlp_input = ops["post_attention_layernorm"][0]
        ops["mlp"] = [
            add_op("gate_proj", (mlp_input,)),
            add_op("up_proj", (mlp_input,)),
        ]
        ops["mlp"].append(add_op("swiglu", tuple(ops["mlp"])))
        ops["mlp"].append(add_op("down_proj", (ops["mlp"][-1],)))
        outputs = add_op("add", (residual, ops["mlp"][-1]))
        ops["output"] = [outputs]
        layer_ops.append(ops)
        previous = outputs

    graph.values = values
    graph.operators = operators
    graph.reindex()

    regions: list[RegionRecord] = []

    def add_region(name: str, node_ids: list[int]) -> None:
        inputs, outputs = derive_boundary(graph, node_ids)
        regions.append(
            RegionRecord(
                id=len(regions),
                name=name,
                node_ids=tuple(node_ids),
                inputs=tuple(inputs),
                outputs=tuple(outputs),
            )
        )

    all_nodes = [op.id for op in graph.ops_in_execution_order()]
    add_region(WHOLE_MODEL, all_nodes)
    for index, ops in enumerate(layer_ops):
        layer = f"layers.{index}"
        block_nodes = [
            node_id
            for key in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
            for node_id in ops[key]
        ] + ops["output"]
        add_region(f"TransformerBlock@{layer}", block_nodes)
        add_region(f"RMSNorm@{layer}.input_layernorm", ops["input_layernorm"])
        add_region(f"Attention@{layer}.self_attn", ops["self_attn"])
        for projection, leaf_index in zip(
            ("q_proj", "k_proj", "v_proj", "o_proj"), (0, 1, 2, 4), strict=True
        ):
            add_region(
                f"Linear@{layer}.self_attn.{projection}",
                [ops["self_attn"][leaf_index]],
            )
        add_region(
            f"RMSNorm@{layer}.post_attention_layernorm",
            ops["post_attention_layernorm"],
        )
        add_region(f"SwiGLU@{layer}.mlp", ops["mlp"])
        for projection, leaf_index in zip(
            MLP_PROJECTIONS, (0, 1, 3), strict=True
        ):
            add_region(f"Linear@{layer}.mlp.{projection}", [ops["mlp"][leaf_index]])

    return TracePackage(manifest=Manifest(), graph=graph, regions=regions)


def _boundary_id(inventory: BoundaryInventory, name: str) -> str:
    for boundary in inventory.boundaries:
        if boundary.name == name:
            return boundary.id
    raise AssertionError(f"missing boundary {name!r}")


def _failing_session(
    inventory: BoundaryInventory,
    current_failure: str,
    states: dict[str, str] | None = None,
) -> dict:
    boundaries = {current_failure: {"state": "failing"}}
    for boundary_id, state in (states or {}).items():
        boundaries[boundary_id] = {"state": state}
    return {"current_failure": current_failure, "boundaries": boundaries}


class _FakeSerialRunner:
    """Planner-driven evaluator that counts executed boundary evaluations."""

    def __init__(self, package: TracePackage) -> None:
        self.package = package
        self.inventory = inventory_boundaries(package)
        self.whole = _boundary_id(self.inventory, WHOLE_MODEL)
        self.fault = _boundary_id(self.inventory, FAULT_NAME)
        self.current_failure = self.whole
        self.states: dict[str, str] = {self.whole: "failing"}
        self.evaluations = 0
        self.inferred = 0
        self.selected: list[str] = []

    def session(self) -> dict:
        return {
            "current_failure": self.current_failure,
            "boundaries": {
                boundary_id: {"state": state}
                for boundary_id, state in self.states.items()
            },
        }

    def _prefix_fails(self, name: str) -> bool:
        path = name.split("@", 1)[1] if "@" in name else ""
        parts = path.split(".")
        if not parts or parts[0] != "layers":
            return True
        layer = int(parts[1])
        if layer != FAULT_LAYER:
            return layer > FAULT_LAYER
        if len(parts) == 2:
            return True
        block = parts[2]
        if block in {"input_layernorm", "self_attn", "post_attention_layernorm"}:
            return False
        if block == "mlp":
            if len(parts) == 3:
                return True
            return parts[3] == "down_proj"
        return False

    def _infer_trailing_failure(self, boundary_id: str) -> bool:
        """Infer the last untested child from the failing parent and prior pass."""
        parent_id = self.inventory.parents.get(boundary_id)
        if parent_id is None:
            return False
        chain = serial_chain_children(
            self.inventory, parent_id, graph=self.package.graph
        )
        index = next(
            (position for position, child in enumerate(chain) if child.id == boundary_id),
            None,
        )
        if index is None or index != len(chain) - 1:
            return False
        if any(self.states.get(child.id) == "failing" for child in chain[:index]):
            return False
        verified = [
            position
            for position, child in enumerate(chain)
            if self.states.get(child.id) == "verified"
        ]
        return index == max(verified, default=-1) + 1

    def run(self, *, max_steps: int = 32) -> None:
        for _ in range(max_steps):
            if self.current_failure == self.fault:
                return
            plan = plan_alignment(self.package, session=self.session())
            assert plan.candidates, "planner produced no candidate"
            candidate = plan.candidates[0]
            self.selected.append(candidate.boundary)
            state = self.states.get(candidate.boundary)
            if state == "failing":
                # The binary search collapsed onto a known failing child.
                self.current_failure = candidate.boundary
                continue
            if state is not None:
                raise AssertionError(f"planner re-proposed {state} boundary {candidate.boundary}")
            if self._infer_trailing_failure(candidate.boundary):
                self.states[candidate.boundary] = "failing"
                self.inferred += 1
                self.current_failure = candidate.boundary
                continue
            self.evaluations += 1
            name = self.inventory.resolve(candidate.boundary).name  # type: ignore[union-attr]
            self.states[candidate.boundary] = (
                "failing" if self._prefix_fails(name) else "verified"
            )
        raise AssertionError("binary search did not isolate the injected failure")


def _parent_of(inventory: BoundaryInventory, boundary_id: str | None) -> str | None:
    if boundary_id is None:
        return None
    return inventory.parents.get(boundary_id)


def test_failing_serial_chain_starts_at_the_midpoint() -> None:
    package = _build_hierarchical_package()
    inventory = inventory_boundaries(package)
    whole = _boundary_id(inventory, WHOLE_MODEL)

    plan = plan_alignment(package, session=_failing_session(inventory, whole))

    assert plan.candidates[0].boundary == _boundary_id(
        inventory, "TransformerBlock@layers.14"
    )
    assert BINARY_SEARCH_REASON in plan.candidates[0].reasons
    # The second computation is identical: no set/dict iteration ordering
    # leaks into the midpoint decision.
    assert plan.to_dict() == plan_alignment(
        package, session=_failing_session(inventory, whole)
    ).to_dict()


def test_binary_search_isolates_28_layer_failure_in_at_most_eight_runs() -> None:
    package = _build_hierarchical_package()
    inventory = inventory_boundaries(package)
    runner = _FakeSerialRunner(package)

    runner.run()

    assert runner.current_failure == runner.fault
    assert runner.evaluations <= 8
    assert runner.evaluations == 7
    assert runner.inferred == 2
    selected_names = [
        inventory.resolve(boundary_id).name for boundary_id in runner.selected  # type: ignore[union-attr]
    ]
    assert selected_names[:5] == [
        "TransformerBlock@layers.14",
        "TransformerBlock@layers.21",
        "TransformerBlock@layers.18",
        "TransformerBlock@layers.20",
        "TransformerBlock@layers.19",
    ]
    assert selected_names[-3:] == [
        "SwiGLU@layers.19.mlp",
        "Linear@layers.19.mlp.up_proj",
        "Linear@layers.19.mlp.down_proj",
    ]


def test_narrowing_orders_serial_children_by_midpoint_and_is_deterministic() -> None:
    package = _build_hierarchical_package()
    inventory = inventory_boundaries(package)
    whole = _boundary_id(inventory, WHOLE_MODEL)
    layer_14 = _boundary_id(inventory, "TransformerBlock@layers.14")
    layer_21 = _boundary_id(inventory, "TransformerBlock@layers.21")

    evidence = {whole: [_Comparison("fail", {})]}
    first = compute_narrowing(inventory, evidence, package_or_graph=package)
    second = compute_narrowing(inventory, evidence, package_or_graph=package)

    assert first.to_dict() == second.to_dict()
    assert first.smallest_confirmed_failure == whole
    assert first.next_candidates[0] == layer_14
    assert _parent_of(inventory, first.next_candidates[0]) == whole

    evidence[layer_14] = [_Comparison("pass", {})]
    narrowed = compute_narrowing(inventory, evidence, package_or_graph=package)
    assert narrowed.next_candidates[0] == layer_21
    assert narrowed.to_dict() == compute_narrowing(
        inventory, evidence, package_or_graph=package
    ).to_dict()

    session = {
        "current_failure": whole,
        "boundaries": {
            whole: {"state": "failing"},
            layer_14: {"state": "verified"},
        },
    }
    assert (
        plan_alignment(package, session=session).candidates[0].boundary
        == narrowed.next_candidates[0]
    )


def test_narrowing_selects_the_deepest_failing_region_deterministically() -> None:
    package = _build_hierarchical_package()
    inventory = inventory_boundaries(package)
    whole = _boundary_id(inventory, WHOLE_MODEL)
    layer_19 = _boundary_id(inventory, "TransformerBlock@layers.19")
    fault = _boundary_id(inventory, FAULT_NAME)

    evidence = {
        whole: [_Comparison("fail", {})],
        layer_19: [_Comparison("fail", {})],
        fault: [_Comparison("fail", {})],
    }
    state = compute_narrowing(inventory, evidence, package_or_graph=package)

    assert state.smallest_confirmed_failure == fault
    assert fault in state.failing_descendants
    assert _parent_of(inventory, state.smallest_confirmed_failure) == _boundary_id(
        inventory, "SwiGLU@layers.19.mlp"
    )
