"""Boundary inventory and deterministic alignment planning (0.10 L2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferref.alignment.boundaries import (
    BoundaryInventory,
    inventory_boundaries,
    select_causal_descendants,
)
from inferref.alignment.planner import plan_alignment
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef


def _package() -> TracePackage:
    graph = Graph()
    graph.values = [
        TensorValueRecord(
            id=value_id,
            shape=(2,),
            producer=producer,
            capture=CaptureInfo(mode="full", payload=f"v{value_id}.irtensor"),
        )
        for value_id, producer in ((1, None), (2, 10), (3, 11), (4, 12))
    ]
    graph.operators = [
        OperatorRecord(
            id=10,
            execution_index=0,
            op="one",
            result=TensorRef(2),
            positional_args=(TensorRef(1),),
            module_stack=(100,),
        ),
        OperatorRecord(
            id=11,
            execution_index=1,
            op="two",
            result=TensorRef(3),
            positional_args=(TensorRef(2),),
            module_stack=(100, 101),
        ),
        OperatorRecord(
            id=12,
            execution_index=2,
            op="three",
            result=TensorRef(4),
            positional_args=(TensorRef(3),),
            module_stack=(100, 102),
        ),
    ]
    graph.reindex()
    return TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[
            ModuleRecord(id=100, path="model"),
            ModuleRecord(id=101, path="model.left"),
            ModuleRecord(id=102, path="model.right"),
        ],
        regions=[
            RegionRecord(id=1, name="parent", node_ids=(10, 11, 12)),
            RegionRecord(id=2, name="left", node_ids=(10, 11)),
            RegionRecord(id=3, name="overlap", node_ids=(11, 12)),
        ],
    )


def test_inventory_canonicalizes_and_builds_dag() -> None:
    inventory = inventory_boundaries(_package())

    # module:100 and region:1 have the same operator set; the region wins.
    assert inventory.aliases["module:100"] == "region:1"
    assert inventory.resolve("module:100").id == "region:1"  # type: ignore[union-attr]
    assert inventory.parents["region:2"] == "region:1"
    assert inventory.parents["module:101"] == "region:2"
    # The overlap is also a minimal parent of the module containing op 11.
    assert "region:3" in inventory.alternate_parents["module:101"]
    assert "region:2" in inventory.child_ids("region:1")
    assert "region:3" in inventory.child_ids("region:1")

    # Public inventory output is JSON-ready and does not retain the package.
    payload = inventory.to_dict()
    assert payload["aliases"]["module:100"] == "region:1"
    assert payload["boundaries"]


def test_causal_descendants_use_first_failure_value_provenance() -> None:
    package = _package()
    inventory = inventory_boundaries(package)
    selection = select_causal_descendants(
        package,
        inventory,
        "region:1",
        {"value_id": 3},
    )

    assert selection.confidence == "high"
    assert selection.provenance_available is True
    assert selection.operator_ids == (10, 11)
    # The right child includes the failing producer (op 11) through overlap;
    # the leaf op 12 is downstream and is not causally selected.
    assert "region:2" in selection.boundary_ids
    assert "region:3" in selection.boundary_ids
    assert "module:102" not in selection.boundary_ids


def test_causal_fallback_is_explicitly_low_confidence() -> None:
    package = _package()
    inventory = inventory_boundaries(package)
    selection = select_causal_descendants(package, inventory, "region:1")

    assert selection.confidence == "low"
    assert selection.provenance_available is False
    assert set(selection.boundary_ids) == set(inventory.descendants("region:1"))


def test_plan_is_read_only_deterministic_and_explains_modifiers(tmp_path: Path) -> None:
    package = _package()
    session = {
        "required_fixture_kinds": ["reference_derived"],
        "boundaries": {
            "region:1": {"state": "failing"},
            "region:2": {"state": "untested"},
            "op:10": {"state": "verified"},
        },
        "narrowing": {
            "current_failure": {
                "boundary": "region:1",
                "first_failure": {"value_id": 3},
            }
        },
        "testcase_cache": {
            "region:2": {"path": "testcases/left", "reproducible": True}
        },
    }
    first = plan_alignment(package, session=session)
    second = plan_alignment(package, session=session)

    assert [candidate.to_dict() for candidate in first] == [
        candidate.to_dict() for candidate in second
    ]
    assert first.current_failure == "region:1"
    assert first.causal_confidence == "high"
    assert first.candidates
    selected = first.candidates[0]
    assert selected.boundary == "region:2"
    assert selected.fixture_kind == "reference_derived"
    assert selected.action == "extract_and_run"
    assert any("immediate child" in reason for reason in selected.reasons)
    assert any("reusable validated testcase" in reason for reason in selected.reasons)
    # The active failure remains inspectable for a rerun; its relevant children
    assert "region:1" in {candidate.boundary for candidate in first}
    assert "op:10" not in {candidate.boundary for candidate in first}
    assert not list(tmp_path.iterdir())


def test_empty_regions_are_excluded() -> None:
    package = _package()
    # Add an empty region with 0 node IDs
    package.regions.append(RegionRecord(id=99, name="empty_region", node_ids=()))
    inventory = inventory_boundaries(package)

    assert inventory.resolve("region:99") is None
    assert "region:99" not in inventory.aliases
    assert not any(boundary.id == "region:99" for boundary in inventory.boundaries)
    assert not any(boundary.id == "region:99" for boundary in inventory.raw_boundaries)


def test_inventory_lossless_round_trip_and_validation() -> None:
    package = _package()
    inventory = inventory_boundaries(package)
    serialized = inventory.to_dict()

    assert "raw_boundaries" in serialized
    restored = BoundaryInventory.from_dict(serialized)

    assert len(restored.boundaries) == len(inventory.boundaries)
    assert len(restored.raw_boundaries) == len(inventory.raw_boundaries)
    assert restored.aliases == inventory.aliases
    assert restored.parents == inventory.parents
    assert restored.alternate_parents == inventory.alternate_parents
    assert restored.children == inventory.children

    # Aliases resolve to canonical boundary identities
    for alias_id, canonical_id in inventory.aliases.items():
        assert restored.resolve(alias_id).id == canonical_id  # type: ignore[union-attr]

    # Test validation: dangling alias target raises ValueError
    bad_alias = dict(serialized)
    bad_alias["aliases"] = {"module:999": "region:nonexistent"}
    with pytest.raises(ValueError, match="dangling alias target"):
        BoundaryInventory.from_dict(bad_alias)

    # Test validation: dangling parent ID raises ValueError
    bad_parent = dict(serialized)
    bad_parent["parents"] = {"region:2": "region:nonexistent"}
    with pytest.raises(ValueError, match="dangling parent ID"):
        BoundaryInventory.from_dict(bad_parent)

    # Test validation: parent cycle raises ValueError
    cycle = dict(serialized)
    cycle["parents"] = {"region:1": "region:2", "region:2": "region:1"}
    with pytest.raises(ValueError, match="cycle detected"):
        BoundaryInventory.from_dict(cycle)


def test_invalid_first_failure_value_id_falls_back_to_containment() -> None:
    package = _package()
    inventory = inventory_boundaries(package)

    # Value 99999 does not exist in graph; even if producer op 10 exists,
    # provenance must be rejected rather than reported as valid.
    selection = select_causal_descendants(
        package,
        inventory,
        "region:1",
        {"value_id": 99999, "producer_op_id": 10},
    )

    assert selection.confidence == "low"
    assert selection.provenance_available is False
    assert selection.value_id is None or not package.graph.has_value(selection.value_id)
    assert set(selection.boundary_ids) == set(inventory.descendants("region:1"))
