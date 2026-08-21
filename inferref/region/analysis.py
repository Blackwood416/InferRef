"""Region boundary preview and property analysis (SPEC §10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.region.boundary import derive_boundary


@dataclass(frozen=True)
class RegionDetails:
    id: str | int
    name: str
    operators: int
    inputs: int
    outputs: int
    activation_bytes: int
    parameter_bytes: int
    largest_tensor: int
    payload_coverage: float
    reproducible: bool
    mutation: bool
    semantic_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "operators": self.operators,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "activation_bytes": self.activation_bytes,
            "parameter_bytes": {
                "estimated": self.parameter_bytes,
            },
            "largest_tensor": self.largest_tensor,
            "payload_coverage": self.payload_coverage,
            "reproducible": self.reproducible,
            "mutation": self.mutation,
            "semantic_confidence": self.semantic_confidence,
        }


def analyze_region(package: TracePackage, region: RegionRecord | Any) -> RegionDetails:
    """Compute detailed boundary and reproducibility metrics for one region."""
    graph = package.graph
    ops_inside = [graph.op(nid) for nid in region.node_ids if graph.has_op(nid)]
    num_ops = len(ops_inside)

    input_ids, output_ids = region.inputs, region.outputs
    if not input_ids and not output_ids and region.node_ids:
        derived_in, derived_out = derive_boundary(graph, region.node_ids)
        input_ids = tuple(derived_in)
        output_ids = tuple(derived_out)

    activation_bytes = 0
    parameter_bytes = 0
    largest_tensor = 0
    total_boundary_tensors = 0
    payload_tensors = 0

    for vid in input_ids:
        if not graph.has_value(vid):
            continue
        val = graph.value(vid)
        nbytes = val.logical_nbytes
        total_boundary_tensors += 1
        largest_tensor = max(largest_tensor, nbytes)
        if val.capture.payload is not None:
            payload_tensors += 1

        # Parameter heuristic: value.producer is None => estimated parameter
        if val.producer is None:
            parameter_bytes += nbytes
        else:
            activation_bytes += nbytes

    for vid in output_ids:
        if not graph.has_value(vid):
            continue
        val = graph.value(vid)
        nbytes = val.logical_nbytes
        total_boundary_tensors += 1
        largest_tensor = max(largest_tensor, nbytes)
        if val.capture.payload is not None:
            payload_tensors += 1
        activation_bytes += nbytes

    payload_coverage = (
        round(payload_tensors / total_boundary_tensors, 4)
        if total_boundary_tensors > 0
        else 1.0
    )

    # Check mutations inside region
    has_mutation = False
    for op in ops_inside:
        if op.effects.mutated_storages:
            has_mutation = True
            break
        op_name = getattr(op, "op", "") or getattr(op, "name", "")
        if op_name.endswith("_") and not op_name.startswith("__"):
            has_mutation = True
            break

    # Reproducibility: complete payload coverage and no missing payloads
    reproducible = (payload_coverage == 1.0) and (total_boundary_tensors > 0 or num_ops > 0)

    # Semantic confidence
    semantic_conf: float | None = None
    if getattr(region, "semantic", None) is not None:
        # Check if semantic detection annotation is in graph
        for op in ops_inside:
            if any(getattr(a, "type", None) == "semantic" for a in getattr(op, "annotations", ())):
                semantic_conf = 1.0
                break
        if semantic_conf is None:
            semantic_conf = 0.95

    return RegionDetails(
        id=region.id,
        name=region.name,
        operators=num_ops,
        inputs=len(input_ids),
        outputs=len(output_ids),
        activation_bytes=activation_bytes,
        parameter_bytes=parameter_bytes,
        largest_tensor=largest_tensor,
        payload_coverage=payload_coverage,
        reproducible=reproducible,
        mutation=has_mutation,
        semantic_confidence=semantic_conf,
    )
