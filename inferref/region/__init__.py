"""Reference regions: named subgraphs with explicit boundaries (SPEC §18)."""

from inferref.region.boundary import derive_boundary, nodes_between, nodes_for_module
from inferref.region.manager import (
    RegionError,
    create_region,
    create_region_from_module,
    create_region_from_ops,
    create_region_from_source_function,
    delete_region,
    refresh_boundaries,
)

__all__ = [
    "RegionError",
    "create_region",
    "create_region_from_module",
    "create_region_from_ops",
    "create_region_from_source_function",
    "delete_region",
    "derive_boundary",
    "nodes_between",
    "nodes_for_module",
    "refresh_boundaries",
]
