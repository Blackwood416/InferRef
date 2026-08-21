"""Reference regions: named subgraphs with explicit boundaries (SPEC §18)."""

from inferref.region.analysis import RegionDetails, analyze_region
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
from inferref.region.recommend import RegionRecommendation, recommend_regions

__all__ = [
    "RegionDetails",
    "RegionError",
    "RegionRecommendation",
    "analyze_region",
    "create_region",
    "create_region_from_module",
    "create_region_from_ops",
    "create_region_from_source_function",
    "delete_region",
    "derive_boundary",
    "nodes_between",
    "nodes_for_module",
    "recommend_regions",
    "refresh_boundaries",
]
