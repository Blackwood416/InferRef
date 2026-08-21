"""InferRef Comparison Package (SPEC §6).

Provides ComparisonSpec wire format definition, validation, two-axis tolerance
resolution, and effective comparison policy generation.
"""

from inferref.comparison.resolution import (
    EffectiveComparison,
    resolve_comparison_policy,
)
from inferref.comparison.schema import (
    COMPARISON_SPEC_FORMAT,
    COMPARISON_SPEC_READ_VERSIONS,
    COMPARISON_SPEC_VERSION,
    ComparisonSpec,
    ComparisonSpecValidationError,
    OutputComparisonSpec,
    validate_comparison_spec,
)

__all__ = [
    "COMPARISON_SPEC_FORMAT",
    "COMPARISON_SPEC_READ_VERSIONS",
    "COMPARISON_SPEC_VERSION",
    "ComparisonSpec",
    "ComparisonSpecValidationError",
    "EffectiveComparison",
    "OutputComparisonSpec",
    "resolve_comparison_policy",
    "validate_comparison_spec",
]
