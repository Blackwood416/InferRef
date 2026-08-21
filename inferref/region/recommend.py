"""Region recommendation scoring and ranking (SPEC §10.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferref.ir.package import TracePackage
from inferref.region.analysis import RegionDetails, analyze_region


@dataclass(frozen=True)
class RegionRecommendation:
    region_id: str
    region_name: str
    score: int
    details: RegionDetails
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "name": self.region_name,
            "score": self.score,
            "details": self.details.to_dict(),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def recommend_regions(
    package: TracePackage,
    *,
    min_score: int | None = None,
    top: int | None = None,
) -> list[RegionRecommendation]:
    """Score and rank regions deterministically for engine extraction."""
    recommendations: list[RegionRecommendation] = []

    for region in package.regions:
        details = analyze_region(package, region)
        score = 0
        reasons: list[str] = []
        warnings: list[str] = []

        # + payload complete
        if details.payload_coverage >= 1.0:
            score += 30
            reasons.append("+30: complete payload coverage")
        else:
            score -= 20
            reasons.append(f"-20: incomplete payload coverage ({details.payload_coverage:.0%})")

        # + reproducible
        if details.reproducible:
            score += 25
            reasons.append("+25: independently reproducible")
        else:
            score -= 15
            reasons.append("-15: not reproducible from boundary")

        # + semantic confidence
        if details.semantic_confidence is not None:
            bonus = int(details.semantic_confidence * 20)
            score += bonus
            reasons.append(f"+{bonus}: semantic confidence ({details.semantic_confidence:.2f})")

        # + small boundary / warnings
        total_boundary_bytes = details.activation_bytes + details.parameter_bytes
        if total_boundary_bytes < 50 * 1024 * 1024:
            score += 15
            reasons.append("+15: compact boundary footprint")
        elif total_boundary_bytes > 200 * 1024 * 1024:
            warnings.append(f"large boundary footprint ({total_boundary_bytes / (1024*1024):.1f} MB)")

        # + reasonable node count
        if 2 <= details.operators <= 150:
            score += 10
            reasons.append(f"+10: reasonable operator count ({details.operators})")

        # - excessive parameter size
        if details.parameter_bytes > 500 * 1024 * 1024:
            score -= 20
            reasons.append(f"-20: excessive parameter size ({details.parameter_bytes / (1024*1024):.1f} MB)")

        # - excessive I/O count
        if details.inputs + details.outputs > 10:
            score -= 15
            reasons.append(f"-15: high I/O count ({details.inputs} inputs, {details.outputs} outputs)")

        # - unresolved mutation
        if details.mutation:
            score -= 25
            reasons.append("-25: region contains in-place mutation effects")

        rec = RegionRecommendation(
            region_id=region.id,
            region_name=region.name,
            score=score,
            details=details,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )
        if min_score is None or score >= min_score:
            recommendations.append(rec)

    # Sort deterministically by score descending, then name ascending
    recommendations.sort(key=lambda r: (-r.score, r.region_name, r.region_id))
    if top is not None and top > 0:
        recommendations = recommendations[:top]
    return recommendations
