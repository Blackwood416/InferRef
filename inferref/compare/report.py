"""Human-readable comparison reports (SPEC §2.6, Appendix B).

JSON output is produced directly by :meth:`ComparisonReport.to_dict`; this
module renders the text form an engineer reads in a terminal.
"""

from __future__ import annotations

from inferref.compare.compare import (
    STATUS_MISSING,
    STATUS_PASS,
    ComparisonReport,
    TensorComparison,
    upstream_context,
)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e6:
        return f"{value:.6e}"
    return f"{value:.8g}"


def render_report(report: ComparisonReport, *, verbose: bool = False) -> str:
    """Render a full comparison report as text."""
    lines: list[str] = []
    lines.append("InferRef Comparison")
    lines.append("")
    lines.append(f"Reference:  {report.reference}")
    lines.append(f"Engine:     {report.actual}")
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Compared tensors: {len(report.comparisons)}")
    lines.append(f"  Passed:           {report.passed_count}")
    lines.append(f"  Failed:           {report.failed_count}")
    if report.missing_count:
        lines.append(f"  Missing:          {report.missing_count}")
    if report.stopped_early:
        lines.append("  (stopped at first failure)")
    lines.append("")

    if verbose:
        lines.append("Tensors:")
        for comparison in report.comparisons:
            mark = "PASS" if comparison.passed else comparison.status.upper()
            detail = ""
            if comparison.metrics is not None:
                detail = f"  max_abs_error={_fmt(comparison.metrics.max_abs_error)}"
            lines.append(f"  {mark:7s} {comparison.name}{detail}")
        lines.append("")

    first = report.first_failure
    if first is None:
        lines.append("Result: PASS")
        return "\n".join(lines)

    lines.extend(_render_failure(first))

    upstream = upstream_context(report)
    if upstream:
        lines.append("")
        lines.append("Upstream status:")
        for comparison in upstream:
            mark = "PASS" if comparison.passed else comparison.status.upper()
            lines.append(f"  {comparison.name}  {mark}")

    lines.append("")
    lines.append("Result: FAIL")
    return "\n".join(lines)


def _render_failure(first: TensorComparison) -> list[str]:
    lines: list[str] = ["First divergence:", ""]
    lines.append(f"  Tensor:    {first.name}")
    if first.value_id is not None:
        lines.append(f"  Value id:  {first.value_id}")
    if first.operator:
        producer = f"#{first.execution_index} {first.operator}"
        lines.append(f"  Producer:  {producer}")
    if first.module_path:
        lines.append(f"  Module:    {first.module_path}")
    if first.region:
        lines.append(f"  Region:    {first.region}")
    if first.source:
        lines.append(f"  Source:    {first.source}")

    if first.status == STATUS_MISSING:
        lines.append("")
        lines.append(f"  {first.message}")
        return lines

    layout = first.layout
    if layout is not None:
        lines.append(f"  Shape:     {list(layout.reference_shape)}")
        lines.append(f"  DType:     {layout.reference_dtype}")
        if layout.any_mismatch:
            lines.append("")
            lines.append("Layout mismatch:")
            if layout.dtype_mismatch:
                lines.append(
                    f"  dtype:          {layout.reference_dtype} -> {layout.actual_dtype}"
                )
            if layout.shape_mismatch:
                lines.append(
                    f"  shape:          {list(layout.reference_shape)} "
                    f"-> {list(layout.actual_shape)}"
                )
            if layout.stride_mismatch:
                lines.append(
                    f"  stride:         {list(layout.reference_stride)} "
                    f"-> {list(layout.actual_stride)}"
                )
            if layout.storage_offset_mismatch:
                lines.append(
                    f"  storage_offset: {layout.reference_storage_offset} "
                    f"-> {layout.actual_storage_offset}"
                )

    if first.message:
        lines.append("")
        lines.append(f"  {first.message}")

    metrics = first.metrics
    if metrics is not None:
        lines.append("")
        lines.append("Metrics:")
        lines.append(f"  max_abs_error:     {_fmt(metrics.max_abs_error)}")
        lines.append(f"  max_rel_error:     {_fmt(metrics.max_rel_error)}")
        lines.append(f"  mean_abs_error:    {_fmt(metrics.mean_abs_error)}")
        lines.append(f"  rmse:              {_fmt(metrics.rmse)}")
        lines.append(f"  cosine_similarity: {_fmt(metrics.cosine_similarity)}")
        lines.append(
            f"  mismatched:        {metrics.mismatch_count} / {metrics.element_count}"
        )
        lines.append(f"  tolerance:         atol={_fmt(first.atol)} rtol={_fmt(first.rtol)}")
        if metrics.nan_count or metrics.inf_count:
            lines.append(
                f"  engine nan/inf:    {metrics.nan_count} / {metrics.inf_count}"
            )
        if metrics.reference_nan_count or metrics.reference_inf_count:
            lines.append(
                f"  reference nan/inf: {metrics.reference_nan_count} / "
                f"{metrics.reference_inf_count}"
            )

        if metrics.first_mismatch_index is not None:
            lines.append("")
            lines.append("First mismatching element:")
            lines.append(f"  index:     {list(metrics.first_mismatch_index)}")
            lines.append(f"  reference: {_fmt(metrics.first_mismatch_reference)}")
            lines.append(f"  actual:    {_fmt(metrics.first_mismatch_actual)}")
    return lines


def render_short(report: ComparisonReport) -> str:
    """One-line result, for CI logs."""
    if report.status == STATUS_PASS:
        return f"PASS ({len(report.comparisons)} tensors)"
    first = report.first_failure
    where = first.name if first else "?"
    return f"FAIL at {where} ({report.failed_count + report.missing_count} bad)"
