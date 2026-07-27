// InferRef numerical comparison — header-only, dependency-free.
//
// Implements the metrics of SPEC §27 and the layout distinction of SPEC §29,
// so an engine's own test harness can produce the same verdict as
// `inferref compare` without shelling out to Python.

#ifndef INFERREF_COMPARE_HPP
#define INFERREF_COMPARE_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "inferref/irtensor.hpp"

namespace inferref
{

// Per-dtype tolerances, matching inferref/compare/tolerance.py.
struct Tolerance
{
    double atol = 1e-5;
    double rtol = 1e-5;

    static Tolerance ForDataType(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::kFloat64: return {1e-9, 1e-9};
        case DataType::kFloat32: return {1e-5, 1e-5};
        case DataType::kFloat16: return {2e-3, 2e-3};
        case DataType::kBFloat16: return {1.6e-2, 1.6e-2};
        default: return {0.0, 0.0}; // integral types must match exactly
        }
    }
};

// SPEC §29: shape / stride / storage-offset / value mismatches stay distinct.
struct LayoutDiff
{
    bool dtype_mismatch = false;
    bool shape_mismatch = false;
    bool stride_mismatch = false;
    bool storage_offset_mismatch = false;

    bool Comparable() const { return !dtype_mismatch && !shape_mismatch; }
    bool AnyMismatch() const
    {
        return dtype_mismatch || shape_mismatch || stride_mismatch ||
               storage_offset_mismatch;
    }
};

// SPEC §27 metrics.
struct Metrics
{
    bool exact_match = false;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
    double cosine_similarity = 1.0;
    std::int64_t nan_count = 0;
    std::int64_t inf_count = 0;
    std::int64_t mismatch_count = 0;
    std::int64_t element_count = 0;

    bool has_first_mismatch = false;
    std::int64_t first_mismatch_flat_index = -1;
    std::vector<std::int64_t> first_mismatch_index;
    double first_mismatch_reference = 0.0;
    double first_mismatch_actual = 0.0;
};

struct ComparisonResult
{
    LayoutDiff layout;
    Metrics metrics;
    bool passed = false;
    std::string message;
};

inline LayoutDiff DiffLayout(const IRTensor &reference, const IRTensor &actual)
{
    LayoutDiff diff;
    diff.dtype_mismatch = reference.dtype != actual.dtype;
    diff.shape_mismatch = reference.shape != actual.shape;
    diff.stride_mismatch = reference.stride != actual.stride;
    diff.storage_offset_mismatch = reference.storage_offset != actual.storage_offset;
    return diff;
}

// Unravel a flat index into multi-dimensional coordinates (row-major).
inline std::vector<std::int64_t> UnravelIndex(std::int64_t flat,
                                              const std::vector<std::int64_t> &shape)
{
    std::vector<std::int64_t> index(shape.size(), 0);
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        if (shape[i] == 0)
        {
            continue;
        }
        index[i] = flat % shape[i];
        flat /= shape[i];
    }
    return index;
}

inline Metrics ComputeMetrics(const std::vector<float> &reference,
                              const std::vector<float> &actual,
                              const std::vector<std::int64_t> &shape,
                              const Tolerance &tolerance)
{
    Metrics metrics;
    metrics.element_count = static_cast<std::int64_t>(reference.size());
    if (reference.size() != actual.size())
    {
        metrics.mismatch_count = metrics.element_count;
        return metrics;
    }
    if (reference.empty())
    {
        metrics.exact_match = true;
        return metrics;
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_act = 0.0;
    std::int64_t finite_count = 0;
    bool all_equal = true;

    for (std::size_t i = 0; i < reference.size(); ++i)
    {
        const double r = static_cast<double>(reference[i]);
        const double a = static_cast<double>(actual[i]);

        if (std::isnan(a))
        {
            ++metrics.nan_count;
        }
        if (std::isinf(a))
        {
            ++metrics.inf_count;
        }

        const bool finite = std::isfinite(r) && std::isfinite(a);
        bool within = false;
        if (finite)
        {
            const double diff = std::fabs(a - r);
            within = diff <= tolerance.atol + tolerance.rtol * std::fabs(r);

            metrics.max_abs_error = std::max(metrics.max_abs_error, diff);
            if (std::fabs(r) > 0.0)
            {
                metrics.max_rel_error = std::max(metrics.max_rel_error, diff / std::fabs(r));
            }
            sum_abs += diff;
            sum_sq += diff * diff;
            dot += r * a;
            norm_ref += r * r;
            norm_act += a * a;
            ++finite_count;
            if (r != a)
            {
                all_equal = false;
            }
        }
        else
        {
            // Non-finite entries agree only if they are the same kind.
            within = (std::isnan(r) && std::isnan(a)) ||
                     (std::isinf(r) && std::isinf(a) && std::signbit(r) == std::signbit(a));
            if (!within)
            {
                all_equal = false;
            }
        }

        if (!within)
        {
            ++metrics.mismatch_count;
            if (!metrics.has_first_mismatch)
            {
                metrics.has_first_mismatch = true;
                metrics.first_mismatch_flat_index = static_cast<std::int64_t>(i);
                metrics.first_mismatch_index =
                    UnravelIndex(static_cast<std::int64_t>(i), shape);
                metrics.first_mismatch_reference = r;
                metrics.first_mismatch_actual = a;
            }
        }
    }

    metrics.exact_match = all_equal;
    if (finite_count > 0)
    {
        metrics.mean_abs_error = sum_abs / static_cast<double>(finite_count);
        metrics.rmse = std::sqrt(sum_sq / static_cast<double>(finite_count));
    }
    if (norm_ref > 0.0 && norm_act > 0.0)
    {
        metrics.cosine_similarity = dot / (std::sqrt(norm_ref) * std::sqrt(norm_act));
        metrics.cosine_similarity =
            std::max(-1.0, std::min(1.0, metrics.cosine_similarity));
    }
    else if (norm_ref == 0.0 && norm_act == 0.0)
    {
        metrics.cosine_similarity = 1.0;
    }
    else
    {
        metrics.cosine_similarity = 0.0;
    }
    return metrics;
}

// Compare two tensors. Layout differences are reported but, matching the
// Python comparator's default, do not by themselves fail: a fused engine is
// not required to reproduce the reference layout (SPEC §20).
inline ComparisonResult CompareTensors(const IRTensor &reference, const IRTensor &actual,
                                       bool strict_layout = false)
{
    ComparisonResult result;
    result.layout = DiffLayout(reference, actual);

    if (!result.layout.Comparable())
    {
        result.passed = false;
        result.message = "shape or dtype mismatch";
        return result;
    }

    const Tolerance tolerance = Tolerance::ForDataType(reference.dtype);
    result.metrics = ComputeMetrics(reference.AsFloat32(), actual.AsFloat32(),
                                    reference.shape, tolerance);

    const bool values_ok = result.metrics.mismatch_count == 0;
    const bool layout_ok = !result.layout.AnyMismatch();
    result.passed = values_ok && (layout_ok || !strict_layout);

    if (values_ok && !layout_ok)
    {
        result.message =
            "values are identical when interpreted logically, but layout differs";
    }
    else if (!values_ok)
    {
        result.message = "value mismatch";
    }
    return result;
}

} // namespace inferref

#endif // INFERREF_COMPARE_HPP
