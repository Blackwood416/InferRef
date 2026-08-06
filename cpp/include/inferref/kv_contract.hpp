// Bounds validation for the kv-cache/indexed-update/v1 contract.
//
// The index tensor is stored as float32 for corpus compactness, but the value
// is semantically an integer position.  All range checks happen on the float
// value *before* conversion to size_t, and the target + update_sequence sum is
// checked in a form that cannot overflow or underflow unsigned arithmetic.

#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace inferref
{

// Returns an empty string when `raw_index` is a valid indexed-update target,
// otherwise a human-readable diagnostic suitable for an invalid_testcase
// report.
inline std::string ValidateIndexedUpdate(double raw_index,
                                         std::size_t old_sequence,
                                         std::size_t update_sequence)
{
    if (!std::isfinite(raw_index) || raw_index < 0.0 ||
        std::floor(raw_index) != raw_index)
        return "index must be one finite non-negative integer";
    // Converting only after the upper bound keeps a huge finite float (for
    // example 1e300) from becoming an implementation-defined size_t value.
    if (raw_index > static_cast<double>(old_sequence))
        return "index must not exceed the cache sequence length";
    const std::size_t target = static_cast<std::size_t>(raw_index);
    if (update_sequence > old_sequence ||
        target > old_sequence - update_sequence)
        return "target + update sequence must not exceed the cache sequence length";
    return "";
}

} // namespace inferref
