// Explicit quantization descriptors and the float32 reference dequantizer.
//
// Testcase format 0.4 attaches a QuantizationSpec to a value and stores the
// referenced scale, zero-point, and codebook tensors as ordinary immutable
// parameters.  This header parses that descriptor strictly (mirroring
// inferref/quantization/spec.py) and reconstructs logical float32 values for
// affine-int8/v1, affine-int4/v1, affine-uint4/v1, and nf4/v1.
//
// The arithmetic order matches inferref/quantization/reference.py and
// inferref/quantization/nf4.py exactly: affine is
// ``(float32(code) - zero_point) * scale`` and NF4 is
// ``codebook[index] * scale[flat_index / group_size]``, both in float32.
// Unrepresentable (infinite) results are rejected rather than returned, like
// the Python reference's ``errstate(over="raise")``.
//
// Header-only and dependency-free: only the C++ standard library and the
// InferRef JSON/IRTensor headers are used.

#ifndef INFERREF_QUANTIZATION_HPP
#define INFERREF_QUANTIZATION_HPP

#include <inferref/irtensor.hpp>
#include <inferref/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferref
{

class QuantizationError : public std::runtime_error
{
public:
    explicit QuantizationError(const std::string &what)
        : std::runtime_error(what) {}
};

inline constexpr const char *kQuantizationFormat = "inferref-quantization";
inline constexpr const char *kQuantizationFormatVersion = "1.0";
inline constexpr const char *kNf4CodebookIdentity = "bitsandbytes-nf4/v1";

// The exact bitsandbytes-nf4/v1 table as float32 bit patterns.  The Python
// reference rejects any other codebook, so the native reader validates the
// identity by bits rather than by decimal text.
inline constexpr std::uint32_t kNf4CodebookBits[16] = {
    0xbf800000u, 0xbf3239b1u, 0xbf066b30u, 0xbeca32a0u,
    0xbe91a24du, 0xbe3d353fu, 0xbdba7871u, 0x00000000u,
    0x3da2faffu, 0x3e24cae3u, 0x3e7c04ddu, 0x3ead033au,
    0x3ee1a4b8u, 0x3f1007abu, 0x3f3913b3u, 0x3f800000u,
};

// One validated descriptor.  Presence flags distinguish an explicit null from
// an absent field, matching the Python dataclass's ``None`` semantics.
struct QuantizationSpec
{
    std::string format;
    std::string format_version;
    std::string scheme;
    std::string storage_dtype;
    std::vector<std::int64_t> logical_shape;
    std::int64_t scale_value_id = -1;
    bool has_zero_point = false;
    std::int64_t zero_point_value_id = -1;
    bool has_codebook = false;
    std::int64_t codebook_value_id = -1;
    std::string codebook;
    std::string granularity;
    bool has_axis = false;
    std::int64_t axis = 0;
    bool has_group_size = false;
    std::int64_t group_size = 0;
    std::string compute_dtype;
    std::string rounding;
    std::string saturation;

    bool IsNf4() const { return scheme == "nf4/v1"; }

    // Inclusive range of legal storage codes.
    std::pair<std::int64_t, std::int64_t> CodeRange() const
    {
        if (storage_dtype == "int8") return {-128, 127};
        if (storage_dtype == "int4") return {-8, 7};
        return {0, 15};
    }

    // Dtype of the referenced zero-point tensor (uint8 for unsigned storage).
    std::string ZeroPointDtype() const
    {
        return storage_dtype == "uint4" ? "uint8" : "int8";
    }

    // Exact required shape of both scale and (when present) zero point.
    // NF4 blocks cover the flattened logical tensor, not each row.
    std::vector<std::int64_t> ParameterShape() const
    {
        if (IsNf4())
        {
            return {(LogicalNumel() + group_size - 1) / group_size};
        }
        if (granularity == "per_tensor")
        {
            return {};
        }
        if (granularity == "per_axis")
        {
            return {logical_shape[static_cast<std::size_t>(axis)]};
        }
        std::vector<std::int64_t> shape = logical_shape;
        const std::size_t index = static_cast<std::size_t>(axis);
        shape[index] = (shape[index] + group_size - 1) / group_size;
        return shape;
    }

    // Referenced parameter IDs in canonical order.
    std::vector<std::int64_t> Dependencies() const
    {
        if (has_codebook)
            return {scale_value_id, codebook_value_id};
        if (!has_zero_point)
            return {scale_value_id};
        return {scale_value_id, zero_point_value_id};
    }

private:
    std::int64_t LogicalNumel() const
    {
        std::int64_t total = 1;
        for (const std::int64_t dim : logical_shape)
        {
            if (dim != 0 &&
                total > std::numeric_limits<std::int64_t>::max() / dim)
                throw QuantizationError("logical_shape product overflows int64");
            total *= dim;
        }
        return total;
    }
};

// Field-wise equality for cross-checking a boundary descriptor against the
// value registry entry it was projected from.
inline bool QuantizationSpecsEqual(const QuantizationSpec &left,
                                   const QuantizationSpec &right)
{
    return left.format == right.format &&
           left.format_version == right.format_version &&
           left.scheme == right.scheme &&
           left.storage_dtype == right.storage_dtype &&
           left.logical_shape == right.logical_shape &&
           left.scale_value_id == right.scale_value_id &&
           left.has_zero_point == right.has_zero_point &&
           left.zero_point_value_id == right.zero_point_value_id &&
           left.has_codebook == right.has_codebook &&
           left.codebook_value_id == right.codebook_value_id &&
           left.codebook == right.codebook &&
           left.granularity == right.granularity &&
           left.has_axis == right.has_axis && left.axis == right.axis &&
           left.has_group_size == right.has_group_size &&
           left.group_size == right.group_size &&
           left.compute_dtype == right.compute_dtype &&
           left.rounding == right.rounding &&
           left.saturation == right.saturation;
}

inline bool IsNf4Codebook(const IRTensor &codebook){
    if (codebook.dtype != DataType::kFloat32 ||
        codebook.shape != std::vector<std::int64_t>({16}) ||
        codebook.data.size() != 16 * sizeof(float))
        return false;
    const auto *bits = reinterpret_cast<const std::uint32_t *>(codebook.data.data());
    for (std::size_t i = 0; i < 16; ++i)
    {
        if (bits[i] != kNf4CodebookBits[i])
            return false;
    }
    return true;
}

namespace detail
{

inline bool IsIntegerValue(const json::Value &value)
{
    return value.IsNumber() && std::floor(value.number) == value.number &&
           std::fabs(value.number) <= 9007199254740992.0;
}

inline std::int64_t IntegerValue(const json::Value &value, const std::string &what,
                                 std::int64_t minimum)
{
    if (!IsIntegerValue(value) || static_cast<double>(minimum) > value.number)
        throw QuantizationError(what + " must be an integer >= " +
                                std::to_string(minimum));
    return static_cast<std::int64_t>(value.number);
}

inline const json::Value &RequiredField(const json::Value &object,
                                        const std::string &name,
                                        const std::string &context)
{
    if (!object.Has(name))
        throw QuantizationError(context + " is missing field '" + name + "'");
    return object.At(name);
}

inline std::string RequiredString(const json::Value &object, const std::string &name,
                                  const std::string &context)
{
    const json::Value &value = RequiredField(object, name, context);
    if (!value.IsString())
        throw QuantizationError(context + "." + name + " must be a string");
    return value.string;
}

inline bool IsNullField(const json::Value &object, const std::string &name,
                        const std::string &context)
{
    return RequiredField(object, name, context).IsNull();
}

inline std::vector<std::int64_t> IntegerArrayField(const json::Value &value,
                                                  const std::string &what,
                                                  std::int64_t minimum)
{
    if (!value.IsArray())
        throw QuantizationError(what + " must be an array");
    std::vector<std::int64_t> items;
    items.reserve(value.array.size());
    for (const json::Value &item : value.array)
    {
        items.push_back(IntegerValue(item, what + " element", minimum));
    }
    return items;
}

inline std::vector<std::int64_t> ShapeField(const json::Value &value,
                                            const std::string &what)
{
    return IntegerArrayField(value, what, 0);
}

// Rejects descriptors that are missing or carry fields the schema does not
// define.  Extra fields must never be silently ignored.
inline void RequireExactFields(const json::Value &object,
                               const std::vector<std::string> &expected,
                               const std::string &context)
{
    for (const std::string &name : expected)
    {
        if (!object.Has(name))
            throw QuantizationError(context + " is missing field '" + name + "'");
    }
    for (const auto &entry : object.object)
    {
        bool known = false;
        for (const std::string &name : expected)
        {
            if (entry.first == name)
            {
                known = true;
                break;
            }
        }
        if (!known)
            throw QuantizationError(context + " has unknown field '" + entry.first + "'");
    }
}

} // namespace detail

// Parse and validate one quantization descriptor.  Throws QuantizationError.
inline QuantizationSpec ParseQuantizationSpec(const json::Value &value,
                                              const std::string &context = "quantization")
{
    if (!value.IsObject())
        throw QuantizationError(context + " descriptor must be an object");

    const std::string scheme = detail::RequiredString(value, "scheme", context);
    std::vector<std::string> expected = {
        "format", "format_version", "scheme", "storage_dtype", "logical_shape",
        "scale_value_id", "zero_point_value_id", "granularity", "axis",
        "group_size", "compute_dtype", "rounding", "saturation",
    };
    const bool nf4 = scheme == "nf4/v1";
    if (nf4)
    {
        expected.push_back("codebook_value_id");
        expected.push_back("codebook");
    }
    detail::RequireExactFields(value, expected, context);

    QuantizationSpec spec;
    spec.format = detail::RequiredString(value, "format", context);
    spec.format_version = detail::RequiredString(value, "format_version", context);
    if (spec.format != kQuantizationFormat ||
        spec.format_version != kQuantizationFormatVersion)
    {
        throw QuantizationError(context +
                                " has an unsupported format or format_version");
    }

    spec.scheme = scheme;
    spec.storage_dtype = detail::RequiredString(value, "storage_dtype", context);
    const std::string expected_dtype = nf4 ? "uint4"
        : scheme == "affine-int8/v1" ? "int8"
        : scheme == "affine-int4/v1" ? "int4"
        : scheme == "affine-uint4/v1" ? "uint4"
        : "";
    if (expected_dtype.empty())
        throw QuantizationError("unsupported quantization scheme '" + scheme + "'");
    if (spec.storage_dtype != expected_dtype)
    {
        throw QuantizationError("unsupported storage_dtype '" + spec.storage_dtype +
                                "' for " + scheme);
    }

    spec.compute_dtype = detail::RequiredString(value, "compute_dtype", context);
    spec.rounding = detail::RequiredString(value, "rounding", context);
    spec.saturation = detail::RequiredString(value, "saturation", context);
    const std::string supported_rounding = nf4 ? "none" : "nearest_even";
    const std::string supported_saturation = nf4 ? "none" : "clamp";
    if (spec.compute_dtype != "float32" || spec.rounding != supported_rounding ||
        spec.saturation != supported_saturation)
    {
        throw QuantizationError(context + " has an unsupported compute_dtype, "
                                "rounding, or saturation policy");
    }

    spec.logical_shape = detail::ShapeField(
        detail::RequiredField(value, "logical_shape", context),
        context + ".logical_shape");
    spec.scale_value_id = detail::IntegerValue(
        detail::RequiredField(value, "scale_value_id", context),
        context + ".scale_value_id", 0);

    spec.has_zero_point =
        !detail::IsNullField(value, "zero_point_value_id", context);
    if (spec.has_zero_point)
    {
        spec.zero_point_value_id = detail::IntegerValue(
            value.At("zero_point_value_id"), context + ".zero_point_value_id", 0);
    }
    spec.has_axis = !detail::IsNullField(value, "axis", context);
    if (spec.has_axis)
        spec.axis = detail::IntegerValue(value.At("axis"), context + ".axis", 0);
    spec.has_group_size = !detail::IsNullField(value, "group_size", context);
    if (spec.has_group_size)
    {
        spec.group_size = detail::IntegerValue(
            value.At("group_size"), context + ".group_size", 1);
    }
    spec.granularity = detail::RequiredString(value, "granularity", context);

    if (nf4)
    {
        spec.has_codebook = true;
        spec.codebook_value_id = detail::IntegerValue(
            detail::RequiredField(value, "codebook_value_id", context),
            context + ".codebook_value_id", 0);
        spec.codebook = detail::RequiredString(value, "codebook", context);
        if (spec.codebook_value_id == spec.scale_value_id ||
            spec.codebook != kNf4CodebookIdentity)
        {
            throw QuantizationError(
                "NF4 requires a distinct codebook ID and bitsandbytes-nf4/v1 identity");
        }
        if (spec.has_zero_point || spec.has_axis ||
            spec.granularity != "per_block")
        {
            throw QuantizationError(
                "NF4 requires per_block granularity, null axis and null zero point");
        }
        if (!spec.has_group_size)
            throw QuantizationError("group_size must be an integer >= 1");
        return spec;
    }

    if (spec.has_zero_point && spec.zero_point_value_id == spec.scale_value_id)
        throw QuantizationError("scale and zero point must have distinct value IDs");
    if (spec.granularity != "per_tensor" && spec.granularity != "per_axis" &&
        spec.granularity != "per_group")
    {
        throw QuantizationError("unsupported granularity '" + spec.granularity + "'");
    }
    if (spec.granularity == "per_tensor")
    {
        if (spec.has_axis || spec.has_group_size)
            throw QuantizationError(
                "per_tensor requires axis=null and group_size=null");
        return spec;
    }
    if (!spec.has_axis)
        throw QuantizationError("axis must be an integer >= 0");
    if (static_cast<std::size_t>(spec.axis) >= spec.logical_shape.size())
        throw QuantizationError("axis must be less than the logical rank");
    if (spec.granularity == "per_axis")
    {
        if (spec.has_group_size)
            throw QuantizationError("per_axis requires group_size=null");
        return spec;
    }
    if (!spec.has_group_size)
        throw QuantizationError("group_size must be an integer >= 1");
    return spec;
}

namespace detail
{

inline std::string ShapeText(const std::vector<std::int64_t> &shape)
{
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i != 0)
            out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

inline void RequireParameterShape(const IRTensor &tensor, DataType dtype,
                                  const std::vector<std::int64_t> &shape,
                                  const std::string &label)
{
    if (tensor.dtype != dtype || tensor.shape != shape)
    {
        throw QuantizationError(label + " must have dtype " +
                                DataTypeName(dtype) + " and shape " +
                                ShapeText(shape));
    }
}

// Decode signed/unsigned 8-bit or packed 4-bit storage codes to int32.
inline std::vector<std::int32_t> DecodeCodes(const IRTensor &codes)
{
    const std::int64_t count = codes.Numel();
    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    switch (codes.dtype)
    {
    case DataType::kInt8:
    {
        for (std::int64_t i = 0; i < count; ++i)
        {
            out[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
                static_cast<std::int8_t>(codes.data[static_cast<std::size_t>(i)]));
        }
        return out;
    }
    case DataType::kUInt8:
    {
        for (std::int64_t i = 0; i < count; ++i)
        {
            out[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
                static_cast<std::uint8_t>(codes.data[static_cast<std::size_t>(i)]));
        }
        return out;
    }
    case DataType::kInt4:
    case DataType::kUInt4:
    {
        if (codes.data.size() !=
            PayloadSize(codes.dtype, static_cast<std::uint64_t>(count)))
        {
            throw QuantizationError("packed code payload size mismatch");
        }
        for (std::int64_t i = 0; i < count; ++i)
        {
            const auto byte = static_cast<std::uint8_t>(
                codes.data[static_cast<std::size_t>(i / 2)]);
            std::int32_t code = (byte >> (4 * (i % 2))) & 0x0F;
            if (codes.dtype == DataType::kInt4 && code >= 8)
                code -= 16;
            out[static_cast<std::size_t>(i)] = code;
        }
        return out;
    }
    default:
        throw QuantizationError(std::string("codes must use int8, int4, or uint4 "
                                            "storage, not ") +
                                DataTypeName(codes.dtype));
    }
}

inline const IRTensor &Parameter(const std::map<std::int64_t, IRTensor> &values,
                                 std::int64_t value_id, const std::string &label)
{
    const auto found = values.find(value_id);
    if (found == values.end())
        throw QuantizationError("missing " + label + " value ID " +
                                std::to_string(value_id));
    return found->second;
}

inline void RequireFinitePositiveScale(const std::vector<float> &scale, bool allow_zero)
{
    for (const float value : scale)
    {
        if (!std::isfinite(value) || value < 0.0f || (!allow_zero && value == 0.0f))
        {
            throw QuantizationError(
                allow_zero
                    ? "NF4 block scales must be finite and nonnegative"
                    : "scale values must be finite and strictly positive");
        }
    }
}

// Row-major coordinate for one logical flat index.
inline void CoordinatesForFlatIndex(std::int64_t flat,
                                    const std::vector<std::int64_t> &shape,
                                    std::vector<std::int64_t> &coordinates)
{
    coordinates.assign(shape.size(), 0);
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        if (shape[i] == 0)
        {
            coordinates[i] = 0;
        }
        else
        {
            coordinates[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
}

inline std::int64_t FlatIndex(const std::vector<std::int64_t> &coordinates,
                              const std::vector<std::int64_t> &shape)
{
    std::int64_t flat = 0;
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        flat = flat * shape[i] + coordinates[i];
    }
    return flat;
}

inline void RequireFiniteResult(float value)
{
    if (!std::isfinite(value))
        throw QuantizationError("dequantization overflow in float32");
}

} // namespace detail

// Reference dequantization to float32, bit-identical to the Python reference.
// `parameters` maps each referenced value ID to its loaded parameter tensor.
inline std::vector<float> DequantizeQuantizationFloat32(
    const QuantizationSpec &spec, const IRTensor &codes,
    const std::map<std::int64_t, IRTensor> &parameters)
{
    const DataType expected_codes_dtype = spec.storage_dtype == "int8"
        ? DataType::kInt8
        : spec.storage_dtype == "int4" ? DataType::kInt4 : DataType::kUInt4;
    if (codes.dtype != expected_codes_dtype ||
        codes.shape != spec.logical_shape)
    {
        throw QuantizationError("codes must have dtype " + spec.storage_dtype +
                                " and shape " + codes.ShapeString() +
                                " matching the descriptor");
    }
    const std::vector<std::int32_t> raw = detail::DecodeCodes(codes);
    const std::int64_t count = static_cast<std::int64_t>(raw.size());
    const auto [low, high] = spec.CodeRange();

    if (spec.IsNf4())
    {
        const IRTensor &scale_tensor = detail::Parameter(
            parameters, spec.scale_value_id, "block scale");
        detail::RequireParameterShape(scale_tensor, DataType::kFloat32,
                                      spec.ParameterShape(), "block scale");
        const std::vector<float> scale = scale_tensor.AsFloat32();
        detail::RequireFinitePositiveScale(scale, true);
        const IRTensor &codebook = detail::Parameter(
            parameters, spec.codebook_value_id, "codebook");
        if (!IsNf4Codebook(codebook))
            throw QuantizationError(
                "codebook values do not match bitsandbytes-nf4/v1");

        const std::vector<float> book = codebook.AsFloat32();
        std::vector<float> out(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i)
        {
            const std::int32_t index = raw[static_cast<std::size_t>(i)];
            if (index < 0 || index > 15)
                throw QuantizationError("NF4 codebook index outside [0, 15]");
            const float value = book[static_cast<std::size_t>(index)] *
                                scale[static_cast<std::size_t>(i / spec.group_size)];
            detail::RequireFiniteResult(value);
            out[static_cast<std::size_t>(i)] = value;
        }
        return out;
    }

    const IRTensor &scale_tensor =
        detail::Parameter(parameters, spec.scale_value_id, "scale");
    detail::RequireParameterShape(scale_tensor, DataType::kFloat32,
                                  spec.ParameterShape(), "scale");
    const std::vector<float> scale = scale_tensor.AsFloat32();
    detail::RequireFinitePositiveScale(scale, false);

    std::vector<float> zero(scale.size(), 0.0f);
    if (spec.has_zero_point)
    {
        const IRTensor &zero_tensor = detail::Parameter(
            parameters, spec.zero_point_value_id, "zero point");
        const DataType zero_dtype =
            spec.storage_dtype == "uint4" ? DataType::kUInt8 : DataType::kInt8;
        detail::RequireParameterShape(zero_tensor, zero_dtype,
                                      spec.ParameterShape(), "zero point");
        zero = zero_tensor.AsFloat32();
        for (const float value : zero)
        {
            if (value < static_cast<float>(low) ||
                value > static_cast<float>(high))
                throw QuantizationError("zero point outside storage code range");
        }
    }

    const std::vector<std::int64_t> scale_shape = spec.ParameterShape();
    std::vector<std::int64_t> coordinates;
    std::vector<float> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i)
    {
        const std::int32_t code = raw[static_cast<std::size_t>(i)];
        if (code < low || code > high)
            throw QuantizationError("codes outside storage range");
        std::size_t parameter_index = 0;
        if (spec.granularity == "per_axis")
        {
            detail::CoordinatesForFlatIndex(i, spec.logical_shape, coordinates);
            parameter_index = static_cast<std::size_t>(
                coordinates[static_cast<std::size_t>(spec.axis)]);
        }
        else if (spec.granularity == "per_group")
        {
            detail::CoordinatesForFlatIndex(i, spec.logical_shape, coordinates);
            const std::size_t axis = static_cast<std::size_t>(spec.axis);
            coordinates[axis] /= spec.group_size;
            parameter_index = static_cast<std::size_t>(
                detail::FlatIndex(coordinates, scale_shape));
        }
        const float value = (static_cast<float>(code) -
                             zero[parameter_index]) * scale[parameter_index];
        detail::RequireFiniteResult(value);
        out[static_cast<std::size_t>(i)] = value;
    }
    return out;
}

} // namespace inferref

#endif // INFERREF_QUANTIZATION_HPP
