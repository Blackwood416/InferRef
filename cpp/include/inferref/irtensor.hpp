// InferRef .irtensor v0.1 reader — header-only, dependency-free.
//
// Proves the core interoperability claim: an inference engine can consume
// InferRef reference tensors without PyTorch, without Python, and without a
// JSON library.
//
// Header layout (little-endian, matching inferref/tensor/codec.py and
// Trace IR v0.1 §21):
//
//     offset  size    field
//     0       4       magic "IRTN"
//     4       2       tensor_format_version (u16)
//     6       2       header_size           (u16)
//     8       2       dtype                 (u16)
//     10      4       flags                 (u32)
//     14      2       rank                  (u16)
//     16      2       reserved              (u16)
//     18      8       logical_numel         (u64)
//     26      8       payload_nbytes        (u64)
//     34      8*rank  shape[]               (i64)
//     34+8r   8*rank  stride[]              (i64)
//     34+16r  8       storage_offset        (i64)
//             pad     -> header_size = 48 + 16*rank
//     header_size     payload
//
// `flags` (offset 10) and `logical_numel` (offset 18) are deliberately not
// naturally aligned — the field order follows the spec. Every field is
// therefore read with an explicit byte copy rather than a struct cast, which
// is both portable and free of undefined behaviour.
//
// Usage:
//
//     auto t = inferref::ReadIRTensor("q_embed.irtensor");
//     std::span<const float> values = t.AsFloat32();

#ifndef INFERREF_IRTENSOR_HPP
#define INFERREF_IRTENSOR_HPP

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferref
{

// Stable dtype codes, frozen at format version 0.1.
// Mirrors inferref/ir/dtypes.py — never renumber, only append.
enum class DataType : std::uint16_t
{
    kInvalid = 0,
    kBool = 1,
    kInt8 = 2,
    kInt16 = 3,
    kInt32 = 4,
    kInt64 = 5,
    kUInt8 = 6,
    kUInt16 = 7,
    kUInt32 = 8,
    kUInt64 = 9,
    kFloat16 = 10,
    kBFloat16 = 11,
    kFloat32 = 12,
    kFloat64 = 13,
    kComplex64 = 14,
    kComplex128 = 15,
};

inline const char *DataTypeName(DataType dtype)
{
    switch (dtype)
    {
    case DataType::kBool: return "bool";
    case DataType::kInt8: return "int8";
    case DataType::kInt16: return "int16";
    case DataType::kInt32: return "int32";
    case DataType::kInt64: return "int64";
    case DataType::kUInt8: return "uint8";
    case DataType::kUInt16: return "uint16";
    case DataType::kUInt32: return "uint32";
    case DataType::kUInt64: return "uint64";
    case DataType::kFloat16: return "float16";
    case DataType::kBFloat16: return "bfloat16";
    case DataType::kFloat32: return "float32";
    case DataType::kFloat64: return "float64";
    case DataType::kComplex64: return "complex64";
    case DataType::kComplex128: return "complex128";
    default: return "invalid";
    }
}

inline std::size_t DataTypeSize(DataType dtype)
{
    switch (dtype)
    {
    case DataType::kBool:
    case DataType::kInt8:
    case DataType::kUInt8: return 1;
    case DataType::kInt16:
    case DataType::kUInt16:
    case DataType::kFloat16:
    case DataType::kBFloat16: return 2;
    case DataType::kInt32:
    case DataType::kUInt32:
    case DataType::kFloat32: return 4;
    case DataType::kInt64:
    case DataType::kUInt64:
    case DataType::kFloat64:
    case DataType::kComplex64: return 8;
    case DataType::kComplex128: return 16;
    default: return 0;
    }
}

// flags bit 0: payload is in canonical logical contiguous order (IR §20).
inline constexpr std::uint32_t kFlagCanonicalContiguous = 1u << 0;

inline constexpr std::uint16_t kTensorFormatVersion = 1;

// IEEE-754 round-to-nearest-even encoders shared by native engines and tests.
inline std::uint16_t EncodeFloat16(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
    {
        if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        std::uint32_t payload = (mantissa >> 13) | 0x0200u;
        return static_cast<std::uint16_t>(sign | 0x7c00u | (payload & 0x03ffu));
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0)
    {
        if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t significand = mantissa | 0x800000u;
        const int shift = 14 - half_exponent;
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder = significand & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_mantissa & 1u)))
    {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x0400u)
        {
            rounded_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exponent) << 10) | rounded_mantissa);
}

inline std::uint16_t EncodeBFloat16(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
        return 0x7fc0u;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

class IRTensorError : public std::runtime_error
{
public:
    explicit IRTensorError(const std::string &what) : std::runtime_error(what) {}
};

// One decoded .irtensor. Owns its payload bytes.
class IRTensor
{
public:
    DataType dtype = DataType::kInvalid;
    std::vector<std::int64_t> shape;
    // The *reference* tensor's stride. It describes the original layout for
    // debugging (SPEC §29); it does NOT describe `data`, which is canonical
    // contiguous.
    std::vector<std::int64_t> stride;
    std::int64_t storage_offset = 0;
    std::uint32_t flags = kFlagCanonicalContiguous;
    std::vector<std::byte> data;

    std::size_t Rank() const { return shape.size(); }

    std::int64_t Numel() const
    {
        std::int64_t total = 1;
        for (std::int64_t dim : shape)
        {
            total *= dim;
        }
        return total;
    }

    bool IsCanonicalContiguous() const
    {
        return (flags & kFlagCanonicalContiguous) != 0;
    }

    // Typed view over the payload. Throws if the dtype does not match.
    template <typename T>
    const T *TypedData() const
    {
        if (sizeof(T) != DataTypeSize(dtype))
        {
            throw IRTensorError(
                std::string("element size mismatch: tensor is ") + DataTypeName(dtype));
        }
        return reinterpret_cast<const T *>(data.data());
    }

    // Widen to float, decoding float16/bfloat16 in software so an engine can
    // compare reduced-precision tensors without a half-float library.
    std::vector<float> AsFloat32() const
    {
        const std::int64_t count = Numel();
        std::vector<float> out(static_cast<std::size_t>(count));
        switch (dtype)
        {
        case DataType::kFloat32:
        {
            std::memcpy(out.data(), data.data(), data.size());
            break;
        }
        case DataType::kFloat64:
        {
            const auto *src = reinterpret_cast<const double *>(data.data());
            for (std::int64_t i = 0; i < count; ++i)
            {
                out[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
            }
            break;
        }
        case DataType::kBFloat16:
        {
            // bfloat16 is the high 16 bits of a float32.
            const auto *src = reinterpret_cast<const std::uint16_t *>(data.data());
            for (std::int64_t i = 0; i < count; ++i)
            {
                const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                out[static_cast<std::size_t>(i)] = value;
            }
            break;
        }
        case DataType::kFloat16:
        {
            const auto *src = reinterpret_cast<const std::uint16_t *>(data.data());
            for (std::int64_t i = 0; i < count; ++i)
            {
                out[static_cast<std::size_t>(i)] = DecodeHalf(src[i]);
            }
            break;
        }
        case DataType::kInt8:
            ConvertIntegral<std::int8_t>(out, count);
            break;
        case DataType::kInt16:
            ConvertIntegral<std::int16_t>(out, count);
            break;
        case DataType::kInt32:
            ConvertIntegral<std::int32_t>(out, count);
            break;
        case DataType::kInt64:
            ConvertIntegral<std::int64_t>(out, count);
            break;
        case DataType::kUInt8:
        case DataType::kBool:
            ConvertIntegral<std::uint8_t>(out, count);
            break;
        case DataType::kUInt16:
            ConvertIntegral<std::uint16_t>(out, count);
            break;
        case DataType::kUInt32:
            ConvertIntegral<std::uint32_t>(out, count);
            break;
        case DataType::kUInt64:
            ConvertIntegral<std::uint64_t>(out, count);
            break;
        default:
            throw IRTensorError(
                std::string("cannot widen dtype ") + DataTypeName(dtype) + " to float32");
        }
        return out;
    }

    std::string ShapeString() const
    {
        std::string out = "[";
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i != 0)
            {
                out += ", ";
            }
            out += std::to_string(shape[i]);
        }
        return out + "]";
    }

    std::string StrideString() const
    {
        std::string out = "[";
        for (std::size_t i = 0; i < stride.size(); ++i)
        {
            if (i != 0)
            {
                out += ", ";
            }
            out += std::to_string(stride[i]);
        }
        return out + "]";
    }

private:
    template <typename T>
    void ConvertIntegral(std::vector<float> &out, std::int64_t count) const
    {
        const auto *src = reinterpret_cast<const T *>(data.data());
        for (std::int64_t i = 0; i < count; ++i)
        {
            out[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
        }
    }

    static float DecodeHalf(std::uint16_t half)
    {
        const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
        const std::uint32_t exponent = (half >> 10) & 0x1Fu;
        const std::uint32_t mantissa = half & 0x3FFu;
        std::uint32_t bits = 0;

        if (exponent == 0)
        {
            if (mantissa != 0)
            {
                // Subnormal: renormalise into a float32 normal.
                std::uint32_t shift = 0;
                std::uint32_t value = mantissa;
                while ((value & 0x400u) == 0)
                {
                    value <<= 1;
                    ++shift;
                }
                value &= 0x3FFu;
                bits = sign | ((127 - 15 - shift) << 23) | (value << 13);
            }
            else
            {
                bits = sign;
            }
        }
        else if (exponent == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
};

namespace detail
{

// Read a little-endian integer at `offset`. Explicit byte assembly keeps this
// correct on big-endian hosts and free of alignment assumptions.
template <typename T>
inline T ReadLE(const std::byte *buffer, std::size_t offset)
{
    typename std::make_unsigned<T>::type value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
    {
        value |= static_cast<typename std::make_unsigned<T>::type>(
                     static_cast<std::uint8_t>(buffer[offset + i]))
                 << (8 * i);
    }
    T out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

inline constexpr std::size_t kFixedHeaderSize = 34;

inline std::size_t HeaderSizeForRank(std::size_t rank)
{
    const std::size_t fixed = kFixedHeaderSize + 16 * rank + 8;
    return (fixed + 7) / 8 * 8;
}

} // namespace detail

// Decode an .irtensor from a memory buffer.
inline IRTensor DecodeIRTensor(const std::byte *buffer, std::size_t size)
{
    if (size < detail::kFixedHeaderSize)
    {
        throw IRTensorError("file too short to be an .irtensor");
    }
    if (std::memcmp(buffer, "IRTN", 4) != 0)
    {
        throw IRTensorError("bad magic: not an .irtensor file");
    }

    const auto version = detail::ReadLE<std::uint16_t>(buffer, 4);
    if (version != kTensorFormatVersion)
    {
        throw IRTensorError("unsupported .irtensor version " + std::to_string(version));
    }

    const auto header_size = detail::ReadLE<std::uint16_t>(buffer, 6);
    const auto dtype_code = detail::ReadLE<std::uint16_t>(buffer, 8);
    const auto flags = detail::ReadLE<std::uint32_t>(buffer, 10);
    const auto rank = detail::ReadLE<std::uint16_t>(buffer, 14);
    const auto numel = detail::ReadLE<std::uint64_t>(buffer, 18);
    const auto payload_nbytes = detail::ReadLE<std::uint64_t>(buffer, 26);

    if (header_size != detail::HeaderSizeForRank(rank))
    {
        throw IRTensorError("header_size does not match rank");
    }
    if (size < header_size + payload_nbytes)
    {
        throw IRTensorError("truncated .irtensor payload");
    }

    IRTensor tensor;
    tensor.dtype = static_cast<DataType>(dtype_code);
    tensor.flags = flags;
    if (DataTypeSize(tensor.dtype) == 0)
    {
        throw IRTensorError("unknown dtype code " + std::to_string(dtype_code));
    }
    if (numel * DataTypeSize(tensor.dtype) != payload_nbytes)
    {
        throw IRTensorError("payload_nbytes inconsistent with numel and dtype");
    }

    tensor.shape.resize(rank);
    tensor.stride.resize(rank);
    std::size_t offset = detail::kFixedHeaderSize;
    for (std::uint16_t i = 0; i < rank; ++i, offset += 8)
    {
        tensor.shape[i] = detail::ReadLE<std::int64_t>(buffer, offset);
    }
    for (std::uint16_t i = 0; i < rank; ++i, offset += 8)
    {
        tensor.stride[i] = detail::ReadLE<std::int64_t>(buffer, offset);
    }
    tensor.storage_offset = detail::ReadLE<std::int64_t>(buffer, offset);

    tensor.data.resize(static_cast<std::size_t>(payload_nbytes));
    if (payload_nbytes != 0)
    {
        std::memcpy(tensor.data.data(), buffer + header_size,
                    static_cast<std::size_t>(payload_nbytes));
    }
    return tensor;
}

// Read an .irtensor from disk.
inline IRTensor ReadIRTensor(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        throw IRTensorError("cannot open " + path);
    }
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(static_cast<std::size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(buffer.data()), size))
    {
        throw IRTensorError("cannot read " + path);
    }
    return DecodeIRTensor(buffer.data(), buffer.size());
}

// Write an .irtensor, so an engine can emit results in the InferRef output
// protocol (SPEC §22) without linking anything.
inline void WriteIRTensor(const std::string &path, const IRTensor &tensor)
{
    const std::size_t rank = tensor.shape.size();
    if (tensor.stride.size() != rank)
    {
        throw IRTensorError("shape and stride rank differ");
    }
    const std::size_t header_size = detail::HeaderSizeForRank(rank);
    std::vector<std::byte> header(header_size, std::byte{0});

    auto write_le = [&header](std::size_t offset, std::uint64_t value, std::size_t width) {
        for (std::size_t i = 0; i < width; ++i)
        {
            header[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xFFu);
        }
    };

    std::memcpy(header.data(), "IRTN", 4);
    write_le(4, kTensorFormatVersion, 2);
    write_le(6, header_size, 2);
    write_le(8, static_cast<std::uint16_t>(tensor.dtype), 2);
    write_le(10, tensor.flags, 4);
    write_le(14, rank, 2);
    write_le(16, 0, 2);

    std::uint64_t numel = 1;
    for (std::int64_t dim : tensor.shape)
    {
        numel *= static_cast<std::uint64_t>(dim);
    }
    write_le(18, numel, 8);
    write_le(26, tensor.data.size(), 8);

    std::size_t offset = detail::kFixedHeaderSize;
    for (std::size_t i = 0; i < rank; ++i, offset += 8)
    {
        write_le(offset, static_cast<std::uint64_t>(tensor.shape[i]), 8);
    }
    for (std::size_t i = 0; i < rank; ++i, offset += 8)
    {
        write_le(offset, static_cast<std::uint64_t>(tensor.stride[i]), 8);
    }
    write_le(offset, static_cast<std::uint64_t>(tensor.storage_offset), 8);

    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw IRTensorError("cannot open " + path + " for writing");
    }
    stream.write(reinterpret_cast<const char *>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char *>(tensor.data.data()),
                 static_cast<std::streamsize>(tensor.data.size()));
    if (!stream)
    {
        throw IRTensorError("cannot write " + path);
    }
}

} // namespace inferref

#endif // INFERREF_IRTENSOR_HPP
