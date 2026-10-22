// Self-test for the C++ testcase 0.4 quantization ABI (InferRef 0.12 M1).
//
// Covers:
//   - QuantizationSpec parsing and null-aware axis/group_size/zero_point.
//   - Loading 0.1-0.4 testcases, the top-level quantization_parameters
//     collection (value_id key), and every fail-closed validation rule.
//   - DequantizeInputToFloat32 for affine-int8/int4/uint4 (per_tensor,
//     per_axis, per_group) and nf4/v1 (per_block), compared as float32 bits.
//   - Every fail-closed rule with a negative manifest.
//
// With one argument it also loads a Python-generated golden directory
// (goldens.json) and compares the C++ dequantizer with the Python reference
// bit-for-bit.  Run it with no arguments for the self-contained checks.

#include <inferref/testcase.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{

int g_failures = 0;

void Check(bool condition, const std::string &what)
{
    if (condition)
    {
        std::printf("  ok   %s\n", what.c_str());
    }
    else
    {
        std::printf("  FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

std::string ShapeJson(const std::vector<std::int64_t> &shape)
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

std::string StrideJson(const std::vector<std::int64_t> &shape)
{
    std::vector<std::int64_t> stride(shape.size(), 1);
    std::int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        stride[i] = running;
        running *= shape[i] == 0 ? 1 : shape[i];
    }
    return ShapeJson(stride);
}

inferref::IRTensor MakeFloat32(const std::vector<std::int64_t> &shape,
                               const std::vector<float> &values)
{
    inferref::IRTensor tensor;
    tensor.dtype = inferref::DataType::kFloat32;
    tensor.shape = shape;
    tensor.stride.assign(shape.size(), 1);
    std::int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        tensor.stride[i] = running;
        running *= shape[i] == 0 ? 1 : shape[i];
    }
    tensor.data.resize(values.size() * sizeof(float));
    if (!values.empty())
        std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
    return tensor;
}

inferref::IRTensor MakeInt8(const std::vector<std::int64_t> &shape,
                            const std::vector<int> &values)
{
    inferref::IRTensor tensor;
    tensor.dtype = inferref::DataType::kInt8;
    tensor.shape = shape;
    tensor.stride.assign(shape.size(), 1);
    std::int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        tensor.stride[i] = running;
        running *= shape[i] == 0 ? 1 : shape[i];
    }
    tensor.data.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        tensor.data[i] = static_cast<std::byte>(static_cast<std::uint8_t>(values[i]));
    return tensor;
}

inferref::IRTensor MakeUInt8(const std::vector<std::int64_t> &shape,
                             const std::vector<int> &values)
{
    inferref::IRTensor tensor = MakeInt8(shape, values);
    tensor.dtype = inferref::DataType::kUInt8;
    return tensor;
}

// Pack signed/unsigned 4-bit codes low nibble first, like the .irtensor v2
// wire format (odd tails leave the high nibble zero).
inferref::IRTensor MakePacked4(inferref::DataType dtype,
                               const std::vector<std::int64_t> &shape,
                               const std::vector<int> &codes)
{
    inferref::IRTensor tensor;
    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.stride.assign(shape.size(), 1);
    std::int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        tensor.stride[i] = running;
        running *= shape[i] == 0 ? 1 : shape[i];
    }
    tensor.data.assign((codes.size() + 1) / 2, std::byte{0});
    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        const int nibble = codes[i] & 0x0F;
        if (i % 2 == 0)
            tensor.data[i / 2] = static_cast<std::byte>(nibble);
        else
            tensor.data[i / 2] = static_cast<std::byte>(
                static_cast<std::uint8_t>(tensor.data[i / 2]) |
                static_cast<std::uint8_t>(nibble << 4));
    }
    return tensor;
}

inferref::IRTensor MakeNf4Codebook()
{
    const float table[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f,
        -0.39491748809814453f, -0.28444138169288635f,
        -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
        0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
        0.7229568362236023f, 1.0f,
    };
    return MakeFloat32({16}, std::vector<float>(table, table + 16));
}

void WriteFile(const std::string &path, const std::string &text)
{
    std::filesystem::path target(path);
    if (target.has_parent_path())
        std::filesystem::create_directories(target.parent_path());
    std::ofstream stream(path, std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadFile(const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void WriteTensor(const std::string &path, const inferref::IRTensor &tensor)
{
    std::filesystem::path target(path);
    if (target.has_parent_path())
        std::filesystem::create_directories(target.parent_path());
    inferref::WriteIRTensor(path, tensor);
}

std::string Metadata(const std::string &dtype,
                     const std::vector<std::int64_t> &shape, std::int64_t value_id)
{
    return "\"dtype\": " + std::string("\"") + dtype + "\", \"shape\": " +
           ShapeJson(shape) + ", \"stride\": " + StrideJson(shape) +
           ", \"storage_offset\": 0, \"contiguous\": true, "
           "\"runtime_object_id\": null, \"storage_id\": " +
           std::to_string(value_id) + ", \"storage_version\": 0";
}

struct DescriptorFields
{
    std::string format_version = "1.0";
    std::string scheme = "affine-int8/v1";
    std::string storage_dtype = "int8";
    std::vector<std::int64_t> logical_shape = {2, 3};
    std::int64_t scale_value_id = 10;
    std::string zero_point_value_id = "11";
    std::string granularity = "per_tensor";
    std::string axis = "null";
    std::string group_size = "null";
    std::int64_t codebook_value_id = 12;
    std::string codebook = "bitsandbytes-nf4/v1";
    std::string extra_fields;
};

std::string Descriptor(const DescriptorFields &fields)
{
    const bool nf4 = fields.scheme == "nf4/v1";
    std::string text =
        "{\"format\": \"inferref-quantization\", \"format_version\": \"" +
        fields.format_version + "\", \"scheme\": \"" + fields.scheme +
        "\", \"storage_dtype\": \"" + fields.storage_dtype +
        "\", \"logical_shape\": " + ShapeJson(fields.logical_shape) +
        ", \"scale_value_id\": " + std::to_string(fields.scale_value_id) +
        ", \"zero_point_value_id\": " + fields.zero_point_value_id +
        ", \"granularity\": \"" + fields.granularity + "\", \"axis\": " +
        fields.axis + ", \"group_size\": " + fields.group_size +
        ", \"compute_dtype\": \"float32\", \"rounding\": \"" +
        (nf4 ? "none" : "nearest_even") + "\", \"saturation\": \"" +
        (nf4 ? "none" : "clamp") + "\"";
    if (nf4)
    {
        text += ", \"codebook_value_id\": " +
                std::to_string(fields.codebook_value_id) + ", \"codebook\": \"" +
                fields.codebook + "\"";
    }
    if (!fields.extra_fields.empty())
        text += ", " + fields.extra_fields;
    return text + "}";
}

std::string ValueRecord(std::int64_t id, const std::string &dtype,
                        const std::vector<std::int64_t> &shape,
                        const std::string &descriptor = std::string())
{
    std::string text = "{\"id\": " + std::to_string(id) + ", " +
                       Metadata(dtype, shape, id) +
                       ", \"role\": \"parameter\", \"producer\": null";
    if (!descriptor.empty())
        text += ", \"quantization\": " + descriptor;
    return text + "}";
}

std::string InputRole(const std::string &name, std::int64_t value_id,
                      const std::string &payload, const std::string &dtype,
                      const std::vector<std::int64_t> &shape,
                      const std::string &descriptor)
{
    return "{\"name\": \"" + name + "\", \"value_id\": " +
           std::to_string(value_id) + ", \"payload\": \"" + payload + "\", " +
           Metadata(dtype, shape, value_id) + ", \"quantization\": " +
           descriptor + "}";
}

std::string PlainInputRole(const std::string &name, const std::string &payload)
{
    return "{\"name\": \"" + name + "\", \"payload\": \"" + payload + "\"}";
}

std::string ParameterEntry(std::int64_t id, const std::string &payload,
                           const std::string &dtype,
                           const std::vector<std::int64_t> &shape,
                           const std::string &nbytes = std::string(),
                           const std::string &key = "value_id")
{
    std::string text = "{\"" + key + "\": " + std::to_string(id) +
                       ", \"name\": \"v" + std::to_string(id) +
                       "\", \"payload\": \"" + payload + "\", " +
                       Metadata(dtype, shape, id);
    if (!nbytes.empty())
        text += ", \"nbytes\": " + nbytes;
    return text + "}";
}

std::string Manifest(const std::string &input_roles, const std::string &values,
                     const std::string &parameters,
                     const std::string &version = "0.4")
{
    std::string text =
        "{\"format\": \"inferref-testcase\", \"format_version\": \"" + version +
        "\", \"origin\": {\"region\": \"quantization/selftest/v1\"}, "
        "\"inputs\": [" + input_roles + "], "
        "\"outputs\": [{\"name\": \"y\"}], \"nodes\": [], "
        "\"values\": [" + values + "]";
    if (!parameters.empty())
        text += ", \"quantization_parameters\": [" + parameters + "]";
    return text + "}";
}

std::string BeginCase(const std::string &base, const std::string &name)
{
    const std::string dir = base + "/" + name;
    std::filesystem::create_directories(dir);
    return dir;
}

void WriteManifest(const std::string &dir, const std::string &manifest)
{
    WriteFile(dir + "/testcase.json", manifest);
}

std::uint32_t FloatBits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool BitsEqual(const std::vector<float> &actual,
               const std::vector<std::uint32_t> &expected)
{
    if (actual.size() != expected.size())
        return false;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (FloatBits(actual[i]) != expected[i])
        {
            std::printf("       first mismatch at %zu: got 0x%08x, want 0x%08x\n",
                        i, FloatBits(actual[i]), expected[i]);
            return false;
        }
    }
    return true;
}

void ExpectLoadRejected(const std::string &dir, const std::string &what,
                        const std::string &expected)
{
    bool rejected = false;
    std::string message;
    try
    {
        (void)inferref::Testcase::Load(dir);
    }
    catch (const inferref::TestcaseError &error)
    {
        message = error.what();
        rejected = message.find(expected) != std::string::npos;
    }
    Check(rejected, what);
    if (!rejected)
        std::printf("       expected '%s', got: %s\n", expected.c_str(),
                    message.empty() ? "(no error)" : message.c_str());
}

void ExpectDequantizeRejected(const inferref::Testcase &testcase,
                              const std::string &input, const std::string &what,
                              const std::string &expected)
{
    bool rejected = false;
    std::string message;
    try
    {
        (void)testcase.DequantizeInputToFloat32(input);
    }
    catch (const inferref::TestcaseError &error)
    {
        message = error.what();
        rejected = message.find(expected) != std::string::npos;
    }
    Check(rejected, what);
    if (!rejected)
        std::printf("       expected '%s', got: %s\n", expected.c_str(),
                    message.empty() ? "(no error)" : message.c_str());
}

// ---------------------------------------------------------------------------
// Self-contained positive cases: every scheme, granularity, and odd tail.
// ---------------------------------------------------------------------------

void TestAffineInt8PerTensor(const std::string &base)
{
    std::printf("affine-int8/v1 per_tensor with an asymmetric zero point\n");
    const std::string dir = BeginCase(base, "positive-int8-per-tensor");
    WriteTensor(dir + "/inputs/weight.irtensor",
                MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
    WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                MakeFloat32({}, {0.5f}));
    WriteTensor(dir + "/quantization_parameters/v11.irtensor",
                MakeInt8({}, {3}));

    DescriptorFields fields;
    fields.logical_shape = {2, 3};
    const std::string descriptor = Descriptor(fields);
    WriteManifest(dir, Manifest(
        InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                  descriptor),
        ValueRecord(100, "int8", {2, 3}, descriptor) + ", " +
            ValueRecord(10, "float32", {}) + ", " + ValueRecord(11, "int8", {}),
        ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
            ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                  "int8", {})));

    const inferref::Testcase testcase = inferref::Testcase::Load(dir);
    Check(testcase.FormatVersion() == "0.4", "FormatVersion() is 0.4");
    Check(testcase.HasInputQuantization("weight"), "input weight is quantized");
    Check(testcase.InputQuantization("weight").scheme == "affine-int8/v1",
          "descriptor scheme");
    Check(testcase.QuantizationParameterIds() ==
              std::vector<std::int64_t>({10, 11}),
          "parameter IDs are [10, 11]");
    Check(testcase.QuantizationParameter(10).AsFloat32()[0] == 0.5f,
          "QuantizationParameter(10) returns the scale");
    Check(BitsEqual(testcase.DequantizeInputToFloat32("weight"),
                    {0xc2830000u, 0xc0400000u, 0xbfc00000u, 0xbf800000u,
                     0x3f800000u, 0x42780000u}),
          "float32 bits match (code - 3) * 0.5");
}

void TestNf4PerBlock(const std::string &base)
{
    std::printf("nf4/v1 per_block with a zero scale and negative zero results\n");
    const std::string dir = BeginCase(base, "positive-nf4-per-block");
    WriteTensor(dir + "/inputs/weight.irtensor",
                MakePacked4(inferref::DataType::kUInt4, {2, 3},
                            {0, 1, 2, 3, 4, 5}));
    WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                MakeFloat32({2}, {0.5f, 0.0f}));
    WriteTensor(dir + "/quantization_parameters/v12.irtensor", MakeNf4Codebook());

    DescriptorFields fields;
    fields.scheme = "nf4/v1";
    fields.storage_dtype = "uint4";
    fields.logical_shape = {2, 3};
    fields.zero_point_value_id = "null";
    fields.granularity = "per_block";
    fields.group_size = "4";
    const std::string descriptor = Descriptor(fields);
    WriteManifest(dir, Manifest(
        InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                  descriptor),
        ValueRecord(100, "uint4", {2, 3}, descriptor) + ", " +
            ValueRecord(10, "float32", {2}) + ", " +
            ValueRecord(12, "float32", {16}),
        ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {2}) +
            ", " + ParameterEntry(12, "quantization_parameters/v12.irtensor",
                                  "float32", {16})));

    const inferref::Testcase testcase = inferref::Testcase::Load(dir);
    Check(testcase.InputQuantization("weight").IsNf4(), "descriptor is NF4");
    Check(!testcase.InputQuantization("weight").has_axis,
          "NF4 axis is explicitly null");
    Check(!testcase.InputQuantization("weight").has_zero_point,
          "NF4 zero point is explicitly null");
    Check(BitsEqual(testcase.DequantizeInputToFloat32("weight"),
                    {0xbf000000u, 0xbeb239b1u, 0xbe866b30u, 0xbe4a32a0u,
                     0x80000000u, 0x80000000u}),
          "float32 bits include negative zero from a zero block scale");
}

void TestAffineInt4PerAxis(const std::string &base)
{
    std::printf("affine-int4/v1 per_axis with an odd packed tail\n");
    const std::string dir = BeginCase(base, "positive-int4-per-axis");
    WriteTensor(dir + "/inputs/weight.irtensor",
                MakePacked4(inferref::DataType::kInt4, {3, 3},
                            {-8, -7, -6, -5, -4, -3, -2, -1, 0}));
    WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                MakeFloat32({3}, {0.5f, 0.25f, 0.125f}));
    WriteTensor(dir + "/quantization_parameters/v11.irtensor",
                MakeInt8({3}, {-1, 3, 7}));

    DescriptorFields fields;
    fields.scheme = "affine-int4/v1";
    fields.storage_dtype = "int4";
    fields.logical_shape = {3, 3};
    fields.granularity = "per_axis";
    fields.axis = "1";
    fields.group_size = "null";
    const std::string descriptor = Descriptor(fields);
    WriteManifest(dir, Manifest(
        InputRole("weight", 100, "inputs/weight.irtensor", "int4", {3, 3},
                  descriptor),
        ValueRecord(100, "int4", {3, 3}, descriptor) + ", " +
            ValueRecord(10, "float32", {3}) + ", " + ValueRecord(11, "int8", {3}),
        ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {3}) +
            ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                  "int8", {3})));

    const inferref::Testcase testcase = inferref::Testcase::Load(dir);
    Check(testcase.InputQuantization("weight").has_axis &&
              testcase.InputQuantization("weight").axis == 1,
          "descriptor axis is 1");
    Check(BitsEqual(testcase.DequantizeInputToFloat32("weight"),
                    {0xc0600000u, 0xc0200000u, 0xbfd00000u, 0xc0000000u,
                     0xbfe00000u, 0xbfa00000u, 0xbf000000u, 0xbf800000u,
                     0xbf600000u}),
          "float32 bits for the 9-element int4 tail");
}

void TestAffineUInt4PerGroup(const std::string &base)
{
    std::printf("affine-uint4/v1 per_group with a short final group\n");
    const std::string dir = BeginCase(base, "positive-uint4-per-group");
    WriteTensor(dir + "/inputs/weight.irtensor",
                MakePacked4(inferref::DataType::kUInt4, {2, 3},
                            {2, 7, 12, 1, 6, 11}));
    WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                MakeFloat32({2, 2}, {0.5f, 0.25f, 0.125f, 0.75f}));
    WriteTensor(dir + "/quantization_parameters/v11.irtensor",
                MakeUInt8({2, 2}, {2, 4, 6, 8}));

    DescriptorFields fields;
    fields.scheme = "affine-uint4/v1";
    fields.storage_dtype = "uint4";
    fields.logical_shape = {2, 3};
    fields.granularity = "per_group";
    fields.axis = "1";
    fields.group_size = "2";
    const std::string descriptor = Descriptor(fields);
    WriteManifest(dir, Manifest(
        InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                  descriptor),
        ValueRecord(100, "uint4", {2, 3}, descriptor) + ", " +
            ValueRecord(10, "float32", {2, 2}) + ", " +
            ValueRecord(11, "uint8", {2, 2}),
        ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {2, 2}) +
            ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                  "uint8", {2, 2})));

    const inferref::Testcase testcase = inferref::Testcase::Load(dir);
    Check(BitsEqual(testcase.DequantizeInputToFloat32("weight"),
                    {0x00000000u, 0x40200000u, 0x40000000u, 0xbf200000u,
                     0x00000000u, 0x40100000u}),
          "float32 bits for per_group uint4");
}

// ---------------------------------------------------------------------------
// Fail-closed validation rules.
// ---------------------------------------------------------------------------

void TestFailClosedRules(const std::string &base)
{
    std::printf("fail-closed quantization metadata rules\n");

    DescriptorFields affine;
    const std::string affine_descriptor = Descriptor(affine);
    const std::string affine_values =
        ValueRecord(100, "int8", {2, 3}, affine_descriptor) + ", " +
        ValueRecord(10, "float32", {}) + ", " + ValueRecord(11, "int8", {});

    // Version gates: formats before 0.4 cannot carry quantization at all.
    {
        const std::string dir = BeginCase(base, "version-quantization-parameters");
        WriteManifest(dir, Manifest(
            PlainInputRole("x", "inputs/x.irtensor"), "",
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}),
            "0.2"));
        ExpectLoadRejected(dir, "0.2 rejects quantization_parameters",
                           "format_version '0.4'");
    }
    {
        const std::string dir = BeginCase(base, "version-input-quantization");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values, "", "0.3"));
        ExpectLoadRejected(dir, "0.3 rejects an input quantization descriptor",
                           "format_version '0.4'");
    }
    {
        const std::string dir = BeginCase(base, "version-unknown");
        WriteManifest(dir, Manifest(PlainInputRole("x", "inputs/x.irtensor"), "",
                                    "", "0.5"));
        ExpectLoadRejected(dir, "0.5 is rejected", "unsupported testcase format_version");
    }

    // Path traversal.
    {
        const std::string dir = BeginCase(base, "path-traversal");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "../escape.irtensor", "float32", {}) + ", " +
                ParameterEntry(11, "quantization_parameters/v11.irtensor",
                               "int8", {})));
        ExpectLoadRejected(dir, "parameter payload traversal is rejected",
                           "safe relative path");
    }

    // Missing scale, zero point, and codebook references.
    {
        DescriptorFields fields = affine;
        fields.scale_value_id = 99;
        const std::string dir = BeginCase(base, "missing-scale-reference");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            ValueRecord(100, "int8", {2, 3}, Descriptor(fields)) + ", " +
                ValueRecord(10, "float32", {}),
            ""));
        ExpectLoadRejected(dir, "missing scale reference is rejected",
                           "missing parameter value ID 99");
    }
    {
        DescriptorFields fields = affine;
        fields.zero_point_value_id = "98";
        const std::string dir = BeginCase(base, "missing-zero-reference");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            ValueRecord(100, "int8", {2, 3}, Descriptor(fields)) + ", " +
                ValueRecord(10, "float32", {}),
            ""));
        ExpectLoadRejected(dir, "missing zero-point reference is rejected",
                           "missing parameter value ID 98");
    }
    {
        DescriptorFields fields;
        fields.scheme = "nf4/v1";
        fields.storage_dtype = "uint4";
        fields.zero_point_value_id = "null";
        fields.granularity = "per_block";
        fields.group_size = "4";
        fields.codebook_value_id = 97;
        const std::string dir = BeginCase(base, "missing-codebook-reference");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                      Descriptor(fields)),
            ValueRecord(100, "uint4", {2, 3}, Descriptor(fields)) + ", " +
                ValueRecord(10, "float32", {2}),
            ""));
        ExpectLoadRejected(dir, "missing codebook reference is rejected",
                           "missing parameter value ID 97");
    }

    // Duplicate / malformed / incomplete parameter collections.
    {
        const std::string dir = BeginCase(base, "duplicate-parameter-ids");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " +
                ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {})));
        ExpectLoadRejected(dir, "duplicate parameter IDs are rejected",
                           "duplicate quantization parameter value_id");
    }
    {
        const std::string dir = BeginCase(base, "parameter-id-key");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {},
                           "", "id") +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "an 'id' parameter entry is rejected",
                           "must use the 'value_id' key");
    }
    {
        DescriptorFields fields = affine;
        fields.zero_point_value_id = "null";
        const std::string dir = BeginCase(base, "unreferenced-parameter");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            ValueRecord(100, "int8", {2, 3}, Descriptor(fields)) + ", " +
                ValueRecord(10, "float32", {}) + ", " + ValueRecord(11, "int8", {}),
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "an unreferenced parameter is rejected",
                           "unreferenced: 11");
    }
    {
        const std::string dir = BeginCase(base, "missing-parameter-entry");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values, ""));
        ExpectLoadRejected(dir, "a missing quantization_parameters collection is rejected",
                           "missing quantization_parameters");
    }
    {
        // A declared parameter whose payload file is absent must surface as a
        // TestcaseError, never as a raw IRTensorError from the payload reader.
        const std::string dir = BeginCase(base, "missing-parameter-payload");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "an absent parameter payload is rejected",
                           "payload 'quantization_parameters/v10.irtensor' is"
                           " unreadable");
    }
    {
        const std::string dir = BeginCase(base, "duplicate-value-ids");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values + ", " + ValueRecord(100, "int8", {2, 3}),
            ""));
        ExpectLoadRejected(dir, "duplicate value IDs are rejected",
                           "duplicate value ID");
    }

    // Header versus manifest dtype/shape/nbytes mismatches.
    {
        const std::string dir = BeginCase(base, "parameter-dtype-mismatch");
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "int8", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "parameter manifest dtype mismatch is rejected",
                           "manifest dtype 'int8' differs from payload header dtype"
                           " 'float32'");
    }
    {
        const std::string dir = BeginCase(base, "parameter-shape-mismatch");
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {2}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "parameter manifest shape mismatch is rejected",
                           "manifest shape [2] differs from payload header shape []");
    }
    {
        const std::string dir = BeginCase(base, "parameter-nbytes-mismatch");
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {},
                           "2") +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "parameter manifest nbytes mismatch is rejected",
                           "manifest nbytes differs from payload header nbytes");
    }
    {
        const std::string dir = BeginCase(base, "input-dtype-mismatch");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeFloat32({2, 3}, {0, 0, 0, 0, 0, 0}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "input manifest dtype mismatch is rejected",
                           "manifest dtype 'int8' differs from payload header dtype"
                           " 'float32'");
    }
    {
        DescriptorFields fields = affine;
        fields.logical_shape = {3, 2};
        const std::string dir = BeginCase(base, "input-shape-mismatch");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {3, 2},
                      Descriptor(fields)),
            ValueRecord(100, "int8", {3, 2}, Descriptor(fields)) + ", " +
                ValueRecord(10, "float32", {}) + ", " + ValueRecord(11, "int8", {}),
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "input manifest shape mismatch is rejected",
                           "manifest shape [3, 2] differs from payload header shape"
                           " [2, 3]");
    }
    {
        const std::string dir = BeginCase(base, "input-values-shape-mismatch");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            ValueRecord(100, "int8", {3, 2}, affine_descriptor) + ", " +
                ValueRecord(10, "float32", {}) + ", " + ValueRecord(11, "int8", {}),
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        ExpectLoadRejected(dir, "values record shape disagreement is rejected",
                           "values record dtype/shape differs");
    }

    // Descriptor schema rules.
    {
        DescriptorFields fields = affine;
        fields.granularity = "per_axis";
        fields.axis = "null";
        const std::string dir = BeginCase(base, "per-axis-axis-missing");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "per_axis without an axis is rejected",
                           "axis must be an integer >= 0");
    }
    {
        DescriptorFields fields = affine;
        fields.granularity = "per_group";
        fields.axis = "1";
        fields.group_size = "null";
        const std::string dir = BeginCase(base, "per-group-group-missing");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "per_group without a group size is rejected",
                           "group_size must be an integer >= 1");
    }
    {
        DescriptorFields fields = affine;
        fields.axis = "0";
        const std::string dir = BeginCase(base, "per-tensor-axis-present");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "per_tensor with an axis is rejected",
                           "per_tensor requires axis=null");
    }
    {
        DescriptorFields fields = affine;
        fields.granularity = "per_axis";
        fields.axis = "2";
        const std::string dir = BeginCase(base, "axis-out-of-range");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "an out-of-range axis is rejected",
                           "axis must be less than the logical rank");
    }
    {
        DescriptorFields fields = affine;
        fields.granularity = "per_channel";
        fields.axis = "1";
        const std::string dir = BeginCase(base, "unsupported-granularity");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "an unsupported granularity is rejected",
                           "unsupported granularity 'per_channel'");
    }
    {
        DescriptorFields fields = affine;
        fields.scheme = "affine-int2/v1";
        fields.storage_dtype = "int2";
        const std::string dir = BeginCase(base, "unsupported-scheme");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "an unsupported scheme is rejected",
                           "unsupported quantization scheme");
    }
    {
        DescriptorFields fields = affine;
        fields.extra_fields = "\"codebook_value_id\": 12, \"codebook\": "
                              "\"bitsandbytes-nf4/v1\"";
        const std::string dir = BeginCase(base, "affine-codebook");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "an affine descriptor with codebook fields is rejected",
                           "has unknown field 'codebook");
    }
    {
        DescriptorFields fields = affine;
        fields.zero_point_value_id = "10";
        const std::string dir = BeginCase(base, "scale-equals-zero");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "scale and zero point must have distinct IDs",
                           "distinct value IDs");
    }
    {
        DescriptorFields fields;
        fields.scheme = "nf4/v1";
        fields.storage_dtype = "uint4";
        fields.granularity = "per_block";
        fields.group_size = "4";
        const std::string dir = BeginCase(base, "nf4-zero-point");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "NF4 with a zero point is rejected",
                           "NF4 requires per_block granularity");
    }
    {
        DescriptorFields fields;
        fields.scheme = "nf4/v1";
        fields.storage_dtype = "uint4";
        fields.zero_point_value_id = "null";
        fields.granularity = "per_axis";
        fields.axis = "1";
        fields.group_size = "null";
        const std::string dir = BeginCase(base, "nf4-per-axis");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "NF4 per_axis is rejected",
                           "NF4 requires per_block granularity");
    }
    {
        DescriptorFields fields;
        fields.format_version = "1.1";
        const std::string dir = BeginCase(base, "descriptor-version");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      Descriptor(fields)),
            affine_values, ""));
        ExpectLoadRejected(dir, "an unsupported descriptor format_version is rejected",
                           "unsupported format or format_version");
    }
    {
        DescriptorFields nested = affine;
        nested.scale_value_id = 10;
        nested.zero_point_value_id = "null";
        const std::string dir = BeginCase(base, "nested-parameter");
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            ValueRecord(100, "int8", {2, 3}, affine_descriptor) + ", " +
                ValueRecord(10, "int8", {}, Descriptor(nested)) + ", " +
                ValueRecord(11, "int8", {}),
            ""));
        ExpectLoadRejected(dir, "a nested quantized parameter is rejected",
                           "nested quantized parameters");
    }

    std::printf("dequantization fail-closed rules\n");
    // A valid affine testcase whose scale is zero (Python rejects this).
    {
        const std::string dir = BeginCase(base, "zero-affine-scale");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.0f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        const inferref::Testcase testcase = inferref::Testcase::Load(dir);
        Check(true, "a zero affine scale still loads");
        ExpectDequantizeRejected(testcase, "weight",
                                 "a zero affine scale is rejected by dequantization",
                                 "strictly positive");
    }
    // A valid NF4 testcase with a mutated codebook.
    {
        DescriptorFields fields;
        fields.scheme = "nf4/v1";
        fields.storage_dtype = "uint4";
        fields.zero_point_value_id = "null";
        fields.granularity = "per_block";
        fields.group_size = "4";
        const std::string descriptor = Descriptor(fields);
        const std::string dir = BeginCase(base, "codebook-mismatch");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakePacked4(inferref::DataType::kUInt4, {2, 3},
                                {0, 1, 2, 3, 4, 5}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({2}, {0.5f, 0.5f}));
        inferref::IRTensor codebook = MakeNf4Codebook();
        const float wrong = 0.125f;
        std::memcpy(codebook.data.data(), &wrong, sizeof(wrong));
        WriteTensor(dir + "/quantization_parameters/v12.irtensor", codebook);
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "uint4", {2, 3},
                      descriptor),
            ValueRecord(100, "uint4", {2, 3}, descriptor) + ", " +
                ValueRecord(10, "float32", {2}) + ", " +
                ValueRecord(12, "float32", {16}),
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {2}) +
                ", " + ParameterEntry(12, "quantization_parameters/v12.irtensor",
                                      "float32", {16})));
        const inferref::Testcase testcase = inferref::Testcase::Load(dir);
        ExpectDequantizeRejected(testcase, "weight",
                                 "a noncanonical NF4 codebook is rejected",
                                 "do not match bitsandbytes-nf4/v1");
    }
    // Affine float32 overflow is rejected, matching numpy errstate(over="raise").
    {
        DescriptorFields fields = affine;
        fields.zero_point_value_id = "null";
        const std::string descriptor = Descriptor(fields);
        const std::string dir = BeginCase(base, "overflow");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {127, 127, 127, 127, 127, 127}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {3.4028234663852886e38f}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      descriptor),
            ValueRecord(100, "int8", {2, 3}, descriptor) + ", " +
                ValueRecord(10, "float32", {}),
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {})));
        const inferref::Testcase testcase = inferref::Testcase::Load(dir);
        ExpectDequantizeRejected(testcase, "weight",
                                 "float32 overflow is rejected",
                                 "overflow in float32");
    }
    // API errors for unknown or unquantized roles/parameters.
    {
        const std::string dir = BeginCase(base, "api-errors");
        WriteTensor(dir + "/inputs/weight.irtensor",
                    MakeInt8({2, 3}, {-128, -3, 0, 1, 5, 127}));
        WriteTensor(dir + "/quantization_parameters/v10.irtensor",
                    MakeFloat32({}, {0.5f}));
        WriteTensor(dir + "/quantization_parameters/v11.irtensor", MakeInt8({}, {3}));
        WriteManifest(dir, Manifest(
            InputRole("weight", 100, "inputs/weight.irtensor", "int8", {2, 3},
                      affine_descriptor),
            affine_values,
            ParameterEntry(10, "quantization_parameters/v10.irtensor", "float32", {}) +
                ", " + ParameterEntry(11, "quantization_parameters/v11.irtensor",
                                      "int8", {})));
        const inferref::Testcase testcase = inferref::Testcase::Load(dir);
        bool unknown_role = false;
        try
        {
            (void)testcase.HasInputQuantization("missing");
        }
        catch (const inferref::TestcaseError &)
        {
            unknown_role = true;
        }
        Check(unknown_role, "HasInputQuantization rejects an unknown role");
        bool unknown_parameter = false;
        try
        {
            (void)testcase.QuantizationParameter(999);
        }
        catch (const inferref::TestcaseError &)
        {
            unknown_parameter = true;
        }
        Check(unknown_parameter, "QuantizationParameter rejects an unknown ID");
    }
    {
        // Unquantized 0.2 testcases keep working and expose no quantization.
        const std::string dir = BeginCase(base, "unquantized");
        WriteTensor(dir + "/inputs/x.irtensor", MakeFloat32({1}, {1.0f}));
        WriteManifest(dir, Manifest(PlainInputRole("x", "inputs/x.irtensor"), "",
                                    "", "0.2"));
        const inferref::Testcase testcase = inferref::Testcase::Load(dir);
        Check(!testcase.HasInputQuantization("x"), "0.2 input is not quantized");
        Check(testcase.FormatVersion() == "0.2", "0.2 FormatVersion is preserved");
        bool unquantized = false;
        try
        {
            (void)testcase.DequantizeInputToFloat32("x");
        }
        catch (const inferref::TestcaseError &)
        {
            unquantized = true;
        }
        Check(unquantized, "DequantizeInputToFloat32 rejects an unquantized role");
    }
}

// ---------------------------------------------------------------------------
// Python-generated goldens: float32 bit equality across every case.
// ---------------------------------------------------------------------------

std::uint32_t ParseHexBits(const std::string &text)
{
    return static_cast<std::uint32_t>(
        std::stoul(text, nullptr, 16));
}

void TestGoldenDirectory(const std::string &directory)
{
    std::printf("Python golden agreement from %s\n", directory.c_str());
    const inferref::json::Value root =
        inferref::json::Parse(ReadFile(directory + "/goldens.json"));
    Check(root.IsObject() &&
              root.At("format").string == "inferref-cpp-quantization-goldens",
          "goldens.json format");
    const auto &cases = root.At("cases").array;
    Check(cases.size() >= 12, "goldens.json has at least 12 cases");
    for (const inferref::json::Value &entry : cases)
    {
        const std::string name = entry.At("name").string;
        const std::string case_dir =
            directory + "/" + entry.At("testcase").string;
        const std::string input = entry.At("input").string;
        const inferref::Testcase testcase =
            inferref::Testcase::Load(case_dir);
        Check(testcase.FormatVersion() == "0.4", name + ": format_version 0.4");
        Check(testcase.HasInputQuantization(input), name + ": quantized input");
        const inferref::QuantizationSpec &spec =
            testcase.InputQuantization(input);
        Check(spec.scheme == entry.At("scheme").string, name + ": scheme");
        Check(spec.granularity == entry.At("granularity").string,
              name + ": granularity");
        Check(!testcase.QuantizationParameterIds().empty(),
              name + ": parameters were loaded");

        std::vector<std::uint32_t> expected;
        expected.reserve(entry.At("bits").array.size());
        for (const inferref::json::Value &bits : entry.At("bits").array)
            expected.push_back(ParseHexBits(bits.string));
        const std::vector<float> actual =
            testcase.DequantizeInputToFloat32(input);
        Check(BitsEqual(actual, expected), name + ": C++ float32 bits match Python");

        const inferref::IRTensor reference =
            inferref::ReadIRTensor(case_dir + "/reference/y.irtensor");
        Check(BitsEqual(reference.AsFloat32(), expected),
              name + ": reference payload matches the recorded bits");
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("InferRef testcase 0.4 quantization C++ self-test\n\n");

    const std::string base = "inferref_quantization_selftest_tmp";
    std::filesystem::remove_all(base);
    try
    {
        TestAffineInt8PerTensor(base);
        TestNf4PerBlock(base);
        TestAffineInt4PerAxis(base);
        TestAffineUInt4PerGroup(base);
        TestFailClosedRules(base);
        if (argc > 1)
            TestGoldenDirectory(argv[1]);
    }
    catch (const std::exception &error)
    {
        std::printf("  FAIL uncaught exception: %s\n", error.what());
        ++g_failures;
    }
    std::filesystem::remove_all(base);

    std::printf("\n%s\n", g_failures == 0 ? "All checks passed." : "FAILURES PRESENT.");
    return g_failures == 0 ? 0 : 1;
}
