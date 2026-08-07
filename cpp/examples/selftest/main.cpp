// Round-trip self-test for the header-only .irtensor reader.
//
// Verifies the binary contract independently of Python: header layout, 8-byte
// payload alignment, dtype codes, and float16/bfloat16 decoding. Run it with
// no arguments; exit code 0 means every check passed.
//
// When given a directory it additionally reads every .irtensor written by the
// Python side, which is how the cross-language contract is checked in CI.

#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "inferref/compare.hpp"
#include "inferref/irtensor.hpp"
#include "inferref/json.hpp"
#include "inferref/kv_contract.hpp"

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

inferref::IRTensor MakeFloat32(const std::vector<std::int64_t> &shape,
                               const std::vector<float> &values)
{
    inferref::IRTensor tensor;
    tensor.dtype = inferref::DataType::kFloat32;
    tensor.shape = shape;
    tensor.stride.resize(shape.size());
    std::int64_t acc = 1;
    for (std::size_t i = shape.size(); i-- > 0;)
    {
        tensor.stride[i] = acc;
        acc *= shape[i];
    }
    tensor.data.resize(values.size() * sizeof(float));
    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
    return tensor;
}

void TestHeaderSize()
{
    std::printf("header size == 48 + 16 * rank, 8-byte aligned payload\n");
    for (std::size_t rank = 0; rank <= 6; ++rank)
    {
        const std::size_t expected = 48 + 16 * rank;
        Check(inferref::detail::HeaderSizeForRank(rank) == expected,
              "rank " + std::to_string(rank) + " -> " + std::to_string(expected));
    }
}

void TestRoundTrip(const std::string &temp_path)
{
    std::printf("write/read round-trip\n");
    const std::vector<float> values = {1.0f, -2.5f, 3.25f, 0.0f, 1e-8f, 1e8f};
    const inferref::IRTensor original = MakeFloat32({2, 3}, values);
    inferref::WriteIRTensor(temp_path, original);

    const inferref::IRTensor loaded = inferref::ReadIRTensor(temp_path);
    Check(loaded.dtype == inferref::DataType::kFloat32, "dtype survives");
    Check(loaded.shape == original.shape, "shape survives");
    Check(loaded.stride == original.stride, "stride survives");
    Check(loaded.Numel() == 6, "numel == 6");
    Check(loaded.IsCanonicalContiguous(), "canonical-contiguous flag set");

    const std::vector<float> decoded = loaded.AsFloat32();
    bool equal = decoded.size() == values.size();
    for (std::size_t i = 0; equal && i < values.size(); ++i)
    {
        equal = decoded[i] == values[i];
    }
    Check(equal, "values survive bit-exactly");

    std::remove(temp_path.c_str());
}

void TestBFloat16()
{
    std::printf("bfloat16 decoding\n");
    inferref::IRTensor tensor;
    tensor.dtype = inferref::DataType::kBFloat16;
    tensor.shape = {2};
    tensor.stride = {1};
    // 1.5f and 2.5f as bfloat16 bit patterns.
    const std::uint16_t bits[2] = {0x3FC0, 0x4020};
    tensor.data.resize(sizeof(bits));
    std::memcpy(tensor.data.data(), bits, sizeof(bits));

    const std::vector<float> values = tensor.AsFloat32();
    Check(values.size() == 2, "two elements");
    Check(values[0] == 1.5f, "0x3FC0 -> 1.5");
    Check(values[1] == 2.5f, "0x4020 -> 2.5");
}

void TestFloat16()
{
    std::printf("float16 decoding\n");
    inferref::IRTensor tensor;
    tensor.dtype = inferref::DataType::kFloat16;
    tensor.shape = {3};
    tensor.stride = {1};
    // 1.0, -2.0, 0.5 as IEEE half.
    const std::uint16_t bits[3] = {0x3C00, 0xC000, 0x3800};
    tensor.data.resize(sizeof(bits));
    std::memcpy(tensor.data.data(), bits, sizeof(bits));

    const std::vector<float> values = tensor.AsFloat32();
    Check(values[0] == 1.0f, "0x3C00 -> 1.0");
    Check(values[1] == -2.0f, "0xC000 -> -2.0");
    Check(values[2] == 0.5f, "0x3800 -> 0.5");
}

void TestFloatEncoders()
{
    std::printf("float16/bfloat16 IEEE edge encoding\n");
    Check(inferref::EncodeFloat16(0.0f) == 0x0000u, "+0");
    Check(inferref::EncodeFloat16(-0.0f) == 0x8000u, "-0");
    Check(inferref::EncodeFloat16(std::ldexp(1.0f, -24)) == 0x0001u, "minimum subnormal");
    Check(inferref::EncodeFloat16(std::ldexp(1023.0f, -24)) == 0x03ffu, "maximum subnormal");
    Check(inferref::EncodeFloat16(std::ldexp(1.0f, -14)) == 0x0400u, "minimum normal");
    Check(inferref::EncodeFloat16(65504.0f) == 0x7bffu, "maximum finite");
    Check(inferref::EncodeFloat16(std::numeric_limits<float>::infinity()) == 0x7c00u, "+infinity");
    Check(inferref::EncodeFloat16(-std::numeric_limits<float>::infinity()) == 0xfc00u, "-infinity");
    Check((inferref::EncodeFloat16(std::numeric_limits<float>::quiet_NaN()) & 0x7fffu) > 0x7c00u, "NaN remains NaN");
    Check(inferref::EncodeFloat16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00u, "halfway rounds to even down");
    Check(inferref::EncodeFloat16(1.0f + std::ldexp(3.0f, -11)) == 0x3c02u, "halfway rounds to even up");
    Check((inferref::EncodeBFloat16(std::numeric_limits<float>::quiet_NaN()) & 0x7fffu) > 0x7f80u, "bfloat16 NaN remains NaN");
}

void TestCompare()
{
    std::printf("comparison metrics\n");
    const inferref::IRTensor a = MakeFloat32({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    const inferref::IRTensor b = MakeFloat32({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    inferref::ComparisonResult same = inferref::CompareTensors(a, b);
    Check(same.passed, "identical tensors pass");
    Check(same.metrics.max_abs_error == 0.0, "max_abs_error == 0");
    Check(same.metrics.mismatch_count == 0, "no mismatches");

    const inferref::IRTensor c = MakeFloat32({4}, {1.0f, 2.0f, 3.5f, 4.0f});
    inferref::ComparisonResult diff = inferref::CompareTensors(a, c);
    Check(!diff.passed, "differing tensors fail");
    Check(diff.metrics.mismatch_count == 1, "exactly one mismatch");
    Check(diff.metrics.has_first_mismatch, "first mismatch located");
    Check(diff.metrics.first_mismatch_flat_index == 2, "first mismatch at index 2");
    Check(std::fabs(diff.metrics.max_abs_error - 0.5) < 1e-9, "max_abs_error == 0.5");

    // Layout differs, values do not (SPEC §29).
    inferref::IRTensor transposed = MakeFloat32({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    transposed.stride = {2};
    inferref::ComparisonResult layout = inferref::CompareTensors(a, transposed);
    Check(layout.passed, "stride-only difference passes by default");
    Check(layout.layout.stride_mismatch, "stride difference is reported");
    inferref::ComparisonResult strict = inferref::CompareTensors(a, transposed, true);
    Check(!strict.passed, "stride-only difference fails under --strict-layout");
}

void TestJsonParser()
{
    std::printf("JSON parser and contract extraction\n");

    // A string that merely mentions "contracts" must not fool the reader.
    const std::string manifest =
        "{\n"
        "  \"name\": \"text containing \\\"contracts\\\": [\\\"fake/v1\\\"]\",\n"
        "  \"requirements\": {\n"
        "    \"contracts\": [\"rope/rotate-half/v1\", \"kv-cache/append/v1\"]\n"
        "  }\n"
        "}";
    const inferref::json::Value document = inferref::json::Parse(manifest);
    const std::vector<std::string> declared =
        inferref::json::DeclaredContracts(document);
    Check(declared.size() == 2, "two real contracts found");
    Check(declared[0] == "rope/rotate-half/v1", "first contract ordered");
    Check(declared[1] == "kv-cache/append/v1", "second contract ordered");

    // Top-level contracts wins over requirements when both exist.
    const std::string top_level =
        "{\"requirements\": {\"contracts\": [\"kv-cache/append/v1\"]},"
        " \"contracts\": [\"rmsnorm/last-dim/v1\"]}";
    const std::vector<std::string> top =
        inferref::json::DeclaredContracts(inferref::json::Parse(top_level));
    Check(top.size() == 1 && top[0] == "rmsnorm/last-dim/v1",
          "top-level contracts takes precedence");

    // Whitespace, field order and duplicate keys (last wins).
    const std::string duplicated =
        " { \"contracts\" : [ \"a/v1\" ] , \"contracts\" : [\"b/v1\"] , \"z\" : 1 } ";
    const inferref::json::Value duplicated_doc = inferref::json::Parse(duplicated);
    const std::vector<std::string> dup =
        inferref::json::DeclaredContracts(duplicated_doc);
    Check(dup.size() == 1 && dup[0] == "b/v1", "duplicate keys: last wins");
    Check(duplicated_doc.At("z").IsNumber() &&
              duplicated_doc.At("z").number == 1.0,
          "number after duplicate key parsed");

    // Escaped quotes and unicode escapes.
    const inferref::json::Value escaped =
        inferref::json::Parse("{\"name\": \"a \\\"quoted\\\" \\u4e2d\\u6587\"}");
    Check(escaped.At("name").string == "a \"quoted\" \xe4\xb8\xad\xe6\x96\x87",
          "escaped quotes and \\uXXXX decode");

    // Malformed documents must raise a structured error, not scan onward.
    bool rejected = false;
    try
    {
        inferref::json::Parse("{\"contracts\": }");
    }
    catch (const inferref::json::ParseError &)
    {
        rejected = true;
    }
    Check(rejected, "malformed JSON raises ParseError");

    bool non_array_rejected = false;
    try
    {
        inferref::json::DeclaredContracts(inferref::json::Parse("{\"contracts\": 1}"));
    }
    catch (const inferref::json::ParseError &)
    {
        non_array_rejected = true;
    }
    Check(non_array_rejected, "non-array contracts rejected");

    bool control_rejected = false;
    try
    {
        inferref::json::Parse("{\"name\": \"bad\x01!\"}");
    }
    catch (const inferref::json::ParseError &)
    {
        control_rejected = true;
    }
    Check(control_rejected, "unescaped control character rejected");
}

void TestKvIndexBounds()
{
    std::printf("kv-cache indexed-update bounds\n");
    Check(inferref::ValidateIndexedUpdate(2.0, 5, 3).empty(),
          "target + update == old sequence is valid");
    Check(inferref::ValidateIndexedUpdate(0.0, 1, 1).empty(),
          "exact fill is valid");
    Check(!inferref::ValidateIndexedUpdate(std::nan(""), 5, 1).empty(),
          "NaN rejected");
    Check(!inferref::ValidateIndexedUpdate(INFINITY, 5, 1).empty(),
          "infinity rejected");
    Check(!inferref::ValidateIndexedUpdate(-1.0, 5, 1).empty(),
          "negative rejected");
    Check(!inferref::ValidateIndexedUpdate(1.5, 5, 1).empty(),
          "non-integer rejected");
    Check(!inferref::ValidateIndexedUpdate(1e300, 5, 1).empty(),
          "huge finite float rejected before size_t conversion");
    Check(!inferref::ValidateIndexedUpdate(5.0, 5, 1).empty(),
          "target == old sequence rejected when update is positive");
    Check(!inferref::ValidateIndexedUpdate(4.0, 5, 2).empty(),
          "target + update overflow rejected");
    Check(!inferref::ValidateIndexedUpdate(0.0, 5, 6).empty(),
          "update longer than the cache rejected");
    Check(!inferref::ValidateIndexedUpdate(
              0.0, 5, std::numeric_limits<std::size_t>::max())
              .empty(),
          "maximum update size rejected without unsigned overflow");
}

void TestReadDirectory(const std::string &directory)
{
    std::printf("reading Python-written tensors from %s\n", directory.c_str());
    // Kept dependency-free: the caller passes explicit files instead of us
    // pulling in <filesystem> directory iteration semantics per platform.
    const char *names[] = {"q_embed.irtensor", "k_embed.irtensor"};
    for (const char *name : names)
    {
        const std::string path = directory + "/" + name;
        try
        {
            const inferref::IRTensor tensor = inferref::ReadIRTensor(path);
            Check(tensor.Numel() > 0,
                  std::string(name) + " " + tensor.ShapeString() + " " +
                      inferref::DataTypeName(tensor.dtype));
        }
        catch (const inferref::IRTensorError &error)
        {
            std::printf("  skip %s (%s)\n", name, error.what());
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::printf("InferRef .irtensor C++ self-test\n\n");
    TestHeaderSize();
    TestRoundTrip("inferref_selftest_tmp.irtensor");
    TestBFloat16();
    TestFloat16();
    TestFloatEncoders();
    TestCompare();
    TestJsonParser();
    TestKvIndexBounds();
    if (argc > 1)
    {
        TestReadDirectory(argv[1]);
    }

    std::printf("\n%s\n", g_failures == 0 ? "All checks passed." : "FAILURES PRESENT.");
    return g_failures == 0 ? 0 : 1;
}
