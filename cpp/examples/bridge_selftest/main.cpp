// Self-test for the generic runtime bridge (cpp/include/inferref/bridge.hpp).
//
// Covers end-to-end dispatch with a KV-append stub engine, missing-region
// failure, missing-output-role failure, and output-directory redirection.
// With one argument it also loads the committed 0.4 quantization goldens
// directory and checks that the Testcase-aware callback can read the
// descriptor, parameters, and dequantized values, that the legacy callback
// refuses a quantized testcase with kBridgeUnsupportedQuantization, and that
// an unquantized testcase still dispatches on both paths.
// With two arguments it loads the given testcase directory and writes engine
// outputs to the given output directory, which is how the fixture
// cross-language comparison is exercised in development.

#include <inferref/bridge.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
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

std::vector<float> FloatValues(const inferref::IRTensor &tensor)
{
    return tensor.AsFloat32();
}

std::uint32_t FloatBits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool FloatBitsEqual(const std::vector<float> &left,
                    const std::vector<float> &right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        if (FloatBits(left[i]) != FloatBits(right[i]))
            return false;
    }
    return true;
}

void WriteFile(const std::string &path, const std::string &text)
{
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

std::vector<std::int64_t> Coordinates(std::int64_t linear,
                                      const std::vector<std::int64_t> &shape,
                                      const std::vector<std::int64_t> &stride)
{
    std::vector<std::int64_t> coord(shape.size());
    std::int64_t remaining = linear;
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        coord[i] = remaining / stride[i];
        remaining %= stride[i];
    }
    return coord;
}

std::int64_t FlatIndex(const std::vector<std::int64_t> &coord,
                       const std::vector<std::int64_t> &stride)
{
    std::int64_t index = 0;
    for (std::size_t i = 0; i < coord.size(); ++i)
    {
        index += coord[i] * stride[i];
    }
    return index;
}

// Concat two float32 tensors along one axis (rank-2 sequence axis for KV).
inferref::IRTensor ConcatAlong(const inferref::IRTensor &a,
                               const inferref::IRTensor &b,
                               std::size_t axis)
{
    if (a.dtype != inferref::DataType::kFloat32 ||
        b.dtype != inferref::DataType::kFloat32)
    {
        throw std::runtime_error("stub concat supports float32 only");
    }
    if (a.shape.size() != b.shape.size() || axis >= a.shape.size())
        throw std::runtime_error("stub concat rank mismatch");
    for (std::size_t i = 0; i < a.shape.size(); ++i)
    {
        if (i != axis && a.shape[i] != b.shape[i])
            throw std::runtime_error("stub concat non-axis shape mismatch");
    }
    inferref::IRTensor out;
    out.dtype = inferref::DataType::kFloat32;
    out.shape = a.shape;
    out.shape[axis] = a.shape[axis] + b.shape[axis];
    out.stride.resize(out.shape.size());
    std::int64_t acc = 1;
    for (std::size_t i = out.shape.size(); i-- > 0;)
    {
        out.stride[i] = acc;
        acc *= out.shape[i];
    }
    out.data.resize(static_cast<std::size_t>(out.Numel()) * sizeof(float));
    const float *pa = reinterpret_cast<const float *>(a.data.data());
    const float *pb = reinterpret_cast<const float *>(b.data.data());
    float *po = reinterpret_cast<float *>(out.data.data());
    const std::int64_t a_len = a.shape[axis];
    for (std::int64_t linear = 0; linear < out.Numel(); ++linear)
    {
        std::vector<std::int64_t> coord =
            Coordinates(linear, out.shape, out.stride);
        const bool from_a = coord[axis] < a_len;
        if (!from_a)
            coord[axis] -= a_len;
        const std::vector<std::int64_t> &stride =
            from_a ? a.stride : b.stride;
        const float *source = from_a ? pa : pb;
        po[linear] = source[FlatIndex(coord, stride)];
    }
    return out;
}

// (update * scale).sum(axis=rank-2, keepdims=True) for float32.
inferref::IRTensor ReduceLogits(const inferref::IRTensor &update,
                                const inferref::IRTensor &scale)
{
    if (update.dtype != inferref::DataType::kFloat32 ||
        scale.dtype != inferref::DataType::kFloat32 || scale.Numel() != 1)
    {
        throw std::runtime_error("stub logits requires float32 scalar scale");
    }
    const std::size_t rank = update.shape.size();
    const std::size_t seq_axis = rank - 2;
    inferref::IRTensor out;
    out.dtype = inferref::DataType::kFloat32;
    out.shape = update.shape;
    out.shape[seq_axis] = 1;
    out.stride.resize(rank);
    std::int64_t acc = 1;
    for (std::size_t i = rank; i-- > 0;)
    {
        out.stride[i] = acc;
        acc *= out.shape[i];
    }
    out.data.resize(static_cast<std::size_t>(out.Numel()) * sizeof(float),
                    std::byte{0});
    const float *pu = reinterpret_cast<const float *>(update.data.data());
    float *po = reinterpret_cast<float *>(out.data.data());
    const float factor =
        reinterpret_cast<const float *>(scale.data.data())[0];
    for (std::int64_t linear = 0; linear < update.Numel(); ++linear)
    {
        std::vector<std::int64_t> coord =
            Coordinates(linear, update.shape, update.stride);
        coord[seq_axis] = 0;
        po[FlatIndex(coord, out.stride)] += pu[linear] * factor;
    }
    return out;
}

std::map<std::string, inferref::IRTensor> KVInvoke(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)
{
    if (region_name != "kv-cache/append/v1")
        throw std::runtime_error("unknown region: " + region_name);
    const auto cache = inputs.find("cache");
    const auto update = inputs.find("update");
    const auto scale = inputs.find("scale");
    if (cache == inputs.end() || update == inputs.end() ||
        scale == inputs.end())
    {
        throw std::runtime_error("stub requires cache, update, and scale");
    }
    return {
        {"cache_out",
         ConcatAlong(cache->second, update->second,
                     cache->second.shape.size() - 2)},
        {"logits", ReduceLogits(update->second, scale->second)},
    };
}

void WriteKvTestcase(const std::string &dir)
{
    std::filesystem::create_directories(dir + "/inputs");
    std::vector<float> cache_values(64);
    for (std::size_t i = 0; i < cache_values.size(); ++i)
        cache_values[i] = static_cast<float>(i + 1);
    std::vector<float> update_values(32);
    for (std::size_t i = 0; i < update_values.size(); ++i)
        update_values[i] = static_cast<float>(100 + i);
    inferref::WriteIRTensor(dir + "/inputs/cache.irtensor",
                            MakeFloat32({1, 2, 4, 8}, cache_values));
    inferref::WriteIRTensor(dir + "/inputs/update.irtensor",
                            MakeFloat32({1, 2, 2, 8}, update_values));
    inferref::WriteIRTensor(dir + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {2.0f}));
    WriteFile(
        dir + "/testcase.json",
        "{\n"
        "  \"format\": \"inferref-testcase\",\n"
        "  \"format_version\": \"0.2\",\n"
        "  \"name\": \"bridge-fixture\",\n"
        "  \"origin\": {\"region\": \"kv-cache/append/v1\"},\n"
        "  \"inputs\": [\n"
        "    {\"name\": \"cache\", \"payload\": \"inputs/cache.irtensor\"},\n"
        "    {\"name\": \"update\", \"payload\": \"inputs/update.irtensor\"},\n"
        "    {\"name\": \"scale\", \"payload\": \"inputs/scale.irtensor\"}\n"
        "  ],\n"
        "  \"outputs\": [\n"
        "    {\"name\": \"cache_out\"},\n"
        "    {\"name\": \"logits\"}\n"
        "  ]\n"
        "}\n");
}

void TestEndToEnd(const std::string &base)
{
    std::printf("end-to-end KV dispatch through RunBridge\n");
    const std::string testcase_dir = base + "/testcase";
    const std::string output_dir = base + "/output";
    std::filesystem::create_directories(testcase_dir);
    std::filesystem::create_directories(output_dir);
    WriteKvTestcase(testcase_dir);

    const int code =
        inferref::RunBridge(testcase_dir, output_dir, KVInvoke);
    Check(code == inferref::kBridgeOk, "RunBridge returns 0");

    const inferref::IRTensor cache_out =
        inferref::ReadIRTensor(output_dir + "/cache_out.irtensor");
    Check(cache_out.shape == std::vector<std::int64_t>({1, 2, 6, 8}),
          "cache_out shape [1,2,6,8]");
    const std::vector<float> cache_values = FloatValues(cache_out);
    // [0,1,4,3] is the first appended block of update at [0,1,0,3].
    Check(cache_values[1 * 48 + 4 * 8 + 3] == 119.0f,
          "cache_out appends update at the sequence axis");
    Check(cache_values[1 * 48 + 5 * 8 + 7] == 131.0f,
          "second appended block matches update");

    const inferref::IRTensor logits =
        inferref::ReadIRTensor(output_dir + "/logits.irtensor");
    Check(logits.shape == std::vector<std::int64_t>({1, 2, 1, 8}),
          "logits shape [1,2,1,8]");
    const std::vector<float> logit_values = FloatValues(logits);
    // update[0,1,0,3]=119, update[0,1,1,3]=127 -> (119+127)*2 = 492.
    Check(logit_values[1 * 8 + 3] == 492.0f, "logits sum update * scale");

    const inferref::json::Value manifest =
        inferref::json::Parse(ReadFile(output_dir + "/manifest.json"));
    Check(manifest.At("outputs").array.size() == 2 &&
              manifest.At("outputs").array[0].At("name").string == "cache_out" &&
              manifest.At("outputs").array[0].At("payload").string ==
                  "cache_out.irtensor",
          "manifest written to the output directory");
    Check(!std::filesystem::exists(testcase_dir + "/cache_out.irtensor"),
          "testcase directory is not polluted with outputs");
}

void TestMissingRegion(const std::string &base)
{
    std::printf("missing region dispatch\n");
    const std::string dir = base + "/missing";
    std::filesystem::create_directories(dir + "/inputs");
    inferref::WriteIRTensor(dir + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(dir + "/testcase.json",
              "{\"format\": \"inferref-testcase\","
              " \"format_version\": \"0.2\","
              " \"inputs\": [{\"name\": \"x\", "
              "\"payload\": \"inputs/x.irtensor\"}],"
              " \"outputs\": [{\"name\": \"y\"}]}");
    const int code =
        inferref::RunBridge(dir, base + "/missing-out", KVInvoke);
    Check(code == inferref::kBridgeMissingRegion,
          "missing region returns missing_region exit code");
    Check(!std::filesystem::exists(base + "/missing-out/manifest.json"),
          "no output published for missing region");
}

void TestMissingOutputRole(const std::string &base)
{
    std::printf("DebugInvoke missing an output role\n");
    const std::string dir = base + "/partial";
    std::filesystem::create_directories(dir);
    WriteKvTestcase(dir);
    inferref::DebugInvoke partial =
        [](const std::string &region_name,
           const std::map<std::string, inferref::IRTensor> &inputs)
    {
        std::map<std::string, inferref::IRTensor> outputs =
            KVInvoke(region_name, inputs);
        outputs.erase("logits");
        return outputs;
    };
    const int code = inferref::RunBridge(dir, base + "/partial-out", partial);
    Check(code == inferref::kBridgeError, "missing output role is an error");
}

void TestExtraOutputRole(const std::string &base)
{
    std::printf("DebugInvoke returning an undeclared output\n");
    const std::string dir = base + "/extra";
    std::filesystem::create_directories(dir);
    WriteKvTestcase(dir);
    inferref::DebugInvoke extra =
        [](const std::string &region_name,
           const std::map<std::string, inferref::IRTensor> &inputs)
    {
        std::map<std::string, inferref::IRTensor> outputs =
            KVInvoke(region_name, inputs);
        outputs["surprise"] = outputs.begin()->second;
        return outputs;
    };
    const int code = inferref::RunBridge(dir, base + "/extra-out", extra);
    Check(code == inferref::kBridgeError, "undeclared output role is an error");
    Check(!std::filesystem::exists(base + "/extra-out/cache_out.irtensor"),
          "no partial outputs written for undeclared role");
}

void TestUnquantizedTestcaseAwareDispatch(const std::string &base)
{
    std::printf("testcase-aware dispatch of an unquantized testcase\n");
    const std::string testcase_dir = base + "/unquantized-ex";
    const std::string output_dir = base + "/unquantized-ex-out";
    std::filesystem::create_directories(testcase_dir);
    WriteKvTestcase(testcase_dir);

    bool saw_unquantized = false;
    int calls = 0;
    const inferref::DebugInvokeEx invoke =
        [&](const inferref::Testcase &testcase, const std::string &region_name,
            const std::map<std::string, inferref::IRTensor> &inputs)
        -> std::map<std::string, inferref::IRTensor>
    {
        ++calls;
        saw_unquantized =
            testcase.FormatVersion() == "0.2" &&
            !testcase.HasInputQuantization("cache") &&
            !testcase.HasInputQuantization("update") &&
            !testcase.HasInputQuantization("scale");
        return KVInvoke(region_name, inputs);
    };
    const int code =
        inferref::RunBridge(testcase_dir, output_dir, invoke);
    Check(code == inferref::kBridgeOk,
          "DebugInvokeEx returns 0 for an unquantized testcase");
    Check(calls == 1, "DebugInvokeEx is called once");
    Check(saw_unquantized,
          "DebugInvokeEx sees no quantization on a 0.2 testcase");
    Check(std::filesystem::exists(output_dir + "/manifest.json"),
          "testcase-aware dispatch publishes manifest.json");
}

// The committed 0.4 fixture from tests/fixtures/cpp_quantization_goldens: the
// per_group case exercises a non-trivial axis, group size, scale, and zero
// point, so the descriptor query is not a per_tensor special case.
void TestQuantizedDispatch(const std::string &goldens_dir)
{
    std::printf("quantized 0.4 dispatch through RunBridge\n");
    const std::string testcase_dir = goldens_dir + "/cases/int8-per-group";
    Check(std::filesystem::exists(testcase_dir + "/testcase.json"),
          "committed int8-per-group 0.4 fixture is present");

    const std::string legacy_output = "inferref_bridge_selftest_legacy_out";
    std::filesystem::remove_all(legacy_output);
    const int legacy_code =
        inferref::RunBridge(testcase_dir, legacy_output, KVInvoke);
    Check(legacy_code == inferref::kBridgeUnsupportedQuantization,
          "legacy DebugInvoke returns kBridgeUnsupportedQuantization (4)");
    Check(!std::filesystem::exists(legacy_output + "/manifest.json"),
          "legacy path does not dispatch or write outputs");

    const std::string output_dir = "inferref_bridge_selftest_quantized_out";
    std::filesystem::remove_all(output_dir);
    int calls = 0;
    bool saw_testcase = false;
    bool saw_quantization = false;
    bool descriptor_matches = false;
    bool parameters_match = false;
    bool inputs_match = false;
    bool region_matches = false;
    std::vector<float> dequantized;
    const inferref::DebugInvokeEx invoke =
        [&](const inferref::Testcase &testcase, const std::string &region_name,
            const std::map<std::string, inferref::IRTensor> &inputs)
        -> std::map<std::string, inferref::IRTensor>
    {
        ++calls;
        saw_testcase = testcase.FormatVersion() == "0.4";
        saw_quantization = testcase.HasInputQuantization("weight");
        const inferref::QuantizationSpec &spec =
            testcase.InputQuantization("weight");
        descriptor_matches =
            spec.scheme == "affine-int8/v1" &&
            spec.granularity == "per_group" && spec.has_axis && spec.axis == 1 &&
            spec.has_group_size && spec.group_size == 3;
        const std::map<std::int64_t, inferref::IRTensor> parameters =
            testcase.QuantizationParameters();
        const inferref::IRTensor scale =
            testcase.QuantizationParameter(spec.scale_value_id);
        const inferref::IRTensor zero_point =
            testcase.QuantizationParameter(spec.zero_point_value_id);
        parameters_match =
            testcase.QuantizationParameterIds() ==
                std::vector<std::int64_t>({10, 11}) &&
            parameters.size() == 2 &&
            scale.dtype == inferref::DataType::kFloat32 &&
            scale.shape == std::vector<std::int64_t>({4, 4}) &&
            zero_point.dtype == inferref::DataType::kInt8;
        const auto weight = inputs.find("weight");
        inputs_match = weight != inputs.end() &&
                       weight->second.dtype == inferref::DataType::kInt8 &&
                       weight->second.shape ==
                           std::vector<std::int64_t>({4, 10});
        region_matches = region_name == "quantization/reference/v1";
        dequantized = testcase.DequantizeInputToFloat32("weight");
        return {{"y", MakeFloat32(spec.logical_shape, dequantized)}};
    };

    const int code = inferref::RunBridge(testcase_dir, output_dir, invoke);
    Check(code == inferref::kBridgeOk,
          "DebugInvokeEx dispatch of a 0.4 testcase returns 0");
    Check(calls == 1, "DebugInvokeEx receives the loaded Testcase once");
    Check(saw_testcase, "callback sees format_version 0.4");
    Check(saw_quantization, "HasInputQuantization(\"weight\") is true");
    Check(descriptor_matches,
          "InputQuantization returns the per_group descriptor");
    Check(parameters_match,
          "QuantizationParameters returns scale and zero point");
    Check(inputs_match, "named inputs still carry the raw int8 codes");
    Check(region_matches, "region name is passed through");

    const inferref::IRTensor reference =
        inferref::ReadIRTensor(testcase_dir + "/reference/y.irtensor");
    Check(FloatBitsEqual(dequantized, reference.AsFloat32()),
          "dequantized input matches the committed float32 reference");
    const inferref::IRTensor written =
        inferref::ReadIRTensor(output_dir + "/y.irtensor");
    Check(FloatBitsEqual(written.AsFloat32(), dequantized),
          "written output matches the dequantized input bits");
    Check(std::filesystem::exists(output_dir + "/manifest.json"),
          "quantized dispatch publishes manifest.json");
    std::filesystem::remove_all(legacy_output);
    std::filesystem::remove_all(output_dir);
}

} // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("InferRef bridge.hpp C++ self-test\n\n");

    if (argc > 2)
    {
        // External mode: run the KV stub against a real testcase directory and
        // write outputs to the given output directory.
        std::printf("bridge run: %s -> %s\n", argv[1], argv[2]);
        return inferref::RunBridge(argv[1], argv[2], KVInvoke);
    }

    const std::string goldens_dir = argc > 1 ? argv[1] : std::string();
    const std::string base = "inferref_bridge_selftest_tmp";
    std::filesystem::remove_all(base);
    try
    {
        TestEndToEnd(base);
        TestMissingRegion(base);
        TestMissingOutputRole(base);
        TestExtraOutputRole(base);
        TestUnquantizedTestcaseAwareDispatch(base);
        if (goldens_dir.empty())
            std::printf("quantization fixtures not provided; skipping\n");
        else
            TestQuantizedDispatch(goldens_dir);
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
