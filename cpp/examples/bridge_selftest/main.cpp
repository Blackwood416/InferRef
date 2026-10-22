// Self-test for the generic runtime bridge (cpp/include/inferref/bridge.hpp).
//
// Covers end-to-end dispatch with a KV-append stub engine, missing-region
// failure, missing-output-role failure, output-directory redirection, and
// batch-manifest execution: validation, ordered dispatch in one process,
// partial failure, per-item manifests, and batch-results.json.
// With one argument it also loads the committed 0.4 quantization goldens
// directory and checks that the Testcase-aware callback can read the
// descriptor, parameters, and dequantized values, that the legacy callback
// refuses a quantized testcase with kBridgeUnsupportedQuantization, and that
// an unquantized testcase still dispatches on both paths.
// With two arguments it loads the given testcase directory and writes engine
// outputs to the given output directory, which is how the fixture
// cross-language comparison is exercised in development.
// With `--batch-manifest FILE` it runs that manifest through the KV stub,
// which is how the Python batch test drives a real engine.

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

std::string JsonQuote(const std::string &value)
{
    std::string out = "\"";
    for (const char ch : value)
    {
        if (ch == '"' || ch == '\\')
            out += '\\';
        out += ch;
    }
    out += "\"";
    return out;
}

// One testcase whose region the KV stub does not implement: the item is
// loaded and dispatched, then fails inside the callback.
void WriteUnsupportedRegionTestcase(const std::string &dir)
{
    std::filesystem::create_directories(dir + "/inputs");
    inferref::WriteIRTensor(dir + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(dir + "/testcase.json",
              "{\"format\": \"inferref-testcase\","
              " \"format_version\": \"0.2\","
              " \"origin\": {\"region\": \"unsupported/region/v1\"},"
              " \"inputs\": [{\"name\": \"x\","
              " \"payload\": \"inputs/x.irtensor\"}],"
              " \"outputs\": [{\"name\": \"y\"}]}");
}

void WriteBatchManifest(const std::string &path,
                        const std::vector<inferref::BatchItem> &items)
{
    std::string text =
        "{\n  \"format\": \"inferref-batch\",\n"
        "  \"format_version\": \"1.0\",\n  \"items\": [\n";
    for (std::size_t i = 0; i < items.size(); ++i)
    {
        text += "    {\"id\": " + JsonQuote(items[i].id) +
                ", \"testcase_dir\": " + JsonQuote(items[i].testcase_dir) +
                ", \"output_dir\": " + JsonQuote(items[i].output_dir) + "}";
        text += i + 1 == items.size() ? "\n" : ",\n";
    }
    text += "  ]\n}\n";
    WriteFile(path, text);
}

inferref::json::Value LoadBatchResults(const std::string &manifest_dir)
{
    return inferref::json::Parse(ReadFile(manifest_dir +
                                          "/batch-results.json"));
}

void TestBatchManifestValidation(const std::string &base)
{
    std::printf("batch manifest validation\n");
    const std::string dir = base + "/batch-manifest";
    std::filesystem::create_directories(dir);

    // A valid manifest resolves relative item paths against its own directory
    // and keeps manifest order.
    const std::string valid_path = dir + "/valid.json";
    WriteBatchManifest(valid_path, {{"first", "cases/first", "out/first"},
                                    {"second", "cases/second", "out/second"}});
    const std::vector<inferref::BatchItem> items =
        inferref::LoadBatchManifest(valid_path);
    Check(items.size() == 2 && items[0].id == "first" &&
              items[1].id == "second",
          "valid manifest keeps item order and ids");
    Check(items[0].testcase_dir ==
              std::filesystem::path(dir + "/cases/first").generic_string(),
          "relative testcase_dir resolves against the manifest directory");
    Check(items[1].output_dir ==
              std::filesystem::path(dir + "/out/second").generic_string(),
          "relative output_dir resolves against the manifest directory");

    struct InvalidCase
    {
        const char *name;
        const char *text;
        const char *fragment;
    };
    const InvalidCase cases[] = {
        {"not-json", "{", "not valid JSON"},
        {"wrong-format",
         "{\"format\": \"inferref-other\", \"format_version\": \"1.0\","
         " \"items\": [{\"id\": \"a\", \"testcase_dir\": \"a\","
         " \"output_dir\": \"b\"}]}",
         "format must be 'inferref-batch'"},
        {"wrong-version",
         "{\"format\": \"inferref-batch\", \"format_version\": \"2.0\","
         " \"items\": [{\"id\": \"a\", \"testcase_dir\": \"a\","
         " \"output_dir\": \"b\"}]}",
         "format_version"},
        {"empty-items",
         "{\"format\": \"inferref-batch\", \"format_version\": \"1.0\","
         " \"items\": []}",
         "must not be empty"},
        {"missing-id",
         "{\"format\": \"inferref-batch\", \"format_version\": \"1.0\","
         " \"items\": [{\"testcase_dir\": \"a\","
         " \"output_dir\": \"b\"}]}",
         "missing 'id'"},
        {"missing-output-dir",
         "{\"format\": \"inferref-batch\", \"format_version\": \"1.0\","
         " \"items\": [{\"id\": \"a\", \"testcase_dir\": \"a\"}]}",
         "missing 'output_dir'"},
        {"duplicate-id",
         "{\"format\": \"inferref-batch\", \"format_version\": \"1.0\","
         " \"items\": [{\"id\": \"a\", \"testcase_dir\": \"a\","
         " \"output_dir\": \"b\"}, {\"id\": \"a\","
         " \"testcase_dir\": \"c\", \"output_dir\": \"d\"}]}",
         "duplicate batch item id"},
        {"traversal",
         "{\"format\": \"inferref-batch\", \"format_version\": \"1.0\","
         " \"items\": [{\"id\": \"a\","
         " \"testcase_dir\": \"../escape\", \"output_dir\": \"b\"}]}",
         "'..' path segments"},
    };
    for (const InvalidCase &invalid : cases)
    {
        const std::string path = dir + "/" + invalid.name + ".json";
        WriteFile(path, invalid.text);
        try
        {
            inferref::LoadBatchManifest(path);
            Check(false, std::string("rejects ") + invalid.name);
        }
        catch (const inferref::BatchManifestError &error)
        {
            Check(std::string(error.what()).find(invalid.fragment) !=
                      std::string::npos,
                  std::string("rejects ") + invalid.name + " (" +
                      error.what() + ")");
        }
    }

    // A rejected manifest runs no item and publishes no results file.
    const inferref::DebugInvoke invoke =
        [](const std::string &,
           const std::map<std::string, inferref::IRTensor> &)
        -> std::map<std::string, inferref::IRTensor>
    { throw std::runtime_error("callback must not run"); };
    const int code =
        inferref::RunBridgeBatch(dir + "/traversal.json", invoke);
    Check(code == inferref::kBridgeError,
          "invalid batch manifest returns kBridgeError");
    Check(!std::filesystem::exists(dir + "/batch-results.json"),
          "invalid batch manifest writes no batch-results.json");
}

void TestBatchExecution(const std::string &base)
{
    std::printf("batch manifest with one failing item\n");
    const std::string dir = base + "/batch-mixed";
    WriteKvTestcase(dir + "/case-a");
    WriteUnsupportedRegionTestcase(dir + "/case-b");
    WriteKvTestcase(dir + "/case-c");
    const std::string manifest_path = dir + "/batch.json";
    WriteBatchManifest(manifest_path, {{"step-a", "case-a", "out-a"},
                                       {"step-b", "case-b", "out-b"},
                                       {"step-c", "case-c", "out-c"}});

    std::vector<std::string> regions;
    int calls = 0;
    const inferref::DebugInvoke invoke =
        [&](const std::string &region_name,
            const std::map<std::string, inferref::IRTensor> &inputs)
    {
        ++calls;
        regions.push_back(region_name);
        return KVInvoke(region_name, inputs);
    };
    const int code = inferref::RunBridgeBatch(manifest_path, invoke);
    Check(code == inferref::kBridgeError,
          "a batch with a failing item exits non-zero");
    Check(calls == 3, "one callback invocation per item in one process");
    Check(regions == std::vector<std::string>({"kv-cache/append/v1",
                                               "unsupported/region/v1",
                                               "kv-cache/append/v1"}),
          "items dispatch in manifest order");
    Check(std::filesystem::exists(dir + "/out-a/manifest.json"),
          "first item writes its manifest.json");
    Check(!std::filesystem::exists(dir + "/out-b/manifest.json"),
          "failing item writes no manifest.json");
    Check(std::filesystem::exists(dir + "/out-c/manifest.json"),
          "execution continues after a failing item");

    const inferref::json::Value results = LoadBatchResults(dir);
    Check(results.At("format").string == inferref::kBatchResultsFormat,
          "batch-results.json format is inferref-batch-results");
    Check(results.At("format_version").string == "1.0",
          "batch-results.json format_version is 1.0");
    const inferref::json::Array &recorded = results.At("items").array;
    Check(recorded.size() == 3,
          "batch-results.json records every item in order");
    Check(recorded[0].At("id").string == "step-a" &&
              recorded[1].At("id").string == "step-b" &&
              recorded[2].At("id").string == "step-c",
          "batch-results.json keeps item ids in manifest order");
    Check(recorded[0].At("status").string == "passed" &&
              recorded[2].At("status").string == "passed",
          "passing items are recorded as passed");
    Check(recorded[1].At("status").string == "failed" &&
              static_cast<int>(recorded[1].At("exit_code").number) ==
                  inferref::kBridgeError,
          "the failing item records status and exit code");
    Check(recorded[1].At("error").IsString() &&
              recorded[1].At("error").string.find("unknown region") !=
                  std::string::npos,
          "the failing item records its diagnostic");
    Check(recorded[0].At("error").IsNull(),
          "passing items record a null error");
    Check(recorded[0].At("testcase_dir").string ==
              std::filesystem::path(dir + "/case-a").generic_string() &&
              recorded[0].At("output_dir").string ==
                  std::filesystem::path(dir + "/out-a").generic_string(),
          "batch-results.json records resolved item paths");
    const inferref::json::Value &counts = results.At("counts");
    Check(static_cast<int>(counts.At("total").number) == 3 &&
              static_cast<int>(counts.At("passed").number) == 2 &&
              static_cast<int>(counts.At("failed").number) == 1,
          "batch-results.json counts total, passed, and failed");
}

void TestBatchAllPass(const std::string &base)
{
    std::printf("batch manifest with every item passing\n");
    const std::string dir = base + "/batch-pass";
    WriteKvTestcase(dir + "/case-1");
    WriteKvTestcase(dir + "/case-2");
    const std::string manifest_path = dir + "/batch.json";
    WriteBatchManifest(manifest_path, {{"step-1", "case-1", "out-1"},
                                       {"step-2", "case-2", "out-2"}});

    int calls = 0;
    bool saw_testcase = false;
    const inferref::DebugInvokeEx invoke =
        [&](const inferref::Testcase &testcase, const std::string &region_name,
            const std::map<std::string, inferref::IRTensor> &inputs)
        -> std::map<std::string, inferref::IRTensor>
    {
        ++calls;
        saw_testcase = testcase.FormatVersion() == "0.2";
        return KVInvoke(region_name, inputs);
    };
    const int code = inferref::RunBridgeBatch(manifest_path, invoke);
    Check(code == inferref::kBridgeOk,
          "an all-passing batch exits zero");
    Check(calls == 2, "testcase-aware callback runs once per item");
    Check(saw_testcase, "batch callback receives the loaded Testcase");
    Check(std::filesystem::exists(dir + "/out-1/manifest.json") &&
              std::filesystem::exists(dir + "/out-2/manifest.json"),
          "every passing item writes its manifest.json");

    const inferref::json::Value results = LoadBatchResults(dir);
    const inferref::json::Array &recorded = results.At("items").array;
    Check(recorded.size() == 2 &&
              recorded[0].At("status").string == "passed" &&
              recorded[1].At("status").string == "passed",
          "every item is recorded as passed");
    Check(static_cast<int>(results.At("counts").At("passed").number) == 2 &&
              static_cast<int>(results.At("counts").At("failed").number) == 0,
          "counts report no failures");
}

// The legacy batch callback must refuse a quantized item exactly like the
// single-testcase legacy path (exit code 4) and still run later items.
void TestBatchQuantizationGuard(const std::string &base,
                                const std::string &goldens_dir)
{
    std::printf("batch legacy callback refuses a quantized item\n");
    const std::string dir = base + "/batch-quantized";
    WriteKvTestcase(dir + "/case-plain");
    const std::string quantized_dir =
        std::filesystem::absolute(std::filesystem::path(goldens_dir) /
                                  "cases/int8-per-group")
            .generic_string();
    Check(std::filesystem::exists(quantized_dir + "/testcase.json"),
          "committed int8-per-group 0.4 fixture is present");
    // Absolute paths are taken as given; the plain item stays relative.
    const std::string manifest_path = dir + "/batch.json";
    WriteBatchManifest(manifest_path,
                       {{"quantized", quantized_dir, dir + "/out-quantized"},
                        {"plain", "case-plain", "out-plain"}});

    int calls = 0;
    const inferref::DebugInvoke invoke =
        [&](const std::string &region_name,
            const std::map<std::string, inferref::IRTensor> &inputs)
    {
        ++calls;
        return KVInvoke(region_name, inputs);
    };
    const int code = inferref::RunBridgeBatch(manifest_path, invoke);
    Check(code == inferref::kBridgeError,
          "a refused quantized item fails the batch");
    Check(calls == 1,
          "the quantized item never reaches the legacy callback");
    Check(!std::filesystem::exists(dir + "/out-quantized/manifest.json"),
          "a refused quantized item writes no manifest.json");
    Check(std::filesystem::exists(dir + "/out-plain/manifest.json"),
          "the batch continues after a refused quantized item");

    const inferref::json::Value results = LoadBatchResults(dir);
    const inferref::json::Array &recorded = results.At("items").array;
    Check(recorded[0].At("status").string == "failed" &&
              static_cast<int>(recorded[0].At("exit_code").number) ==
                  inferref::kBridgeUnsupportedQuantization,
          "the quantized item records kBridgeUnsupportedQuantization (4)");
    Check(recorded[0].At("error").IsString() &&
              recorded[0].At("error").string.find("unsupported_quantization") !=
                  std::string::npos,
          "the quantized item records the legacy diagnostic");
    Check(recorded[1].At("status").string == "passed",
          "the later plain item passes");
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

    if (argc > 2 && std::string(argv[1]) == "--batch-manifest")
    {
        // External batch mode: run every manifest item through the KV stub in
        // this one process.
        std::printf("bridge batch: %s\n", argv[2]);
        return inferref::RunBridgeBatch(argv[2], KVInvoke);
    }

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
        TestBatchManifestValidation(base);
        TestBatchExecution(base);
        TestBatchAllPass(base);
        if (goldens_dir.empty())
            std::printf("quantization fixtures not provided; skipping\n");
        else
        {
            TestQuantizedDispatch(goldens_dir);
            TestBatchQuantizationGuard(base, goldens_dir);
        }
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
