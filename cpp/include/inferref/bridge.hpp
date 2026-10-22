// Generic runtime bridge (InferRef 0.8 Adapter DX, section 7).
//
// One bridge binary dispatches any region into an existing runtime through a
// DebugInvoke callback: region name + named inputs -> named outputs.  The
// bridge contains no model knowledge; the runtime registers the callback.
//
// Testcase format 0.4 carries explicit quantization (InferRef 0.12 section
// 5.4).  A runtime that needs the descriptors, parameters, or dequantized
// inputs registers the Testcase-aware DebugInvokeEx callback.  The legacy
// two-argument DebugInvoke cannot see them, so RunBridge refuses quantized
// inputs with kBridgeUnsupportedQuantization instead of silently dropping
// scales.
//
// Batch execution (InferRef 0.12 section 7.3) runs an ordered list of
// testcases through one callback in one process, so a runtime loads a
// multi-hundred-megabyte engine once instead of once per testcase:
//
//     {"format": "inferref-batch", "format_version": "1.0",
//      "items": [{"id": "step-0",
//                 "testcase_dir": "cases/step-0",
//                 "output_dir": "out/step-0"}]}
//
// Relative item paths resolve against the manifest's directory.  Relative
// paths with ".." segments are rejected; absolute paths are taken as given
// (the manifest author already chooses where every item runs).
//
// Header-only and dependency-free, layered on testcase.hpp.

#ifndef INFERREF_BRIDGE_HPP
#define INFERREF_BRIDGE_HPP

#include <inferref/testcase.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inferref
{

// Runtime callback: region name + named inputs -> named outputs.  Outputs are
// keyed by role name, matching Engine Adapter v0.2's exact output-role
// contract.  It sees no testcase metadata, so RunBridge only accepts
// unquantized testcases on this path.
using DebugInvoke = std::function<std::map<std::string, inferref::IRTensor>(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)>;

// Testcase-aware runtime callback: the loaded Testcase is passed alongside the
// resolved region key and named inputs, so a quantized runtime can call
// HasInputQuantization(), InputQuantization(), QuantizationParameter(), and
// DequantizeInputToFloat32() while it dispatches.
using DebugInvokeEx = std::function<std::map<std::string, inferref::IRTensor>(
    const inferref::Testcase &testcase,
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)>;

// RunBridge exit codes.
inline constexpr int kBridgeOk = 0;
inline constexpr int kBridgeError = 1;
inline constexpr int kBridgeUsage = 2;
inline constexpr int kBridgeMissingRegion = 3;
inline constexpr int kBridgeUnsupportedQuantization = 4;

// Batch manifest wire format (InferRef 0.12 section 7.3).
inline constexpr const char *kBatchFormat = "inferref-batch";
inline constexpr const char *kBatchFormatVersion = "1.0";

// RunBridgeBatch writes this file next to the manifest it consumed.
inline constexpr const char *kBatchResultsFormat = "inferref-batch-results";
inline constexpr const char *kBatchResultsFileName = "batch-results.json";

class BatchManifestError : public std::runtime_error
{
public:
    explicit BatchManifestError(const std::string &what)
        : std::runtime_error(what)
    {
    }
};

// One ordered batch entry with its paths already resolved.
struct BatchItem
{
    std::string id;
    std::string testcase_dir;
    std::string output_dir;
};

namespace detail
{

inline std::string JsonString(const std::string &value)
{
    std::string out = "\"";
    for (const char ch : value)
    {
        switch (ch)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20)
            {
                const char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[(static_cast<unsigned char>(ch) >> 4) & 0x0F];
                out += hex[static_cast<unsigned char>(ch) & 0x0F];
            }
            else
            {
                out += ch;
            }
            break;
        }
    }
    out += "\"";
    return out;
}

inline void WriteFile(const std::string &path, const std::string &text)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open " + path + " for writing");
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
        throw std::runtime_error("cannot write " + path);
}

inline std::string ManifestDirectory(const std::string &manifest_path)
{
    const std::filesystem::path path(manifest_path);
    if (path.has_parent_path() && !path.parent_path().empty())
        return path.parent_path().generic_string();
    return ".";
}

inline std::string ResolveItemPath(const std::string &manifest_dir,
                                   const std::string &value,
                                   const std::string &context)
{
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.has_root_name())
        return path.lexically_normal().generic_string();
    for (const std::filesystem::path &part : path)
    {
        if (part == "..")
            throw BatchManifestError(context +
                                     " must not contain '..' path segments");
    }
    return (std::filesystem::path(manifest_dir) / path)
        .lexically_normal()
        .generic_string();
}

inline std::string RequiredItemString(const json::Value &entry,
                                      const std::string &field,
                                      const std::string &context)
{
    if (!entry.Has(field))
        throw BatchManifestError(context + " is missing '" + field + "'");
    const json::Value &value = entry.At(field);
    if (!value.IsString() || value.string.empty())
    {
        throw BatchManifestError(context + " requires a non-empty string '" +
                                 field + "'");
    }
    return value.string;
}

inline std::string BatchResultsPath(const std::string &manifest_path)
{
    const std::filesystem::path directory(ManifestDirectory(manifest_path));
    return (directory / kBatchResultsFileName).generic_string();
}

} // namespace detail

// Load and validate a batch manifest.  Items keep manifest order; relative
// testcase/output paths are resolved against the manifest's directory.
// Throws BatchManifestError for malformed JSON, a wrong format or version, an
// empty item list, a missing field, a duplicate id, or an unsafe path.
inline std::vector<BatchItem> LoadBatchManifest(const std::string &manifest_path)
{
    std::ifstream stream(manifest_path, std::ios::binary);
    if (!stream)
        throw BatchManifestError("cannot open batch manifest " + manifest_path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();

    json::Value root;
    try
    {
        root = json::Parse(buffer.str());
    }
    catch (const json::ParseError &error)
    {
        throw BatchManifestError("batch manifest is not valid JSON: " +
                                 std::string(error.what()));
    }
    if (!root.IsObject())
        throw BatchManifestError("batch manifest root must be a JSON object");
    if (!root.Has("format") || !root.At("format").IsString() ||
        root.At("format").string != kBatchFormat)
    {
        throw BatchManifestError(
            "batch manifest format must be 'inferref-batch'");
    }
    if (!root.Has("format_version") || !root.At("format_version").IsString() ||
        root.At("format_version").string != kBatchFormatVersion)
    {
        throw BatchManifestError(
            "unsupported batch manifest format_version; this build reads '1.0'");
    }
    if (!root.Has("items") || !root.At("items").IsArray())
        throw BatchManifestError("batch manifest 'items' must be an array");
    const json::Array &entries = root.At("items").array;
    if (entries.empty())
        throw BatchManifestError("batch manifest 'items' must not be empty");

    const std::string manifest_dir = detail::ManifestDirectory(manifest_path);
    std::vector<BatchItem> items;
    items.reserve(entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        const json::Value &entry = entries[index];
        const std::string context =
            "batch manifest item " + std::to_string(index);
        if (!entry.IsObject())
            throw BatchManifestError(context + " must be a JSON object");
        BatchItem item;
        item.id = detail::RequiredItemString(entry, "id", context);
        item.testcase_dir = detail::ResolveItemPath(
            manifest_dir, detail::RequiredItemString(entry, "testcase_dir", context),
            context + " testcase_dir");
        item.output_dir = detail::ResolveItemPath(
            manifest_dir, detail::RequiredItemString(entry, "output_dir", context),
            context + " output_dir");
        for (const BatchItem &existing : items)
        {
            if (existing.id == item.id)
                throw BatchManifestError("duplicate batch item id '" + item.id +
                                         "'");
        }
        items.push_back(std::move(item));
    }
    return items;
}

namespace detail
{

// Dispatch an already-loaded testcase and publish its outputs.  `caller_label`
// names the callback in manifest errors, preserving each path's diagnostics.
// A bridge-detected failure is returned with its diagnostic so batch execution
// can record it; callback exceptions propagate.
struct DispatchOutcome
{
    int exit_code = kBridgeOk;
    std::string error;
};

inline DispatchOutcome DispatchOnce(Testcase &testcase,
                                    const std::string &caller_label,
                                    const DebugInvoke &invoke)
{
    const std::string region = testcase.RegionName();
    if (region.empty())
    {
        return {kBridgeMissingRegion,
                "missing_region: testcase has no origin.region, origin.contract, "
                "or contracts[0]"};
    }
    const std::map<std::string, IRTensor> outputs =
        invoke(region, testcase.Inputs());
    testcase.WriteOutputs(outputs, caller_label);
    testcase.Finish();
    return {};
}

// Diagnostic for the legacy callback path; empty when the testcase carries no
// quantized input role.
inline std::vector<std::string> QuantizedInputRoles(const Testcase &testcase)
{
    std::vector<std::string> roles;
    for (const std::string &name : testcase.InputNames())
    {
        if (testcase.HasInputQuantization(name))
            roles.push_back(name);
    }
    return roles;
}

inline std::string QuoteRoles(const std::vector<std::string> &roles)
{
    std::string quoted;
    for (const std::string &role : roles)
    {
        if (!quoted.empty())
            quoted += ", ";
        quoted += "'" + role + "'";
    }
    return quoted;
}

inline std::string LegacyQuantizationDiagnostic(const Testcase &testcase)
{
    const std::vector<std::string> quantized = QuantizedInputRoles(testcase);
    if (quantized.empty())
        return {};
    return "unsupported_quantization: testcase format_version " +
           testcase.FormatVersion() + " quantizes input role(s) " +
           QuoteRoles(quantized) +
           "; the legacy DebugInvoke callback cannot see scales. Register a "
           "DebugInvokeEx callback and query testcase.InputQuantization() / "
           "QuantizationParameter() / DequantizeInputToFloat32() instead.";
}

// One item's recorded outcome in batch-results.json.
struct BatchItemResult
{
    std::string id;
    std::string status;
    int exit_code = kBridgeOk;
    std::string testcase_dir;
    std::string output_dir;
    std::string error;
};

// Invoke one loaded testcase for a batch item: code plus a diagnostic message
// when the bridge itself refuses the item.
using BatchDispatch = std::function<int(Testcase &, std::string &)>;

inline BatchItemResult RunBatchItem(const BatchItem &item,
                                    const BatchDispatch &dispatch)
{
    BatchItemResult result;
    result.id = item.id;
    result.testcase_dir = item.testcase_dir;
    result.output_dir = item.output_dir;
    try
    {
        Testcase testcase = Testcase::Load(item.testcase_dir);
        testcase.SetOutputDir(item.output_dir);
        std::string error;
        result.exit_code = dispatch(testcase, error);
        result.error = error;
    }
    catch (const std::exception &error)
    {
        result.exit_code = kBridgeError;
        result.error = error.what();
    }
    catch (...)
    {
        result.exit_code = kBridgeError;
        result.error = "unknown non-standard exception";
    }
    result.status = result.exit_code == kBridgeOk ? "passed" : "failed";
    return result;
}

inline std::string BatchResultsJson(const std::vector<BatchItemResult> &results,
                                    int passed, int failed)
{
    std::string text = "{\"format\":";
    text += JsonString(kBatchResultsFormat);
    text += ",\"format_version\":";
    text += JsonString(kBatchFormatVersion);
    text += ",\"items\":[";
    for (std::size_t index = 0; index < results.size(); ++index)
    {
        const BatchItemResult &result = results[index];
        if (index != 0)
            text += ",";
        text += "{\"id\":";
        text += JsonString(result.id);
        text += ",\"status\":";
        text += JsonString(result.status);
        text += ",\"exit_code\":";
        text += std::to_string(result.exit_code);
        text += ",\"testcase_dir\":";
        text += JsonString(result.testcase_dir);
        text += ",\"output_dir\":";
        text += JsonString(result.output_dir);
        text += ",\"error\":";
        text += result.error.empty() ? std::string("null") : JsonString(result.error);
        text += "}";
    }
    text += "],\"counts\":{\"total\":";
    text += std::to_string(results.size());
    text += ",\"passed\":";
    text += std::to_string(passed);
    text += ",\"failed\":";
    text += std::to_string(failed);
    text += "}}";
    return text;
}

// Run every manifest item in order in this process.  A failing item is
// recorded and the next item still runs.  batch-results.json is written next
// to the manifest; the returned exit code is kBridgeOk only when every item
// passed, and kBridgeError when the manifest is invalid or any item failed.
inline int RunBatch(const std::string &manifest_path,
                    const std::string &program_name,
                    const BatchDispatch &dispatch)
{
    std::vector<BatchItem> items;
    try
    {
        items = LoadBatchManifest(manifest_path);
    }
    catch (const std::exception &error)
    {
        std::cerr << program_name << ": invalid batch manifest " << manifest_path
                  << ": " << error.what() << "\n";
        return kBridgeError;
    }

    std::vector<BatchItemResult> results;
    results.reserve(items.size());
    int passed = 0;
    int failed = 0;
    for (const BatchItem &item : items)
    {
        BatchItemResult result = RunBatchItem(item, dispatch);
        if (result.exit_code == kBridgeOk)
        {
            ++passed;
        }
        else
        {
            ++failed;
            std::cerr << program_name << ": batch item '" << result.id << "': "
                      << result.error << "\n";
        }
        results.push_back(std::move(result));
    }

    try
    {
        WriteFile(BatchResultsPath(manifest_path),
                  BatchResultsJson(results, passed, failed));
    }
    catch (const std::exception &error)
    {
        std::cerr << program_name << ": cannot write batch results: "
                  << error.what() << "\n";
        return kBridgeError;
    }
    return failed == 0 ? kBridgeOk : kBridgeError;
}

} // namespace detail

// Load one testcase, dispatch through the Testcase-aware `invoke`, write every
// declared output by role name into `output_dir`, and publish manifest.json
// once.  The callback receives the loaded Testcase so quantized runtimes can
// read its descriptors and parameter tensors.
inline int RunBridge(const std::string &testcase_dir,
                     const std::string &output_dir,
                     const DebugInvokeEx &invoke,
                     const std::string &program_name = "inferref-bridge")
{
    try
    {
        Testcase testcase = Testcase::Load(testcase_dir);
        testcase.SetOutputDir(output_dir);
        const detail::DispatchOutcome outcome = detail::DispatchOnce(
            testcase, "DebugInvokeEx",
            [&](const std::string &region,
                const std::map<std::string, IRTensor> &inputs)
            { return invoke(testcase, region, inputs); });
        if (!outcome.error.empty())
            std::cerr << program_name << ": " << outcome.error << "\n";
        return outcome.exit_code;
    }
    catch (const std::exception &error)
    {
        std::cerr << program_name << ": " << error.what() << "\n";
        return kBridgeError;
    }
    catch (...)
    {
        std::cerr << program_name << ": unknown non-standard exception\n";
        return kBridgeError;
    }
}

// Legacy callback path.  Unquantized testcases dispatch exactly as before.  A
// quantized testcase is refused before the callback runs because the
// two-argument callback cannot see the descriptor or parameter tensors and
// would silently ignore scales.
inline int RunBridge(const std::string &testcase_dir,
                     const std::string &output_dir,
                     const DebugInvoke &invoke,
                     const std::string &program_name = "inferref-bridge")
{
    try
    {
        Testcase testcase = Testcase::Load(testcase_dir);
        const std::string diagnostic =
            detail::LegacyQuantizationDiagnostic(testcase);
        if (!diagnostic.empty())
        {
            std::cerr << program_name << ": " << diagnostic << "\n";
            return kBridgeUnsupportedQuantization;
        }
        testcase.SetOutputDir(output_dir);
        const detail::DispatchOutcome outcome =
            detail::DispatchOnce(testcase, "DebugInvoke", invoke);
        if (!outcome.error.empty())
            std::cerr << program_name << ": " << outcome.error << "\n";
        return outcome.exit_code;
    }
    catch (const std::exception &error)
    {
        std::cerr << program_name << ": " << error.what() << "\n";
        return kBridgeError;
    }
    catch (...)
    {
        std::cerr << program_name << ": unknown non-standard exception\n";
        return kBridgeError;
    }
}

// Run a batch manifest through the Testcase-aware callback.  The callback is
// invoked once per item in manifest order inside this process; every item
// writes its own outputs and manifest.json.  A failing item does not stop the
// batch.  Returns kBridgeOk when every item passed, kBridgeError otherwise.
inline int RunBridgeBatch(const std::string &manifest_path,
                          const DebugInvokeEx &invoke,
                          const std::string &program_name = "inferref-bridge")
{
    return detail::RunBatch(
        manifest_path, program_name,
        [&](Testcase &testcase, std::string &error)
        {
            const detail::DispatchOutcome outcome = detail::DispatchOnce(
                testcase, "DebugInvokeEx",
                [&](const std::string &region,
                    const std::map<std::string, IRTensor> &inputs)
                { return invoke(testcase, region, inputs); });
            error = outcome.error;
            return outcome.exit_code;
        });
}

// Run a batch manifest through the legacy callback.  Each item is guarded like
// the single-testcase legacy path: a quantized testcase fails that item with
// kBridgeUnsupportedQuantization and the batch moves on.
inline int RunBridgeBatch(const std::string &manifest_path,
                          const DebugInvoke &invoke,
                          const std::string &program_name = "inferref-bridge")
{
    return detail::RunBatch(
        manifest_path, program_name,
        [&](Testcase &testcase, std::string &error)
        {
            const std::string diagnostic =
                detail::LegacyQuantizationDiagnostic(testcase);
            if (!diagnostic.empty())
            {
                error = diagnostic;
                return kBridgeUnsupportedQuantization;
            }
            const detail::DispatchOutcome outcome =
                detail::DispatchOnce(testcase, "DebugInvoke", invoke);
            error = outcome.error;
            return outcome.exit_code;
        });
}

} // namespace inferref

#endif // INFERREF_BRIDGE_HPP
