// High-level testcase helper for native engines (InferRef 0.8 Adapter DX).
//
// A runtime should not need to know about testcase.json, .irtensor headers,
// or the engine output manifest.  This header owns those details: load the
// manifest once, read named input tensors, write named output tensors, and
// emit the optional output manifest.
//
// Header-only and dependency-free, like irtensor.hpp and json.hpp: it uses
// only the C++ standard library and those two InferRef headers.
//
// Usage:
//
//     auto testcase = inferref::Testcase::Load("repro/rope");
//     auto inputs = testcase.Inputs();
//     auto outputs = RunYourEngine(testcase.RegionName(), inputs);
//     for (const auto &name : testcase.OutputNames())
//         testcase.WriteOutput(name, outputs.at(name));
//     testcase.Finish();

#ifndef INFERREF_TESTCASE_HPP
#define INFERREF_TESTCASE_HPP

#include <inferref/irtensor.hpp>
#include <inferref/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferref
{

class TestcaseError : public std::runtime_error
{
public:
    explicit TestcaseError(const std::string &what) : std::runtime_error(what) {}
};

// One loaded standalone testcase: named input/output roles plus the resolved
// region dispatch key.  Inputs and outputs are always addressed by role name,
// never by vector position, matching Engine Adapter v0.2's output-role
// contract.
class Testcase
{
public:
    static Testcase Load(const std::string &testcase_dir)
    {
        const std::string manifest_path = testcase_dir + "/testcase.json";
        const std::string text = ReadFile(manifest_path);
        const json::Value root = json::Parse(text);
        if (!root.IsObject())
            throw TestcaseError("testcase manifest root must be a JSON object");
        if (!root.Has("format") || !root.At("format").IsString() ||
            root.At("format").string != "inferref-testcase")
        {
            throw TestcaseError(
                "not an InferRef testcase: format must be 'inferref-testcase'");
        }
        if (!root.Has("format_version") || !root.At("format_version").IsString() ||
            (root.At("format_version").string != "0.1" &&
             root.At("format_version").string != "0.2"))
        {
            throw TestcaseError(
                "unsupported testcase format_version; this build reads 0.1 and 0.2");
        }

        Testcase result;
        result.dir_ = testcase_dir;
        result.region_name_ = ResolveRegionName(root);
        result.input_names_ = ParseRoles(root, "inputs");
        result.output_names_ = ParseRoles(root, "outputs");
        result.input_payloads_ = ParsePayloads(root, "inputs", result.input_names_);
        return result;
    }

    // Dispatch key: origin.region, then origin.contract / contracts[0], then
    // empty.  The runtime bridge treats an empty key as missing_region.
    std::string RegionName() const { return region_name_; }

    std::vector<std::string> InputNames() const { return input_names_; }
    std::vector<std::string> OutputNames() const { return output_names_; }

    // Load one input by role name.  Throws TestcaseError for an unknown role
    // or a role without a runnable payload.
    IRTensor Input(const std::string &name) const
    {
        const std::string payload =
            PayloadFor(input_payloads_, input_names_, name, "input");
        return ReadIRTensor(dir_ + "/" + payload);
    }

    // Load every input, keyed by role name.
    std::map<std::string, IRTensor> Inputs() const
    {
        std::map<std::string, IRTensor> result;
        for (const std::string &name : input_names_)
        {
            result.emplace(name, Input(name));
        }
        return result;
    }

    // Redirect output tensors and manifest.json to an explicit output
    // directory (Engine Adapter v0.2's {output} directory), creating it if
    // needed.  Defaults to the testcase directory; adapters and the runtime
    // bridge always set it.
    void SetOutputDir(const std::string &output_dir)
    {
        std::filesystem::create_directories(output_dir);
        output_dir_ = output_dir;
    }

    // Write one output file as "<name>.irtensor" in the output directory.
    // Does not touch manifest.json; Finish() publishes the output manifest
    // once.
    void WriteOutput(const std::string &name, const IRTensor &tensor)
    {
        if (finished_)
            throw TestcaseError("cannot write output after Finish()");
        if (std::find(output_names_.begin(), output_names_.end(), name) ==
            output_names_.end())
        {
            throw TestcaseError("unknown output role '" + name + "'");
        }
        const std::string relative = name + ".irtensor";
        WriteIRTensor(OutputDir() + "/" + relative, tensor);
        written_outputs_[name] = relative;
    }

    // Write manifest.json with every output written so far, in declared output
    // order.  Idempotent: later calls are no-ops.  manifest.json is optional
    // for `inferref compare`; it exists for runtimes that want explicit output
    // metadata.
    void Finish()
    {
        if (finished_)
            return;
        std::string manifest = "{\"outputs\":[";
        bool first = true;
        for (const std::string &name : output_names_)
        {
            const auto written = written_outputs_.find(name);
            if (written == written_outputs_.end())
                continue;
            if (!first)
                manifest += ",";
            first = false;
            manifest += "{\"name\":";
            manifest += JsonEscape(name);
            manifest += ",\"payload\":";
            manifest += JsonEscape(written->second);
            manifest += "}";
        }
        manifest += "]}";
        WriteFile(OutputDir() + "/manifest.json", manifest);
        finished_ = true;
    }

    const std::string &OutputDir() const
    {
        return output_dir_.empty() ? dir_ : output_dir_;
    }

private:
    std::string dir_;
    std::string output_dir_;
    std::string region_name_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::map<std::string, std::string> input_payloads_;
    std::map<std::string, std::string> written_outputs_;
    bool finished_ = false;

    static std::string ReadFile(const std::string &path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw TestcaseError("cannot open " + path);
        std::string text;
        char buffer[4096];
        while (stream)
        {
            stream.read(buffer, sizeof(buffer));
            text.append(buffer, static_cast<std::size_t>(stream.gcount()));
        }
        if (stream.bad())
            throw TestcaseError("cannot read " + path);
        return text;
    }

    static void WriteFile(const std::string &path, const std::string &text)
    {
        std::ofstream stream(path, std::ios::binary);
        if (!stream)
            throw TestcaseError("cannot open " + path + " for writing");
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!stream)
            throw TestcaseError("cannot write " + path);
    }

    // Reject payload paths that could escape the testcase directory: absolute
    // paths, drive letters, backslashes, and ".." segments.
    static bool IsSafeRelativePath(const std::string &payload)
    {
        if (payload.empty() || payload[0] == '/' || payload[0] == '\\')
            return false;
        for (std::size_t i = 0; i + 1 < payload.size(); ++i)
        {
            if (payload[i] == '.' && payload[i + 1] == '.')
                return false;
        }
        if (payload.find('\\') != std::string::npos)
            return false;
        if (payload.find(':') != std::string::npos)
            return false;
        return true;
    }

    static std::vector<std::string> ParseRoles(const json::Value &root,
                                               const std::string &field)
    {
        if (!root.Has(field) || !root.At(field).IsArray())
            throw TestcaseError("testcase manifest '" + field + "' must be an array");
        std::vector<std::string> names;
        for (const json::Value &entry : root.At(field).array)
        {
            if (!entry.IsObject())
                throw TestcaseError("testcase manifest '" + field +
                                    "' entries must be objects");
            if (!entry.Has("name") || !entry.At("name").IsString() ||
                entry.At("name").string.empty())
            {
                throw TestcaseError("testcase manifest '" + field +
                                    "' entry requires a non-empty name");
            }
            names.push_back(entry.At("name").string);
        }
        if (names.empty())
            throw TestcaseError("testcase manifest '" + field + "' must not be empty");
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            for (std::size_t j = i + 1; j < names.size(); ++j)
            {
                if (names[i] == names[j])
                {
                    throw TestcaseError("duplicate '" + field + "' role name '" +
                                        names[i] + "'");
                }
            }
        }
        return names;
    }

    static std::map<std::string, std::string> ParsePayloads(
        const json::Value &root, const std::string &field,
        const std::vector<std::string> &names)
    {
        std::map<std::string, std::string> payloads;
        const json::Value &array = root.At(field);
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            const json::Value &entry = array.array[i];
            const std::string name = names[i];
            if (!entry.Has("payload") || entry.At("payload").IsNull())
                continue;  // role declared without a runnable payload
            if (!entry.At("payload").IsString())
                throw TestcaseError("testcase manifest '" + field +
                                    "' payload for '" + name + "' must be a string");
            const std::string payload = entry.At("payload").string;
            if (!IsSafeRelativePath(payload))
                throw TestcaseError("testcase manifest '" + field +
                                    "' payload for '" + name +
                                    "' is not a safe relative path");
            payloads[name] = payload;
        }
        return payloads;
    }

    static std::string PayloadFor(
        const std::map<std::string, std::string> &payloads,
        const std::vector<std::string> &names,
        const std::string &name,
        const std::string &boundary)
    {
        const auto found = payloads.find(name);
        if (found == payloads.end())
        {
            if (std::find(names.begin(), names.end(), name) != names.end())
            {
                throw TestcaseError(boundary + " role '" + name +
                                    "' has no runnable payload");
            }
            throw TestcaseError("unknown " + boundary + " role '" + name + "'");
        }
        return found->second;
    }

    static std::string ResolveRegionName(const json::Value &root)
    {
        if (root.Has("origin") && root.At("origin").IsObject())
        {
            const json::Value &origin = root.At("origin");
            if (origin.Has("region") && origin.At("region").IsString() &&
                !origin.At("region").string.empty())
            {
                return origin.At("region").string;
            }
            if (origin.Has("contract") && origin.At("contract").IsString() &&
                !origin.At("contract").string.empty())
            {
                return origin.At("contract").string;
            }
        }
        const std::vector<std::string> contracts = json::DeclaredContracts(root);
        if (!contracts.empty())
            return contracts.front();
        return "";
    }

    static std::string JsonEscape(const std::string &value)
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
};

} // namespace inferref

#endif // INFERREF_TESTCASE_HPP
