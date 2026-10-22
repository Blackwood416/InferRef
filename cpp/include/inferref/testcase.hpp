// High-level testcase helper for native engines (InferRef 0.8 Adapter DX).
//
// A runtime should not need to know about testcase.json, .irtensor headers,
// or the engine output manifest.  This header owns those details: load the
// manifest once, read named input tensors, write named output tensors, and
// emit the optional output manifest.
//
// Testcase format 0.4 adds explicit quantization: a value carries a
// QuantizationSpec and references immutable parameter tensors (scale,
// zero point, codebook) by value ID.  Load() validates that metadata
// fail-closed before any engine kernel runs, and exposes the descriptors plus
// a float32 reference dequantizer for quantized input roles.
//
// Header-only and dependency-free, like irtensor.hpp, json.hpp, and
// quantization.hpp: it uses only the C++ standard library and those headers.
//
// Usage:
//
//     auto testcase = inferref::Testcase::Load("repro/rope");
//     auto inputs = testcase.Inputs();
//     auto outputs = RunYourEngine(testcase.RegionName(), inputs);
//     testcase.WriteOutputs(outputs, "RunYourEngine");
//     testcase.Finish();
//
//     if (testcase.HasInputQuantization("weight"))
//     {
//         const auto &spec = testcase.InputQuantization("weight");
//         auto scale = testcase.QuantizationParameter(spec.scale_value_id);
//         auto logical = testcase.DequantizeInputToFloat32("weight");
//     }

#ifndef INFERREF_TESTCASE_HPP
#define INFERREF_TESTCASE_HPP

#include <inferref/irtensor.hpp>
#include <inferref/json.hpp>
#include <inferref/quantization.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
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
             root.At("format_version").string != "0.2" &&
             root.At("format_version").string != "0.3" &&
             root.At("format_version").string != "0.4"))
        {
            throw TestcaseError(
                "unsupported testcase format_version; this build reads 0.1, 0.2, "
                "0.3, and 0.4");
        }
        const std::string version = root.At("format_version").string;
        const bool quantized_format = version == "0.4";
        RejectQuantizationMetadataBefore04(root, quantized_format);

        Testcase result;
        result.dir_ = testcase_dir;
        result.format_version_ = version;
        result.region_name_ = ResolveRegionName(root);
        result.input_names_ = ParseRoles(root, "inputs");
        result.output_names_ = ParseRoles(root, "outputs");
        result.input_payloads_ = ParsePayloads(root, "inputs", result.input_names_);
        result.output_payloads_ =
            ParsePayloads(root, "outputs", result.output_names_);
        if (quantized_format)
            result.LoadQuantization(root);
        return result;
    }

    // Dispatch key: origin.region, then origin.contract / contracts[0], then
    // empty.  The runtime bridge treats an empty key as missing_region.
    std::string RegionName() const { return region_name_; }

    std::string FormatVersion() const { return format_version_; }

    std::vector<std::string> InputNames() const { return input_names_; }
    std::vector<std::string> OutputNames() const { return output_names_; }

    // Load one input by role name.  Throws TestcaseError for an unknown role
    // or a role without a runnable payload.  A quantized input was already
    // loaded and validated by Load(); this returns the cached copy.
    IRTensor Input(const std::string &name) const
    {
        const auto cached = quantized_inputs_.find(name);
        if (cached != quantized_inputs_.end())
            return cached->second;
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

    // True when this input role carries an explicit quantization descriptor.
    bool HasInputQuantization(const std::string &name) const
    {
        RequireInputRole(name);
        return input_quantizations_.count(name) != 0;
    }

    // Descriptor for a quantized input role.  Throws for an unknown or
    // unquantized role.
    const QuantizationSpec &InputQuantization(const std::string &name) const
    {
        RequireInputRole(name);
        const auto found = input_quantizations_.find(name);
        if (found == input_quantizations_.end())
            throw TestcaseError("input role '" + name + "' is not quantized");
        return found->second;
    }

    // Referenced quantization parameter IDs in ascending order.
    std::vector<std::int64_t> QuantizationParameterIds() const
    {
        std::vector<std::int64_t> ids;
        ids.reserve(quantization_parameters_.size());
        for (const auto &entry : quantization_parameters_)
            ids.push_back(entry.first);
        return ids;
    }

    // One parameter tensor by value ID.  Throws for an unreferenced ID.
    IRTensor QuantizationParameter(std::int64_t value_id) const
    {
        const auto found = quantization_parameters_.find(value_id);
        if (found == quantization_parameters_.end())
        {
            throw TestcaseError("no quantization parameter with value ID " +
                                std::to_string(value_id));
        }
        return found->second;
    }

    std::map<std::int64_t, IRTensor> QuantizationParameters() const
    {
        return quantization_parameters_;
    }

    // Reconstruct the logical float32 values of a quantized input role.
    // Bit-identical to inferref/quantization/reference.py and nf4.py.
    std::vector<float> DequantizeInputToFloat32(const std::string &name) const
    {
        const QuantizationSpec &spec = InputQuantization(name);
        const IRTensor codes = Input(name);
        try
        {
            return DequantizeQuantizationFloat32(spec, codes,
                                                 quantization_parameters_);
        }
        catch (const QuantizationError &error)
        {
            throw TestcaseError("input role '" + name +
                                "' dequantization failed: " + error.what());
        }
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

    // Validate all declared output roles are present and no undeclared roles
    // are returned before writing any files to disk. Only after all checks pass,
    // write each output IRTensor to disk and record it for Finish().
    void WriteOutputs(
        const std::map<std::string, IRTensor> &outputs,
        const std::string &caller_label = "engine")
    {
        if (finished_)
            throw TestcaseError("cannot write output after Finish()");
        for (const std::string &name : output_names_)
        {
            if (outputs.find(name) == outputs.end())
            {
                throw TestcaseError(caller_label + " did not return output role '" +
                                    name + "'");
            }
        }
        for (const auto &entry : outputs)
        {
            if (std::find(output_names_.begin(), output_names_.end(),
                          entry.first) == output_names_.end())
            {
                throw TestcaseError(caller_label +
                                    " returned undeclared output role '" +
                                    entry.first + "'");
            }
        }
        for (const std::string &name : output_names_)
        {
            const std::string relative = name + ".irtensor";
            WriteIRTensor(OutputDir() + "/" + relative, outputs.at(name));
            written_outputs_[name] = relative;
        }
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
    // One `values` registry record: enough metadata to validate descriptors
    // and parameter dependencies without loading every value payload.
    struct ValueRecord
    {
        std::string dtype;
        std::vector<std::int64_t> shape;
        bool has_quantization = false;
        QuantizationSpec spec;
    };

    std::string dir_;
    std::string output_dir_;
    std::string format_version_;
    std::string region_name_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::map<std::string, std::string> input_payloads_;
    std::map<std::string, std::string> output_payloads_;
    std::map<std::string, std::string> written_outputs_;
    std::map<std::string, QuantizationSpec> input_quantizations_;
    std::map<std::int64_t, IRTensor> quantization_parameters_;
    std::map<std::string, IRTensor> quantized_inputs_;
    bool finished_ = false;

    // Formats before 0.4 cannot carry embedded quantization.  Rejecting here
    // keeps an explicit quantized tensor from being silently read as an
    // ordinary integer tensor.
    static void RejectQuantizationMetadataBefore04(const json::Value &root,
                                                   bool quantized_format)
    {
        if (quantized_format)
            return;
        if (root.Has("quantization_parameters"))
        {
            throw TestcaseError(
                "quantization metadata requires testcase format_version '0.4'");
        }
        for (const char *field : {"inputs", "outputs", "values"})
        {
            if (!root.Has(field) || !root.At(field).IsArray())
                continue;
            for (const json::Value &entry : root.At(field).array)
            {
                if (entry.IsObject() && entry.Has("quantization"))
                {
                    throw TestcaseError(
                        "quantization metadata requires testcase format_version '0.4'");
                }
            }
        }
    }

    // Parse and cross-check every quantization descriptor, parameter payload,
    // and dependency reference: the structural and header rules of the Python
    // validate_testcase.  Any inconsistency fails the whole Load().  Numeric
    // validity (for example a strictly positive scale) is enforced later, by
    // dequantization, matching the Python reference.
    void LoadQuantization(const json::Value &root)
    {
        try
        {
            LoadQuantizationChecked(root);
        }
        catch (const QuantizationError &error)
        {
            throw TestcaseError(std::string("invalid quantization metadata: ") +
                                error.what());
        }
    }

    void LoadQuantizationChecked(const json::Value &root)
    {
        const std::map<std::int64_t, ValueRecord> values =
            ParseValueRegistry(root);
        const std::map<std::string, QuantizationSpec> role_specs =
            ParseRoleQuantizations(root, values);

        // Validate the whole registry, like the Python reference: every
        // quantized value's dependencies must be present and unquantized.
        std::vector<std::int64_t> parameter_ids;
        for (const auto &entry : values)
        {
            if (!entry.second.has_quantization)
                continue;
            for (const std::int64_t dependency : entry.second.spec.Dependencies())
            {
                const auto parameter = values.find(dependency);
                if (parameter == values.end())
                {
                    throw QuantizationError(
                        "value " + std::to_string(entry.first) +
                        " references missing parameter value ID " +
                        std::to_string(dependency));
                }
                if (parameter->second.has_quantization)
                {
                    throw QuantizationError(
                        "nested quantized parameters are unsupported at value ID " +
                        std::to_string(dependency));
                }
                parameter_ids.push_back(dependency);
            }
        }
        std::sort(parameter_ids.begin(), parameter_ids.end());
        parameter_ids.erase(
            std::unique(parameter_ids.begin(), parameter_ids.end()),
            parameter_ids.end());

        const std::map<std::int64_t, IRTensor> parameters =
            ParseQuantizationParameters(root, values, parameter_ids);

        // The referenced parameter set and the top-level collection must
        // agree exactly: no missing, extra, or duplicate entries.
        for (const auto &entry : values)
        {
            if (!entry.second.has_quantization)
                continue;
            const QuantizationSpec &spec = entry.second.spec;
            for (const std::int64_t dependency : spec.Dependencies())
            {
                const ValueRecord &record = values.at(dependency);
                const DataType expected_dtype = dependency == spec.scale_value_id
                    ? DataType::kFloat32
                    : dependency == spec.codebook_value_id
                        ? DataType::kFloat32
                        : (spec.storage_dtype == "uint4" ? DataType::kUInt8
                                                         : DataType::kInt8);
                const std::vector<std::int64_t> expected_shape =
                    dependency == spec.codebook_value_id
                        ? std::vector<std::int64_t>({16})
                        : spec.ParameterShape();
                if (record.dtype != DataTypeName(expected_dtype) ||
                    record.shape != expected_shape)
                {
                    throw QuantizationError(
                        "parameter " + std::to_string(dependency) +
                        " must have dtype " + DataTypeName(expected_dtype) +
                        " and shape " +
                        detail::ShapeText(expected_shape) + " for " +
                        spec.scheme);
                }
            }
        }

        // Eagerly validate quantized input payloads and keep them for the
        // dequantization helper so metadata and header cannot disagree later.
        for (const std::string &name : input_names_)
        {
            const auto spec = role_specs.find(name);
            if (spec == role_specs.end())
                continue;
            const auto payload = input_payloads_.find(name);
            if (payload == input_payloads_.end())
                continue;  // declared role without a runnable payload
            const IRTensor tensor = ReadPayloadTensor(
                payload->second, "input role '" + name + "'");
            const json::Value &entry = RoleEntry(root, "inputs", name);
            ValidatePayloadMetadata(entry, tensor,
                                    "input role '" + name + "'");
            const std::int64_t value_id = RoleValueId(entry, "inputs", name);
            const auto record = values.find(value_id);
            if (record == values.end())
            {
                throw QuantizationError("input role '" + name + "' value_id " +
                                        std::to_string(value_id) +
                                        " is not in values");
            }
            ValidateQuantizedValueMetadata(tensor, spec->second, record->second,
                                           "input role '" + name + "'");
            quantized_inputs_[name] = tensor;
        }

        input_quantizations_ = role_specs;
        quantization_parameters_ = parameters;
    }

    static std::map<std::int64_t, Testcase::ValueRecord> ParseValueRegistry(
        const json::Value &root)
    {
        std::map<std::int64_t, ValueRecord> values;
        if (!root.Has("values"))
            return values;
        if (!root.At("values").IsArray())
            throw QuantizationError("testcase manifest 'values' must be an array");
        for (const json::Value &entry : root.At("values").array)
        {
            if (!entry.IsObject())
                throw QuantizationError("testcase manifest 'values' entries must be objects");
            if (!entry.Has("id") || !detail::IsIntegerValue(entry.At("id")))
            {
                throw QuantizationError(
                    "testcase manifest 'values' entry requires an integer id");
            }
            const std::int64_t id =
                static_cast<std::int64_t>(entry.At("id").number);
            if (id < 0)
                throw QuantizationError("value id must be non-negative");
            if (values.count(id) != 0)
                throw QuantizationError("duplicate value ID " + std::to_string(id));
            ValueRecord record;
            if (entry.Has("dtype") && entry.At("dtype").IsString())
                record.dtype = entry.At("dtype").string;
            if (entry.Has("shape"))
                record.shape = detail::ShapeField(entry.At("shape"), "value shape");
            if (entry.Has("quantization"))
            {
                record.has_quantization = true;
                record.spec = ParseQuantizationSpec(
                    entry.At("quantization"),
                    "value " + std::to_string(id) + " quantization");
            }
            values.emplace(id, std::move(record));
        }
        return values;
    }

    static std::map<std::string, QuantizationSpec> ParseRoleQuantizations(
        const json::Value &root,
        const std::map<std::int64_t, ValueRecord> &values)
    {
        std::map<std::string, QuantizationSpec> specs;
        for (const char *field : {"inputs", "outputs"})
        {
            if (!root.Has(field) || !root.At(field).IsArray())
                continue;
            for (const json::Value &entry : root.At(field).array)
            {
                if (!entry.IsObject() || !entry.Has("quantization"))
                    continue;
                const std::string boundary =
                    std::string(field) == "inputs" ? "input" : "output";
                if (!entry.Has("name") || !entry.At("name").IsString() ||
                    entry.At("name").string.empty())
                {
                    throw QuantizationError(std::string("quantized ") + boundary +
                                            " role requires a non-empty name");
                }
                const std::string name = entry.At("name").string;
                const std::int64_t value_id = RoleValueId(entry, field, name);
                const QuantizationSpec spec = ParseQuantizationSpec(
                    entry.At("quantization"),
                    boundary + " role '" + name + "' quantization");
                const auto record = values.find(value_id);
                if (record == values.end())
                {
                    throw QuantizationError(
                        boundary + " role '" + name + "' value_id " +
                        std::to_string(value_id) + " is not in values");
                }
                if (!record->second.has_quantization ||
                    !QuantizationSpecsEqual(record->second.spec, spec))
                {
                    throw QuantizationError(
                        boundary + " role '" + name +
                        "' quantization metadata disagrees with its values record");
                }
                if (boundary == "input")
                    specs[name] = spec;
            }
        }
        return specs;
    }

    static std::int64_t RoleValueId(const json::Value &entry,
                                    const std::string &field,
                                    const std::string &name)
    {
        if (!entry.Has("value_id") || !detail::IsIntegerValue(entry.At("value_id")))
        {
            throw QuantizationError(std::string("quantized ") +
                                    (field == "inputs" ? "input" : "output") +
                                    " role '" + name +
                                    "' requires an integer value_id");
        }
        const double number = entry.At("value_id").number;
        if (number < 0)
            throw QuantizationError("value_id must be non-negative");
        return static_cast<std::int64_t>(number);
    }

    static const json::Value &RoleEntry(const json::Value &root,
                                        const std::string &field,
                                        const std::string &name)
    {
        for (const json::Value &entry : root.At(field).array)
        {
            if (entry.IsObject() && entry.Has("name") &&
                entry.At("name").IsString() && entry.At("name").string == name)
                return entry;
        }
        throw QuantizationError("missing '" + field + "' role '" + name + "'");
    }

    static void ValidatePayloadMetadata(const json::Value &entry,
                                        const IRTensor &tensor,
                                        const std::string &context)
    {
        if (!entry.Has("dtype") || !entry.At("dtype").IsString())
            throw QuantizationError(context + " requires a manifest dtype");
        if (entry.At("dtype").string != DataTypeName(tensor.dtype))
        {
            throw QuantizationError(
                context + " manifest dtype '" + entry.At("dtype").string +
                "' differs from payload header dtype '" +
                DataTypeName(tensor.dtype) + "'");
        }
        if (!entry.Has("shape"))
            throw QuantizationError(context + " requires a manifest shape");
        const std::vector<std::int64_t> shape =
            detail::ShapeField(entry.At("shape"), context + " shape");
        if (shape != tensor.shape)
        {
            throw QuantizationError(
                context + " manifest shape " + detail::ShapeText(shape) +
                " differs from payload header shape " + tensor.ShapeString());
        }
        if (entry.Has("stride"))
        {
            const std::vector<std::int64_t> stride = detail::IntegerArrayField(
                entry.At("stride"), context + " stride",
                std::numeric_limits<std::int64_t>::min());
            if (stride != tensor.stride)
                throw QuantizationError(context +
                                        " manifest stride differs from payload header");
        }
        if (entry.Has("storage_offset"))
        {
            const std::int64_t offset = detail::IntegerValue(
                entry.At("storage_offset"), context + " storage_offset", 0);
            if (offset != tensor.storage_offset)
            {
                throw QuantizationError(
                    context + " manifest storage_offset differs from payload header");
            }
        }
        const std::uint64_t expected_nbytes = PayloadSize(
            tensor.dtype, static_cast<std::uint64_t>(tensor.Numel()));
        if (expected_nbytes != tensor.data.size())
        {
            throw QuantizationError(
                context + " payload nbytes differs from its header dtype/shape");
        }
        if (entry.Has("nbytes"))
        {
            const std::int64_t nbytes = detail::IntegerValue(
                entry.At("nbytes"), context + " nbytes", 0);
            if (static_cast<std::uint64_t>(nbytes) != expected_nbytes)
            {
                throw QuantizationError(
                    context + " manifest nbytes differs from payload header nbytes");
            }
        }
    }

    static void ValidateQuantizedValueMetadata(
        const IRTensor &tensor, const QuantizationSpec &spec,
        const ValueRecord &record, const std::string &context)
    {
        const DataType expected_dtype = spec.storage_dtype == "int8"
            ? DataType::kInt8
            : spec.storage_dtype == "int4" ? DataType::kInt4 : DataType::kUInt4;
        if (tensor.dtype != expected_dtype || tensor.shape != spec.logical_shape)
        {
            throw QuantizationError(
                context + " storage dtype/shape differs from its quantization "
                "descriptor");
        }
        if (record.dtype != spec.storage_dtype || record.shape != spec.logical_shape)
        {
            throw QuantizationError(
                context + " values record dtype/shape differs from its "
                "quantization descriptor");
        }
    }

    // Payload read failures (absent file, truncated header, ...) are reported
    // as TestcaseError so Load() exposes one error class for an invalid
    // testcase directory.  The context names the offending value ID or role;
    // the message also carries the payload path.
    IRTensor ReadPayloadTensor(const std::string &payload,
                               const std::string &context) const
    {
        try
        {
            return ReadIRTensor(dir_ + "/" + payload);
        }
        catch (const IRTensorError &error)
        {
            throw TestcaseError(context + " payload '" + payload +
                                "' is unreadable: " + error.what());
        }
    }

    std::map<std::int64_t, IRTensor> ParseQuantizationParameters(
        const json::Value &root,
        const std::map<std::int64_t, ValueRecord> &values,
        const std::vector<std::int64_t> &parameter_ids) const
    {
        std::map<std::int64_t, IRTensor> parameters;
        if (!root.Has("quantization_parameters"))
        {
            if (!parameter_ids.empty())
            {
                throw QuantizationError(
                    "testcase manifest is missing quantization_parameters");
            }
            return parameters;
        }
        if (!root.At("quantization_parameters").IsArray())
        {
            throw QuantizationError(
                "testcase manifest 'quantization_parameters' must be an array");
        }

        // First pass: validate every declared ID, reference, and payload path
        // without touching payload bytes.  A malformed collection must fail
        // with its own diagnosis even when a payload file is absent.
        struct Declaration
        {
            std::int64_t value_id;
            std::string payload;
            const json::Value *entry;
        };
        std::vector<Declaration> declarations;
        std::map<std::int64_t, bool> declared;
        for (const json::Value &entry : root.At("quantization_parameters").array)
        {
            if (!entry.IsObject())
            {
                throw QuantizationError(
                    "'quantization_parameters' entries must be objects");
            }
            // The schema key is value_id; an `id`-only entry would silently
            // drop the parameter reference.
            if (!entry.Has("value_id"))
            {
                throw QuantizationError(
                    "quantization parameter entries must use the 'value_id' key");
            }
            if (!detail::IsIntegerValue(entry.At("value_id")))
            {
                throw QuantizationError(
                    "quantization parameter value_id must be an integer");
            }
            const double number = entry.At("value_id").number;
            if (number < 0)
            {
                throw QuantizationError(
                    "quantization parameter value_id must be non-negative");
            }
            const std::int64_t value_id = static_cast<std::int64_t>(number);
            if (declared.count(value_id) != 0)
            {
                throw QuantizationError(
                    "duplicate quantization parameter value_id " +
                    std::to_string(value_id));
            }
            declared[value_id] = true;
            const auto record = values.find(value_id);
            if (record == values.end())
            {
                throw QuantizationError(
                    "quantization parameter value_id " + std::to_string(value_id) +
                    " is not in values");
            }
            if (record->second.has_quantization)
            {
                throw QuantizationError(
                    "nested quantized parameters are unsupported at value ID " +
                    std::to_string(value_id));
            }
            if (!entry.Has("payload") || entry.At("payload").IsNull())
            {
                throw QuantizationError(
                    "quantization parameter " + std::to_string(value_id) +
                    " has no payload");
            }
            if (!entry.At("payload").IsString())
            {
                throw QuantizationError(
                    "quantization parameter payload must be a string");
            }
            const std::string payload = entry.At("payload").string;
            if (!IsSafeRelativePath(payload))
            {
                throw QuantizationError(
                    "quantization parameter payload for value " +
                    std::to_string(value_id) +
                    " is not a safe relative path");
            }
            declarations.push_back({value_id, payload, &entry});
        }

        std::vector<std::int64_t> declared_ids;
        declared_ids.reserve(declared.size());
        for (const auto &item : declared)
            declared_ids.push_back(item.first);
        if (declared_ids != parameter_ids)
        {
            std::string missing;
            for (const std::int64_t id : parameter_ids)
            {
                if (declared.count(id) == 0)
                    missing += (missing.empty() ? "" : ", ") + std::to_string(id);
            }
            std::string unreferenced;
            for (const std::int64_t id : declared_ids)
            {
                if (!std::binary_search(parameter_ids.begin(),
                                        parameter_ids.end(), id))
                {
                    unreferenced +=
                        (unreferenced.empty() ? "" : ", ") + std::to_string(id);
                }
            }
            throw QuantizationError(
                "quantization_parameters must contain each referenced parameter "
                "exactly once (missing: " + (missing.empty() ? "none" : missing) +
                "; unreferenced: " +
                (unreferenced.empty() ? "none" : unreferenced) + ")");
        }

        // Second pass: load and cross-check each payload against its manifest
        // metadata and the value registry record.
        for (const Declaration &declaration : declarations)
        {
            const std::string context =
                "quantization parameter '" + std::to_string(declaration.value_id) +
                "'";
            const IRTensor tensor =
                ReadPayloadTensor(declaration.payload, context);
            ValidatePayloadMetadata(*declaration.entry, tensor, context);
            const ValueRecord &record = values.at(declaration.value_id);
            if (record.dtype != DataTypeName(tensor.dtype) ||
                record.shape != tensor.shape)
            {
                throw QuantizationError(context +
                                        " values record dtype/shape differs from "
                                        "the payload header");
            }
            parameters.emplace(declaration.value_id, tensor);
        }
        return parameters;
    }

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

    void RequireInputRole(const std::string &name) const
    {
        if (std::find(input_names_.begin(), input_names_.end(), name) ==
            input_names_.end())
        {
            throw TestcaseError("unknown input role '" + name + "'");
        }
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
