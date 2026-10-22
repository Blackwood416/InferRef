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
// Header-only and dependency-free, layered on testcase.hpp.

#ifndef INFERREF_BRIDGE_HPP
#define INFERREF_BRIDGE_HPP

#include <inferref/testcase.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
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

namespace detail
{

// Quantized input roles in manifest order.  Empty for testcase formats 0.1-0.3
// and for 0.4 testcases whose inputs carry no quantization descriptor.
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

// Dispatch an already-loaded testcase and publish its outputs.  `caller_label`
// names the callback in manifest errors, preserving each path's diagnostics.
inline int Dispatch(Testcase &testcase, const std::string &program_name,
                    const std::string &caller_label,
                    const DebugInvoke &invoke)
{
    const std::string region = testcase.RegionName();
    if (region.empty())
    {
        std::cerr
            << program_name << ": missing_region: testcase has no "
               "origin.region, origin.contract, or contracts[0]\n";
        return kBridgeMissingRegion;
    }
    const std::map<std::string, IRTensor> outputs =
        invoke(region, testcase.Inputs());
    testcase.WriteOutputs(outputs, caller_label);
    testcase.Finish();
    return kBridgeOk;
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
        return detail::Dispatch(
            testcase, program_name, "DebugInvokeEx",
            [&](const std::string &region,
                const std::map<std::string, IRTensor> &inputs)
            { return invoke(testcase, region, inputs); });
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
        const std::vector<std::string> quantized =
            detail::QuantizedInputRoles(testcase);
        if (!quantized.empty())
        {
            std::cerr
                << program_name
                << ": unsupported_quantization: testcase format_version "
                << testcase.FormatVersion() << " quantizes input role(s) "
                << detail::QuoteRoles(quantized)
                << "; the legacy DebugInvoke callback cannot see scales. "
                   "Register a DebugInvokeEx callback and query "
                   "testcase.InputQuantization() / QuantizationParameter() / "
                   "DequantizeInputToFloat32() instead.\n";
            return kBridgeUnsupportedQuantization;
        }
        testcase.SetOutputDir(output_dir);
        return detail::Dispatch(testcase, program_name, "DebugInvoke", invoke);
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

} // namespace inferref

#endif // INFERREF_BRIDGE_HPP
