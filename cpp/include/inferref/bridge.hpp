// Generic runtime bridge (InferRef 0.8 Adapter DX, section 7).
//
// One bridge binary dispatches any region into an existing runtime through a
// DebugInvoke callback: region name + named inputs -> named outputs.  The
// bridge contains no model knowledge; the runtime registers the callback.
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

namespace inferref
{

// Runtime callback: region name + named inputs -> named outputs.  Outputs are
// keyed by role name, matching Engine Adapter v0.2's exact output-role
// contract.
using DebugInvoke = std::function<std::map<std::string, inferref::IRTensor>(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)>;

// RunBridge exit codes.
inline constexpr int kBridgeOk = 0;
inline constexpr int kBridgeError = 1;
inline constexpr int kBridgeUsage = 2;
inline constexpr int kBridgeMissingRegion = 3;

// Load one testcase, dispatch through `invoke`, write every declared output
// by role name into `output_dir`, and publish manifest.json once.
inline int RunBridge(const std::string &testcase_dir,
                     const std::string &output_dir,
                     const DebugInvoke &invoke,
                     const std::string &program_name = "inferref-bridge")
{
    try
    {
        Testcase testcase = Testcase::Load(testcase_dir);
        testcase.SetOutputDir(output_dir);
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
        testcase.WriteOutputs(outputs, "DebugInvoke");
        testcase.Finish();
        return kBridgeOk;
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
