// Reference runtime bridge (InferRef 0.8 Adapter DX, section 7.3).
//
// The bridge contains no model knowledge.  A downstream runtime registers one
// DebugInvoke callback here and supports every future region without writing
// another adapter:
//
//     inferref-bridge --testcase testcase/ --output output/

#include <inferref/bridge.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }
    if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref-bridge --testcase DIR --output DIR\n";
        return inferref::kBridgeUsage;
    }

    // Register one DebugInvoke callback for your runtime here.  The bridge
    // resolves the region key (origin.region -> origin.contract ->
    // contracts[0]) and dispatches by name; the callback must return every
    // declared output by role name.
    inferref::DebugInvoke invoke =
        [](const std::string &region_name,
           const std::map<std::string, inferref::IRTensor> &inputs)
            -> std::map<std::string, inferref::IRTensor>
    {
        (void)region_name;
        (void)inputs;
        throw std::runtime_error(
            "inferref-bridge: no DebugInvoke registered by the runtime");
    };

    return inferref::RunBridge(testcase_dir, output_dir, invoke);
}
