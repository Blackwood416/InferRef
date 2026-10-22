// Reference runtime bridge (InferRef 0.8 Adapter DX, section 7.3).
//
// The bridge contains no model knowledge.  A downstream runtime registers one
// DebugInvoke callback here and supports every future region without writing
// another adapter:
//
//     inferref-bridge --testcase testcase/ --output output/
//     inferref-bridge --batch-manifest batch.json
//
// Batch mode (InferRef 0.12 section 7.3) dispatches every manifest item
// through the one registered callback in this process, so the engine is
// initialized once.  batch-results.json is written next to the manifest.

#include <inferref/bridge.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    std::string batch_manifest;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
        else if (arg == "--batch-manifest" && i + 1 < argc)
            batch_manifest = argv[++i];
    }
    if (!batch_manifest.empty())
    {
        if (!testcase_dir.empty() || !output_dir.empty())
        {
            std::cerr << "inferref-bridge: --batch-manifest cannot be combined "
                         "with --testcase/--output\n";
            return inferref::kBridgeUsage;
        }
    }
    else if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref-bridge --testcase DIR --output DIR\n"
                     "       inferref-bridge --batch-manifest FILE\n";
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

    if (!batch_manifest.empty())
        return inferref::RunBridgeBatch(batch_manifest, invoke);

    return inferref::RunBridge(testcase_dir, output_dir, invoke);
}
