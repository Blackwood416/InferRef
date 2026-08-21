// Self-test for the high-level testcase helper (cpp/include/inferref/testcase.hpp).
//
// Covers load, named input reads, named output writes, Finish() idempotence,
// region dispatch-key fallback, missing-role errors, and unsafe payload
// rejection.  Run it with no arguments; exit code 0 means every check passed.

#include <inferref/testcase.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
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

std::string TestcaseJson(const std::string &origin_json, const std::string &contracts_json)
{
    std::string json =
        "{\n"
        "  \"format\": \"inferref-testcase\",\n"
        "  \"format_version\": \"0.2\",\n"
        "  \"name\": \"selftest\"";
    if (!origin_json.empty())
        json += ",\n  " + origin_json;
    json +=
        ",\n"
        "  \"inputs\": [\n"
        "    {\"name\": \"x\", \"payload\": \"inputs/x.irtensor\"},\n"
        "    {\"name\": \"scale\", \"payload\": \"inputs/scale.irtensor\"}\n"
        "  ],\n"
        "  \"outputs\": [\n"
        "    {\"name\": \"y\"},\n"
        "    {\"name\": \"z\"}\n"
        "  ]";
    if (!contracts_json.empty())
        json += ",\n  " + contracts_json;
    json += "\n}\n";
    return json;
}

void TestLoadAndInputs(const std::string &dir)
{
    std::printf("load, names, region, and named input reads\n");
    const inferref::Testcase testcase = inferref::Testcase::Load(dir);

    Check(testcase.RegionName() == "demo/region/v1", "RegionName from origin.region");
    Check(testcase.InputNames() == std::vector<std::string>({"x", "scale"}),
          "InputNames order preserved");
    Check(testcase.OutputNames() == std::vector<std::string>({"y", "z"}),
          "OutputNames order preserved");

    const inferref::IRTensor x = testcase.Input("x");
    Check(x.dtype == inferref::DataType::kFloat32, "Input(x) dtype");
    Check(x.shape == std::vector<std::int64_t>({2, 2}), "Input(x) shape");
    const std::vector<float> values = x.AsFloat32();
    Check(values.size() == 4 && values[0] == 1.0f && values[3] == 4.0f,
          "Input(x) values");

    const std::map<std::string, inferref::IRTensor> all = testcase.Inputs();
    Check(all.size() == 2, "Inputs() size");
    Check(all.count("x") == 1 && all.count("scale") == 1, "Inputs() keys");
    Check(all.at("scale").Numel() == 1, "Inputs() scale loaded");

    bool unknown_input_rejected = false;
    try
    {
        (void)testcase.Input("missing");
    }
    catch (const inferref::TestcaseError &)
    {
        unknown_input_rejected = true;
    }
    Check(unknown_input_rejected, "Input(unknown role) throws TestcaseError");
}

void TestOutputsAndFinish(const std::string &dir)
{
    std::printf("named output writes and idempotent Finish\n");
    inferref::Testcase testcase = inferref::Testcase::Load(dir);
    const inferref::IRTensor out = MakeFloat32({2}, {9.0f, 8.0f});

    testcase.WriteOutput("y", out);
    testcase.WriteOutput("z", MakeFloat32({1}, {7.0f}));
    Check(std::filesystem::exists(dir + "/y.irtensor"), "WriteOutput(y) file exists");
    Check(std::filesystem::exists(dir + "/z.irtensor"), "WriteOutput(z) file exists");

    const inferref::IRTensor reloaded = inferref::ReadIRTensor(dir + "/y.irtensor");
    Check(reloaded.AsFloat32()[1] == 8.0f, "written output round-trips");

    bool unknown_output_rejected = false;
    try
    {
        testcase.WriteOutput("nope", out);
    }
    catch (const inferref::TestcaseError &)
    {
        unknown_output_rejected = true;
    }
    Check(unknown_output_rejected, "WriteOutput(unknown role) throws TestcaseError");

    testcase.Finish();
    const std::string first = ReadFile(dir + "/manifest.json");
    const inferref::json::Value manifest = inferref::json::Parse(first);
    Check(manifest.At("outputs").array.size() == 2, "manifest has two outputs");
    Check(manifest.At("outputs").array[0].At("name").string == "y" &&
              manifest.At("outputs").array[0].At("payload").string == "y.irtensor",
          "manifest entry y ordered first");
    Check(manifest.At("outputs").array[1].At("name").string == "z" &&
              manifest.At("outputs").array[1].At("payload").string == "z.irtensor",
          "manifest entry z ordered second");

    testcase.Finish();
    Check(ReadFile(dir + "/manifest.json") == first, "Finish() is idempotent");

    bool write_after_finish_rejected = false;
    try
    {
        testcase.WriteOutput("y", out);
    }
    catch (const inferref::TestcaseError &)
    {
        write_after_finish_rejected = true;
    }
    Check(write_after_finish_rejected, "WriteOutput after Finish throws");
}

void TestRegionFallback(const std::string &base)
{
    std::printf("region dispatch-key fallback\n");
    const std::string contract_only = base + "/contract-only";
    std::filesystem::create_directories(contract_only + "/inputs");
    inferref::WriteIRTensor(contract_only + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    inferref::WriteIRTensor(contract_only + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(contract_only + "/testcase.json",
              TestcaseJson("\"origin\": {\"contract\": \"fallback/contract/v1\"}",
                           "\"contracts\": [\"ignored/contract/v1\"]"));
    Check(inferref::Testcase::Load(contract_only).RegionName() ==
              "fallback/contract/v1",
          "origin.contract wins over contracts[0]");

    const std::string contracts_only = base + "/contracts-only";
    std::filesystem::create_directories(contracts_only + "/inputs");
    inferref::WriteIRTensor(contracts_only + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    inferref::WriteIRTensor(contracts_only + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(contracts_only + "/testcase.json",
              TestcaseJson("", "\"contracts\": [\"rope/rotate-half/v1\"]"));
    Check(inferref::Testcase::Load(contracts_only).RegionName() ==
              "rope/rotate-half/v1",
          "contracts[0] used when origin has no region");

    const std::string no_key = base + "/no-key";
    std::filesystem::create_directories(no_key + "/inputs");
    inferref::WriteIRTensor(no_key + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    inferref::WriteIRTensor(no_key + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(no_key + "/testcase.json", TestcaseJson("", ""));
    Check(inferref::Testcase::Load(no_key).RegionName().empty(),
          "empty RegionName when no dispatch key exists");
}

void TestErrors(const std::string &base)
{
    std::printf("missing payload, unsafe paths, and load errors\n");
    const std::string missing_payload = base + "/missing-payload";
    std::filesystem::create_directories(missing_payload + "/inputs");
    inferref::WriteIRTensor(missing_payload + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(missing_payload + "/testcase.json",
              "{\"format\": \"inferref-testcase\","
              " \"format_version\": \"0.2\","
              " \"inputs\": [{\"name\": \"x\", \"payload\": \"inputs/x.irtensor\"},"
              " {\"name\": \"scale\", \"payload\": null}],"
              " \"outputs\": [{\"name\": \"y\"}]}");
    inferref::Testcase loaded = inferref::Testcase::Load(missing_payload);
    bool missing_payload_rejected = false;
    try
    {
        (void)loaded.Input("scale");
    }
    catch (const inferref::TestcaseError &error)
    {
        missing_payload_rejected =
            std::string(error.what()).find("has no runnable payload") !=
            std::string::npos;
    }
    Check(missing_payload_rejected,
          "Input without payload says 'has no runnable payload'");

    const std::string unsafe = base + "/unsafe";
    std::filesystem::create_directories(unsafe);
    WriteFile(unsafe + "/testcase.json",
              "{\"format\": \"inferref-testcase\","
              " \"format_version\": \"0.2\","
              " \"inputs\": [{\"name\": \"x\", \"payload\": \"../escape.irtensor\"}],"
              " \"outputs\": [{\"name\": \"y\"}]}");
    bool unsafe_rejected = false;
    try
    {
        (void)inferref::Testcase::Load(unsafe);
    }
    catch (const inferref::TestcaseError &)
    {
        unsafe_rejected = true;
    }
    Check(unsafe_rejected, "unsafe payload path rejected at load");

    bool missing_dir_rejected = false;
    try
    {
        (void)inferref::Testcase::Load(base + "/does-not-exist");
    }
    catch (const inferref::TestcaseError &)
    {
        missing_dir_rejected = true;
    }
    Check(missing_dir_rejected, "Load on missing directory throws");

    const std::string malformed = base + "/malformed";
    std::filesystem::create_directories(malformed);
    WriteFile(malformed + "/testcase.json", "{not json");
    bool malformed_rejected = false;
    try
    {
        (void)inferref::Testcase::Load(malformed);
    }
    catch (const inferref::json::ParseError &)
    {
        malformed_rejected = true;
    }
    Check(malformed_rejected, "malformed manifest JSON raises ParseError");
}

void TestWriteOutputs(const std::string &base)
{
    std::printf("batch WriteOutputs with role validation and atomicity\n");
    const std::string dir = base + "/write-outputs";
    std::filesystem::create_directories(dir + "/inputs");
    inferref::WriteIRTensor(dir + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    inferref::WriteIRTensor(dir + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(dir + "/testcase.json", TestcaseJson("", ""));

    // 1. Missing output role error + atomic guarantee (no partial file written)
    {
        inferref::Testcase testcase = inferref::Testcase::Load(dir);
        std::map<std::string, inferref::IRTensor> partial_map = {
            {"y", MakeFloat32({2}, {1.0f, 2.0f})},
        };
        bool missing_role_threw = false;
        std::string err_msg;
        try
        {
            testcase.WriteOutputs(partial_map, "RunYourEngine");
        }
        catch (const inferref::TestcaseError &error)
        {
            missing_role_threw = true;
            err_msg = error.what();
        }
        Check(missing_role_threw, "WriteOutputs missing role throws TestcaseError");
        Check(err_msg == "RunYourEngine did not return output role 'z'",
              "WriteOutputs error message formats '{caller_label} did not return output role '{role}''");
        Check(!std::filesystem::exists(dir + "/y.irtensor"),
              "WriteOutputs leaves no partial files on missing role error");
    }

    // 2. Default caller label check on missing role
    {
        inferref::Testcase testcase = inferref::Testcase::Load(dir);
        std::map<std::string, inferref::IRTensor> partial_map = {
            {"y", MakeFloat32({2}, {1.0f, 2.0f})},
        };
        std::string err_msg;
        try
        {
            testcase.WriteOutputs(partial_map);
        }
        catch (const inferref::TestcaseError &error)
        {
            err_msg = error.what();
        }
        Check(err_msg == "engine did not return output role 'z'",
              "WriteOutputs default caller label is 'engine'");
    }

    // 3. Undeclared output role error + atomic guarantee
    {
        inferref::Testcase testcase = inferref::Testcase::Load(dir);
        std::map<std::string, inferref::IRTensor> extra_map = {
            {"y", MakeFloat32({2}, {1.0f, 2.0f})},
            {"z", MakeFloat32({1}, {3.0f})},
            {"extra_role", MakeFloat32({1}, {4.0f})},
        };
        bool extra_role_threw = false;
        std::string err_msg;
        try
        {
            testcase.WriteOutputs(extra_map, "RunYourEngine");
        }
        catch (const inferref::TestcaseError &error)
        {
            extra_role_threw = true;
            err_msg = error.what();
        }
        Check(extra_role_threw, "WriteOutputs undeclared role throws TestcaseError");
        Check(err_msg == "RunYourEngine returned undeclared output role 'extra_role'",
              "WriteOutputs error message formats '{caller_label} returned undeclared output role '{role}''");
        Check(!std::filesystem::exists(dir + "/y.irtensor"),
              "WriteOutputs leaves no partial files on undeclared role error");
    }

    // 4. Happy path: valid outputs write all files and manifest matches on Finish()
    {
        inferref::Testcase testcase = inferref::Testcase::Load(dir);
        std::map<std::string, inferref::IRTensor> valid_map = {
            {"y", MakeFloat32({2}, {10.0f, 20.0f})},
            {"z", MakeFloat32({1}, {30.0f})},
        };
        testcase.WriteOutputs(valid_map, "RunYourEngine");
        Check(std::filesystem::exists(dir + "/y.irtensor"), "valid WriteOutputs writes y.irtensor");
        Check(std::filesystem::exists(dir + "/z.irtensor"), "valid WriteOutputs writes z.irtensor");

        testcase.Finish();
        const std::string manifest_str = ReadFile(dir + "/manifest.json");
        const inferref::json::Value manifest = inferref::json::Parse(manifest_str);
        Check(manifest.At("outputs").array.size() == 2, "manifest has 2 outputs after Finish");
        Check(manifest.At("outputs").array[0].At("name").string == "y", "manifest output 0 is y");
        Check(manifest.At("outputs").array[1].At("name").string == "z", "manifest output 1 is z");

        // 5. WriteOutputs after Finish throws TestcaseError
        bool after_finish_threw = false;
        try
        {
            testcase.WriteOutputs(valid_map, "RunYourEngine");
        }
        catch (const inferref::TestcaseError &error)
        {
            after_finish_threw = true;
            Check(std::string(error.what()) == "cannot write output after Finish()",
                  "WriteOutputs after Finish throws 'cannot write output after Finish()'");
        }
        Check(after_finish_threw, "WriteOutputs after Finish throws");
    }
}

void TestPartialFinish(const std::string &base)
{
    std::printf("Finish publishes only written outputs\n");
    const std::string partial = base + "/partial";
    std::filesystem::create_directories(partial + "/inputs");
    inferref::WriteIRTensor(partial + "/inputs/x.irtensor",
                            MakeFloat32({1}, {1.0f}));
    inferref::WriteIRTensor(partial + "/inputs/scale.irtensor",
                            MakeFloat32({1}, {1.0f}));
    WriteFile(partial + "/testcase.json", TestcaseJson("", ""));
    inferref::Testcase testcase = inferref::Testcase::Load(partial);
    testcase.WriteOutput("y", MakeFloat32({1}, {1.0f}));
    testcase.Finish();
    const inferref::json::Value manifest =
        inferref::json::Parse(ReadFile(partial + "/manifest.json"));
    Check(manifest.At("outputs").array.size() == 1 &&
              manifest.At("outputs").array[0].At("name").string == "y",
          "unwritten output omitted from manifest");
}

void TestExternalTestcase(const std::string &directory)
{
    std::printf("loading Python-written testcase from %s\n", directory.c_str());
    try
    {
        const inferref::Testcase testcase = inferref::Testcase::Load(directory);
        std::printf("  region: %s\n",
                    testcase.RegionName().empty() ? "(none)" : testcase.RegionName().c_str());
        std::printf("  inputs: ");
        for (const std::string &name : testcase.InputNames())
            std::printf("%s ", name.c_str());
        std::printf("\n");
        std::printf("  loaded inputs: %zu\n", testcase.Inputs().size());
        Check(testcase.Inputs().size() == testcase.InputNames().size(),
              "every declared input has a runnable payload");
    }
    catch (const inferref::TestcaseError &error)
    {
        std::printf("  skip (%s)\n", error.what());
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("InferRef testcase.hpp C++ self-test\n\n");

    const std::string base = "inferref_testcase_selftest_tmp";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base + "/inputs");
    try
    {
        inferref::WriteIRTensor(base + "/inputs/x.irtensor",
                                MakeFloat32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
        inferref::WriteIRTensor(base + "/inputs/scale.irtensor",
                                MakeFloat32({1}, {2.5f}));
        WriteFile(base + "/testcase.json",
                  TestcaseJson("\"origin\": {\"region\": \"demo/region/v1\"}",
                               "\"contracts\": [\"ignored/contract/v1\"]"));

        TestLoadAndInputs(base);
        TestOutputsAndFinish(base);
        TestWriteOutputs(base);
        TestRegionFallback(base);
        TestErrors(base);
        TestPartialFinish(base);
    }
    catch (const std::exception &error)
    {
        std::printf("  FAIL uncaught exception: %s\n", error.what());
        ++g_failures;
    }

    std::filesystem::remove_all(base);
    if (argc > 1)
    {
        TestExternalTestcase(argv[1]);
    }
    std::printf("\n%s\n", g_failures == 0 ? "All checks passed." : "FAILURES PRESENT.");
    return g_failures == 0 ? 0 : 1;
}
