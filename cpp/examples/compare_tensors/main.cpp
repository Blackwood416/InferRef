// inferref_compare — compare two .irtensor files without PyTorch or Python.
//
//     inferref_compare <reference.irtensor> <actual.irtensor> [--strict-layout]
//
// Exit code 0 on PASS, 1 on FAIL, 2 on usage/IO error — so it drops straight
// into a CI step or an agent's build-and-check loop (SPEC §43).

#include <cstdio>
#include <cstring>
#include <string>

#include "inferref/compare.hpp"
#include "inferref/irtensor.hpp"

namespace
{

void PrintUsage(const char *program)
{
    std::printf("usage: %s <reference.irtensor> <actual.irtensor> [--strict-layout]\n",
                program);
}

void PrintTensor(const char *label, const inferref::IRTensor &tensor)
{
    std::printf("%s\n", label);
    std::printf("  dtype:          %s\n", inferref::DataTypeName(tensor.dtype));
    std::printf("  shape:          %s\n", tensor.ShapeString().c_str());
    std::printf("  stride:         %s\n", tensor.StrideString().c_str());
    std::printf("  storage_offset: %lld\n",
                static_cast<long long>(tensor.storage_offset));
    std::printf("  numel:          %lld\n", static_cast<long long>(tensor.Numel()));
}

std::string IndexString(const std::vector<std::int64_t> &index)
{
    std::string out = "[";
    for (std::size_t i = 0; i < index.size(); ++i)
    {
        if (i != 0)
        {
            out += ", ";
        }
        out += std::to_string(index[i]);
    }
    return out + "]";
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        PrintUsage(argv[0]);
        return 2;
    }

    bool strict_layout = false;
    for (int i = 3; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--strict-layout") == 0)
        {
            strict_layout = true;
        }
        else
        {
            std::fprintf(stderr, "error: unknown option %s\n", argv[i]);
            PrintUsage(argv[0]);
            return 2;
        }
    }

    inferref::IRTensor reference;
    inferref::IRTensor actual;
    try
    {
        reference = inferref::ReadIRTensor(argv[1]);
        actual = inferref::ReadIRTensor(argv[2]);
    }
    catch (const inferref::IRTensorError &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }

    std::printf("InferRef C++ comparison (no PyTorch, no Python)\n\n");
    PrintTensor("Reference:", reference);
    std::printf("\n");
    PrintTensor("Actual:", actual);
    std::printf("\n");

    const inferref::ComparisonResult result =
        inferref::CompareTensors(reference, actual, strict_layout);

    if (!result.layout.Comparable())
    {
        std::printf("Layout: NOT COMPARABLE (%s)\n", result.message.c_str());
        std::printf("\nResult: FAIL\n");
        return 1;
    }

    if (result.layout.AnyMismatch())
    {
        std::printf("Layout differences:\n");
        if (result.layout.stride_mismatch)
        {
            std::printf("  stride:         %s -> %s\n", reference.StrideString().c_str(),
                        actual.StrideString().c_str());
        }
        if (result.layout.storage_offset_mismatch)
        {
            std::printf("  storage_offset: %lld -> %lld\n",
                        static_cast<long long>(reference.storage_offset),
                        static_cast<long long>(actual.storage_offset));
        }
        std::printf("\n");
    }

    const inferref::Metrics &metrics = result.metrics;
    std::printf("Metrics:\n");
    std::printf("  max_abs_error:     %.9g\n", metrics.max_abs_error);
    std::printf("  max_rel_error:     %.9g\n", metrics.max_rel_error);
    std::printf("  mean_abs_error:    %.9g\n", metrics.mean_abs_error);
    std::printf("  rmse:              %.9g\n", metrics.rmse);
    std::printf("  cosine_similarity: %.9g\n", metrics.cosine_similarity);
    std::printf("  mismatched:        %lld / %lld\n",
                static_cast<long long>(metrics.mismatch_count),
                static_cast<long long>(metrics.element_count));
    std::printf("  nan / inf:         %lld / %lld\n",
                static_cast<long long>(metrics.nan_count),
                static_cast<long long>(metrics.inf_count));

    if (metrics.has_first_mismatch)
    {
        std::printf("\nFirst mismatching element:\n");
        std::printf("  index:     %s\n",
                    IndexString(metrics.first_mismatch_index).c_str());
        std::printf("  reference: %.9g\n", metrics.first_mismatch_reference);
        std::printf("  actual:    %.9g\n", metrics.first_mismatch_actual);
    }

    if (!result.message.empty())
    {
        std::printf("\n%s\n", result.message.c_str());
    }
    std::printf("\nResult: %s\n", result.passed ? "PASS" : "FAIL");
    return result.passed ? 0 : 1;
}
