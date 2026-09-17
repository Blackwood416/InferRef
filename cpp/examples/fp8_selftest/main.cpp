// Exhaustive FP8 decoding and optional Python -> C++ -> Python wire goldens.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "inferref/compare.hpp"
#include "inferref/irtensor.hpp"

void Require(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

bool Same(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) ||
           (a == b && std::signbit(a) == std::signbit(b));
}

int main(int argc, char **argv)
{
    try
    {
        for (auto dtype : {inferref::DataType::kFloat8E4M3FN, inferref::DataType::kFloat8E5M2})
        {
            const bool e4 = dtype == inferref::DataType::kFloat8E4M3FN;
            Require(static_cast<int>(dtype) == (e4 ? 16 : 17), "frozen FP8 code");
            Require(inferref::DataTypeSize(dtype) == 1, "FP8 size");
            // Construct expected positive exponent bins, independently of decoder.
            std::vector<float> expected;
            const int bins = e4 ? 8 : 4;
            const float sub_step = e4 ? 1.0f / 512 : 1.0f / 65536;
            for (int i = 0; i < bins; ++i) expected.push_back(i * sub_step);
            for (int e = e4 ? -6 : -14; e <= (e4 ? 8 : 15); ++e)
                for (int i = 0; i < bins; ++i)
                    expected.push_back(std::ldexp(1.0f + static_cast<float>(i) / bins, e));
            if (e4) expected.back() = std::nanf("");
            else
            {
                expected.push_back(std::numeric_limits<float>::infinity());
                for (int i = 0; i < 3; ++i) expected.push_back(std::nanf(""));
            }
            Require(expected.size() == 128, "expected positive encodings");
            for (int i = 0; i < 128; ++i) expected.push_back(-expected[i]);
            inferref::IRTensor t;
            t.dtype = dtype;
            t.shape = {16, 16};
            t.stride = {16, 1};
            for (int i = 0; i < 256; ++i) t.data.push_back(static_cast<std::byte>(i));
            const auto decoded = t.AsFloat32();
            for (int i = 0; i < 256; ++i)
                Require(Same(decoded[i], expected[i]), "FP8 exhaustive decode");
            auto changed = t;
            changed.data[e4 ? 56 : 60] = static_cast<std::byte>(e4 ? 57 : 61);
            Require(!inferref::CompareTensors(t, changed).passed, "FP8 numerical mismatch");
            if (argc > 1)
            {
                const std::string base = std::string(argv[1]) + "/" + inferref::DataTypeName(dtype);
                const auto loaded = inferref::ReadIRTensor(base + ".irtensor");
                Require(loaded.dtype == dtype && loaded.shape == t.shape && loaded.data == t.data,
                        "Python wire input");
                const auto golden = inferref::ReadIRTensor(base + ".float32.irtensor").AsFloat32();
                Require(golden.size() == decoded.size(), "Python golden size");
                for (int i = 0; i < 256; ++i)
                    Require(Same(decoded[i], golden[i]), "Python/C++ FP8 parity");
                inferref::WriteIRTensor(base + ".roundtrip.irtensor", loaded);
            }
        }
        bool rejected = false;
        try { inferref::DecodeFloat8(0, inferref::DataType::kUInt8); }
        catch (const inferref::IRTensorError &) { rejected = true; }
        Require(rejected, "wrong dtype rejected");
        std::cout << "FP8: all 512 encodings passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
