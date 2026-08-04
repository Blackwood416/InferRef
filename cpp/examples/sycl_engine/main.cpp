#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "inferref/irtensor.hpp"

namespace fs = std::filesystem;

namespace
{
std::string ReadText(const fs::path &path)
{
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string TestcaseName(const fs::path &root)
{
    const std::string text = ReadText(root / "testcase.json");
    const auto key = text.find("\"name\"");
    if (key == std::string::npos) throw std::runtime_error("testcase.json has no name");
    const auto colon = text.find(':', key);
    const auto first = text.find('"', colon + 1);
    const auto last = text.find('"', first + 1);
    if (colon == std::string::npos || first == std::string::npos || last == std::string::npos)
        throw std::runtime_error("testcase.json has no name");
    return text.substr(first + 1, last - first - 1);
}

std::uint16_t FloatToHalf(float value)
{
    std::uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    mantissa += 0x1000u;
    if (mantissa & 0x800000u) { mantissa = 0; ++exponent; }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
}

inferref::IRTensor EncodeLike(const inferref::IRTensor &source, const std::vector<float> &values)
{
    inferref::IRTensor out = source;
    out.storage_offset = 0;
    out.data.resize(values.size() * inferref::DataTypeSize(out.dtype));
    if (out.dtype == inferref::DataType::kFloat32)
        std::memcpy(out.data.data(), values.data(), out.data.size());
    else if (out.dtype == inferref::DataType::kFloat16)
    {
        auto *dst = reinterpret_cast<std::uint16_t *>(out.data.data());
        for (std::size_t i = 0; i < values.size(); ++i) dst[i] = FloatToHalf(values[i]);
    }
    else if (out.dtype == inferref::DataType::kBFloat16)
    {
        auto *dst = reinterpret_cast<std::uint16_t *>(out.data.data());
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            std::uint32_t bits; std::memcpy(&bits, &values[i], sizeof(bits));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            dst[i] = static_cast<std::uint16_t>(bits >> 16);
        }
    }
    else throw std::runtime_error("SYCL engine supports floating outputs only");
    return out;
}

void WriteManifest(const fs::path &output, const std::vector<std::string> &names)
{
    std::ofstream stream(output / "manifest.json");
    stream << "{\n  \"engine\": \"inferref-sycl\",\n  \"outputs\": [\n";
    for (std::size_t i = 0; i < names.size(); ++i)
        stream << "    {\"name\": \"" << names[i] << "\", \"payload\": \"" << names[i]
               << ".irtensor\"}" << (i + 1 == names.size() ? "\n" : ",\n");
    stream << "  ]\n}\n";
}

void InjectError(const fs::path &path)
{
    auto tensor = inferref::ReadIRTensor(path.string());
    auto values = tensor.AsFloat32();
    if (values.empty()) throw std::runtime_error("cannot inject an error into an empty tensor");
    values[0] += 1.0f;
    inferref::WriteIRTensor(path.string(), EncodeLike(tensor, values));
}

void RunRmsNorm(sycl::queue &queue, const fs::path &root, const fs::path &output)
{
    auto x_tensor = inferref::ReadIRTensor((root / "inputs/x.irtensor").string());
    auto w_tensor = inferref::ReadIRTensor((root / "inputs/weight.irtensor").string());
    auto e_tensor = inferref::ReadIRTensor((root / "inputs/epsilon.irtensor").string());
    auto x = x_tensor.AsFloat32(); auto w = w_tensor.AsFloat32(); const float eps = e_tensor.AsFloat32()[0];
    const std::size_t width = static_cast<std::size_t>(x_tensor.shape.back());
    float *dx = sycl::malloc_shared<float>(x.size(), queue); float *dw = sycl::malloc_shared<float>(w.size(), queue); float *dy = sycl::malloc_shared<float>(x.size(), queue);
    std::copy(x.begin(), x.end(), dx); std::copy(w.begin(), w.end(), dw);
    queue.parallel_for(sycl::range<1>(x.size() / width), [=](sycl::id<1> row) {
        float sum = 0.0f; const std::size_t base = row[0] * width;
        for (std::size_t j = 0; j < width; ++j) sum += dx[base + j] * dx[base + j];
        const float inv = sycl::rsqrt(sum / static_cast<float>(width) + eps);
        for (std::size_t j = 0; j < width; ++j) dy[base + j] = dx[base + j] * inv * dw[j];
    }).wait();
    std::vector<float> y(dy, dy + x.size()); sycl::free(dx, queue); sycl::free(dw, queue); sycl::free(dy, queue);
    inferref::WriteIRTensor((output / "y.irtensor").string(), EncodeLike(x_tensor, y));
    WriteManifest(output, {"y"});
}

std::vector<float> Rope(sycl::queue &queue, const inferref::IRTensor &tensor, const std::vector<float> &cos, const std::vector<float> &sin)
{
    auto x = tensor.AsFloat32(); const std::size_t dim = static_cast<std::size_t>(tensor.shape.back()); const std::size_t half = dim / 2;
    const std::size_t token_count = cos.size() / dim;
    float *dx = sycl::malloc_shared<float>(x.size(), queue); float *dc = sycl::malloc_shared<float>(cos.size(), queue); float *ds = sycl::malloc_shared<float>(sin.size(), queue); float *dy = sycl::malloc_shared<float>(x.size(), queue);
    std::copy(x.begin(), x.end(), dx); std::copy(cos.begin(), cos.end(), dc); std::copy(sin.begin(), sin.end(), ds);
    queue.parallel_for(sycl::range<1>(x.size()), [=](sycl::id<1> id) {
        const std::size_t i = id[0], d = i % dim, token = (i / dim) % token_count;
        const std::size_t rotated = d < half ? i + half : i - half;
        const float sign = d < half ? -1.0f : 1.0f;
        dy[i] = dx[i] * dc[token * dim + d] + sign * dx[rotated] * ds[token * dim + d];
    }).wait();
    std::vector<float> y(dy, dy + x.size());
    for (float *p : {dx, dc, ds, dy}) sycl::free(p, queue);
    return y;
}

void RunRope(sycl::queue &queue, const fs::path &root, const fs::path &output)
{
    auto q = inferref::ReadIRTensor((root / "inputs/query.irtensor").string());
    auto k = inferref::ReadIRTensor((root / "inputs/key.irtensor").string());
    auto c = inferref::ReadIRTensor((root / "inputs/cos.irtensor").string());
    auto s = inferref::ReadIRTensor((root / "inputs/sin.irtensor").string());
    const auto cos = c.AsFloat32(), sin = s.AsFloat32();
    inferref::WriteIRTensor((output / "q_embed.irtensor").string(), EncodeLike(q, Rope(queue, q, cos, sin)));
    inferref::WriteIRTensor((output / "k_embed.irtensor").string(), EncodeLike(k, Rope(queue, k, cos, sin)));
    WriteManifest(output, {"q_embed", "k_embed"});
}

void RunCache(sycl::queue &queue, const fs::path &root, const fs::path &output, bool indexed)
{
    auto cache_tensor = inferref::ReadIRTensor((root / "inputs/cache.irtensor").string());
    auto update_tensor = inferref::ReadIRTensor((root / "inputs/update.irtensor").string());
    auto cache = cache_tensor.AsFloat32(), update = update_tensor.AsFloat32();
    const std::size_t width = static_cast<std::size_t>(cache_tensor.shape.back());
    const std::size_t old_sequence = static_cast<std::size_t>(cache_tensor.shape[cache_tensor.Rank() - 2]);
    const std::size_t update_sequence = static_cast<std::size_t>(update_tensor.shape[update_tensor.Rank() - 2]);
    const std::size_t groups = cache.size() / (old_sequence * width);
    const std::size_t output_sequence = indexed ? old_sequence : old_sequence + update_sequence;
    std::vector<float> result(groups * output_sequence * width);
    float *device_cache = sycl::malloc_shared<float>(cache.size(), queue);
    float *device_update = sycl::malloc_shared<float>(update.size(), queue);
    float *device_result = sycl::malloc_shared<float>(result.size(), queue);
    std::copy(cache.begin(), cache.end(), device_cache);
    std::copy(update.begin(), update.end(), device_update);
    if (indexed)
    {
        auto index = inferref::ReadIRTensor((root / "inputs/index.irtensor").string()).AsFloat32();
        const std::size_t target = static_cast<std::size_t>(index[0]);
        if (target + update_sequence > old_sequence) throw std::runtime_error("cache update index is out of bounds");
        queue.parallel_for(sycl::range<1>(result.size()), [=](sycl::id<1> id) {
            const std::size_t i = id[0];
            const std::size_t group = i / (old_sequence * width);
            const std::size_t within = i % (old_sequence * width);
            const std::size_t token = within / width;
            const std::size_t component = within % width;
            if (token >= target && token < target + update_sequence)
                device_result[i] = device_update[(group * update_sequence + token - target) * width + component];
            else
                device_result[i] = device_cache[i];
        }).wait();
    }
    else
    {
        queue.parallel_for(sycl::range<1>(result.size()), [=](sycl::id<1> id) {
            const std::size_t i = id[0];
            const std::size_t group = i / (output_sequence * width);
            const std::size_t within = i % (output_sequence * width);
            const std::size_t token = within / width;
            const std::size_t component = within % width;
            if (token < old_sequence)
                device_result[i] = device_cache[(group * old_sequence + token) * width + component];
            else
                device_result[i] = device_update[(group * update_sequence + token - old_sequence) * width + component];
        }).wait();
        cache_tensor.shape[cache_tensor.Rank() - 2] += update_tensor.shape[update_tensor.Rank() - 2];
        cache_tensor.stride.assign(cache_tensor.Rank(), 1);
        for (std::size_t i = cache_tensor.Rank() - 1; i > 0; --i) cache_tensor.stride[i - 1] = cache_tensor.stride[i] * cache_tensor.shape[i];
    }
    std::copy(device_result, device_result + result.size(), result.begin());
    sycl::free(device_cache, queue); sycl::free(device_update, queue); sycl::free(device_result, queue);
    inferref::WriteIRTensor((output / "cache_out.irtensor").string(), EncodeLike(cache_tensor, result));
    WriteManifest(output, {"cache_out"});
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        fs::path testcase, output;
        bool inject_error = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--inject-error") inject_error = true;
            else if (argument == "--testcase" && i + 1 < argc) testcase = argv[++i];
            else if (argument == "--output" && i + 1 < argc) output = argv[++i];
            else throw std::runtime_error("unknown or incomplete argument: " + argument);
        }
        if (testcase.empty() || output.empty()) throw std::runtime_error("usage: inferref_sycl_engine --testcase DIR --output DIR");
        fs::create_directories(output);
        sycl::queue queue{sycl::default_selector_v};
        std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        const std::string name = TestcaseName(testcase);
        std::string first_output;
        if (name.rfind("rmsnorm", 0) == 0) { RunRmsNorm(queue, testcase, output); first_output = "y.irtensor"; }
        else if (name.rfind("rope", 0) == 0) { RunRope(queue, testcase, output); first_output = "q_embed.irtensor"; }
        else if (name.rfind("kv-index", 0) == 0) { RunCache(queue, testcase, output, true); first_output = "cache_out.irtensor"; }
        else if (name.rfind("kv-", 0) == 0) { RunCache(queue, testcase, output, false); first_output = "cache_out.irtensor"; }
        else throw std::runtime_error("unsupported testcase name: " + name);
        if (inject_error) InjectError(output / first_output);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
