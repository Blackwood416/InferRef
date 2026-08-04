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

std::string JsonString(std::string value)
{
    std::string escaped;
    for (const char character : value)
    {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

[[noreturn]] void InvalidContract(const std::string &contract,
                                  const std::string &input,
                                  const std::string &expected,
                                  const std::string &actual)
{
    throw std::runtime_error(
        "{\"status\":\"invalid_testcase\",\"contract\":\"" + JsonString(contract) +
        "\",\"input\":\"" + JsonString(input) + "\",\"expected\":\"" +
        JsonString(expected) + "\",\"actual\":\"" + JsonString(actual) + "\"}");
}

void RequireAllocation(const void *pointer, const std::string &name)
{
    if (pointer == nullptr) throw std::runtime_error("SYCL USM allocation failed for " + name);
}

void ValidateAndWriteDeviceEvidence(const fs::path &output, const sycl::queue &queue)
{
    const auto device = queue.get_device();
    const auto backend = queue.get_backend();
    const auto vendor = device.get_info<sycl::info::device::vendor>();
    if (!device.is_gpu()) throw std::runtime_error("selected SYCL device is not a GPU");
    if (vendor.find("Intel") == std::string::npos)
        throw std::runtime_error("selected SYCL GPU is not an Intel device: " + vendor);
    if (backend != sycl::backend::ext_oneapi_level_zero)
        throw std::runtime_error("native Intel XPU execution requires the Level Zero backend");

    std::ofstream stream(output / "inferref-sycl-device.json");
    if (!stream) throw std::runtime_error("cannot write SYCL device evidence");
    stream << "{\n"
           << "  \"format\": \"inferref-sycl-device\",\n"
           << "  \"format_version\": \"0.1\",\n"
           << "  \"backend\": \"ext_oneapi_level_zero\",\n"
           << "  \"device_type\": \"gpu\",\n"
           << "  \"vendor\": \"" << JsonString(vendor) << "\",\n"
           << "  \"name\": \"" << JsonString(device.get_info<sycl::info::device::name>()) << "\",\n"
           << "  \"driver\": \"" << JsonString(device.get_info<sycl::info::device::driver_version>()) << "\",\n"
           << "  \"global_mem_bytes\": " << device.get_info<sycl::info::device::global_mem_size>() << "\n"
           << "}\n";
}

std::string TestcaseContract(const fs::path &root)
{
    const std::string text = ReadText(root / "testcase.json");
    const auto key = text.find("\"contracts\"");
    const auto open = key == std::string::npos ? std::string::npos : text.find('[', key);
    const auto close = open == std::string::npos ? std::string::npos : text.find(']', open);
    if (open == std::string::npos || close == std::string::npos)
        throw std::runtime_error("testcase has no executable contracts array");
    const std::string declared = text.substr(open, close - open + 1);
    const std::vector<std::string> supported = {
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
    };
    std::string selected;
    for (const auto &contract : supported)
        if (declared.find("\"" + contract + "\"") != std::string::npos)
        {
            if (!selected.empty()) throw std::runtime_error("testcase declares multiple executable contracts");
            selected = contract;
        }
    if (selected.empty()) throw std::runtime_error("testcase has no supported executable contract");
    return selected;
}

void WriteU16LE(std::byte *destination, std::uint16_t value)
{
    destination[0] = static_cast<std::byte>(value & 0xffu);
    destination[1] = static_cast<std::byte>((value >> 8) & 0xffu);
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
        for (std::size_t i = 0; i < values.size(); ++i)
            WriteU16LE(out.data.data() + i * 2, inferref::EncodeFloat16(values[i]));
    }
    else if (out.dtype == inferref::DataType::kBFloat16)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
            WriteU16LE(out.data.data() + i * 2, inferref::EncodeBFloat16(values[i]));
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
    const std::string contract = "rmsnorm/last-dim/v1";
    if (x_tensor.Rank() < 1 || x_tensor.Numel() <= 0) InvalidContract(contract, "x", "non-empty rank >= 1", x_tensor.ShapeString());
    if (x_tensor.shape.back() <= 0) InvalidContract(contract, "x", "last dimension > 0", x_tensor.ShapeString());
    const std::size_t width = static_cast<std::size_t>(x_tensor.shape.back());
    if (w_tensor.Numel() != static_cast<std::int64_t>(width))
        InvalidContract(contract, "weight", "numel == x.shape[-1]", w_tensor.ShapeString());
    if (e_tensor.Numel() < 1) InvalidContract(contract, "epsilon", "at least one value", e_tensor.ShapeString());
    if (x_tensor.Numel() % static_cast<std::int64_t>(width) != 0)
        InvalidContract(contract, "x", "numel divisible by last dimension", x_tensor.ShapeString());
    auto x = x_tensor.AsFloat32(); auto w = w_tensor.AsFloat32(); const float eps = e_tensor.AsFloat32()[0];
    float *dx = sycl::malloc_shared<float>(x.size(), queue); float *dw = sycl::malloc_shared<float>(w.size(), queue); float *dy = sycl::malloc_shared<float>(x.size(), queue);
    RequireAllocation(dx, "rmsnorm x"); RequireAllocation(dw, "rmsnorm weight"); RequireAllocation(dy, "rmsnorm output");
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
    RequireAllocation(dx, "rope input"); RequireAllocation(dc, "rope cos"); RequireAllocation(ds, "rope sin"); RequireAllocation(dy, "rope output");
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
    const std::string contract = "rope/rotate-half/v1";
    if (q.Rank() < 2 || q.Numel() <= 0) InvalidContract(contract, "query", "non-empty rank >= 2", q.ShapeString());
    if (k.Rank() < 2 || k.Numel() <= 0) InvalidContract(contract, "key", "non-empty rank >= 2", k.ShapeString());
    const auto dim = q.shape.back();
    if (dim <= 0 || dim % 2 != 0) InvalidContract(contract, "query", "positive even last dimension", q.ShapeString());
    if (k.shape.back() != dim) InvalidContract(contract, "key", "last dimension equal to query", k.ShapeString());
    if (c.Rank() != 2 || s.Rank() != 2 || c.shape != s.shape)
        InvalidContract(contract, "cos/sin", "matching [sequence, rotary_dim] tensors", c.ShapeString() + " / " + s.ShapeString());
    if (c.shape.back() != dim || c.shape[0] <= 0)
        InvalidContract(contract, "cos", "shape [sequence, query.shape[-1]]", c.ShapeString());
    if (q.shape[q.Rank() - 2] != c.shape[0] || k.shape[k.Rank() - 2] != c.shape[0])
        InvalidContract(contract, "query/key", "sequence dimension equal to cos.shape[0]", q.ShapeString() + " / " + k.ShapeString());
    const auto cos = c.AsFloat32(), sin = s.AsFloat32();
    inferref::WriteIRTensor((output / "q_embed.irtensor").string(), EncodeLike(q, Rope(queue, q, cos, sin)));
    inferref::WriteIRTensor((output / "k_embed.irtensor").string(), EncodeLike(k, Rope(queue, k, cos, sin)));
    WriteManifest(output, {"q_embed", "k_embed"});
}

void RunCache(sycl::queue &queue, const fs::path &root, const fs::path &output, bool indexed)
{
    auto cache_tensor = inferref::ReadIRTensor((root / "inputs/cache.irtensor").string());
    auto update_tensor = inferref::ReadIRTensor((root / "inputs/update.irtensor").string());
    const std::string contract = indexed ? "kv-cache/indexed-update/v1" : "kv-cache/append/v1";
    if (cache_tensor.Rank() < 2 || cache_tensor.Numel() <= 0) InvalidContract(contract, "cache", "non-empty rank >= 2", cache_tensor.ShapeString());
    if (update_tensor.Rank() != cache_tensor.Rank() || update_tensor.Numel() <= 0)
        InvalidContract(contract, "update", "non-empty with rank equal to cache", update_tensor.ShapeString());
    for (std::size_t axis = 0; axis < cache_tensor.Rank(); ++axis)
        if (axis != cache_tensor.Rank() - 2 && cache_tensor.shape[axis] != update_tensor.shape[axis])
            InvalidContract(contract, "update", "all non-sequence dimensions equal to cache", update_tensor.ShapeString());
    if (cache_tensor.shape.back() <= 0 || cache_tensor.shape[cache_tensor.Rank() - 2] <= 0 ||
        update_tensor.shape[update_tensor.Rank() - 2] <= 0)
        InvalidContract(contract, "cache/update", "positive sequence and width dimensions", cache_tensor.ShapeString() + " / " + update_tensor.ShapeString());
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
    RequireAllocation(device_cache, "cache input"); RequireAllocation(device_update, "cache update"); RequireAllocation(device_result, "cache output");
    std::copy(cache.begin(), cache.end(), device_cache);
    std::copy(update.begin(), update.end(), device_update);
    if (indexed)
    {
        auto index = inferref::ReadIRTensor((root / "inputs/index.irtensor").string()).AsFloat32();
        if (index.size() != 1 || !std::isfinite(index[0]) || index[0] < 0.0f || std::floor(index[0]) != index[0])
            InvalidContract(contract, "index", "one finite non-negative integer", index.empty() ? "empty" : std::to_string(index[0]));
        const std::size_t target = static_cast<std::size_t>(index[0]);
        if (target + update_sequence > old_sequence)
            InvalidContract(contract, "index", "target + update sequence <= cache sequence", std::to_string(target));
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
        sycl::queue queue{sycl::gpu_selector_v};
        ValidateAndWriteDeviceEvidence(output, queue);
        std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        const std::string contract = TestcaseContract(testcase);
        std::string first_output;
        if (contract == "rmsnorm/last-dim/v1") { RunRmsNorm(queue, testcase, output); first_output = "y.irtensor"; }
        else if (contract == "rope/rotate-half/v1") { RunRope(queue, testcase, output); first_output = "q_embed.irtensor"; }
        else if (contract == "kv-cache/indexed-update/v1") { RunCache(queue, testcase, output, true); first_output = "cache_out.irtensor"; }
        else if (contract == "kv-cache/append/v1") { RunCache(queue, testcase, output, false); first_output = "cache_out.irtensor"; }
        else throw std::runtime_error("unsupported testcase contract: " + contract);
        if (inject_error) InjectError(output / first_output);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
