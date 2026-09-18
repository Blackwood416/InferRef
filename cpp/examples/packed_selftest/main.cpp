#include <iostream>
#include "inferref/irtensor.hpp"

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv)
{
    try
    {
        for (auto dtype : {inferref::DataType::kInt4, inferref::DataType::kUInt4})
        {
            Require(inferref::DataTypeBits(dtype) == 4, "four bit elements");
            Require(inferref::PayloadSize(dtype, 3) == 2, "odd payload size");
            for (int byte = 0; byte < 256; ++byte)
            {
                std::vector<std::byte> wire(65, std::byte{0});
                std::memcpy(wire.data(), "IRTN", 4);
                wire[4] = std::byte{2}; wire[6] = std::byte{64};
                wire[8] = static_cast<std::byte>(dtype);
                wire[10] = std::byte{1}; wire[14] = std::byte{1};
                wire[18] = std::byte{2}; wire[26] = std::byte{1};
                wire[34] = std::byte{2}; wire[42] = std::byte{1};
                wire[64] = static_cast<std::byte>(byte);
                auto tensor = inferref::DecodeIRTensor(wire.data(), wire.size());
                auto values = tensor.AsFloat32();
                for (int i = 0; i < 2; ++i)
                {
                    int expected = i ? byte / 16 : byte % 16;
                    if (dtype == inferref::DataType::kInt4 && expected >= 8) expected -= 16;
                    Require(values[i] == expected, "exhaustive nibble decoding");
                }
                wire[18] = std::byte{1}; wire[34] = std::byte{1};
                bool rejected = false;
                try { inferref::DecodeIRTensor(wire.data(), wire.size()); }
                catch (const std::exception &) { rejected = true; }
                Require(rejected == (byte >= 16), "odd padding rejection");
                wire[4] = std::byte{1};
                rejected = false;
                try { inferref::DecodeIRTensor(wire.data(), wire.size()); }
                catch (const std::exception &) { rejected = true; }
                Require(rejected, "packed dtype cannot use legacy format");
            }
        }
        if (argc == 3)
        {
            const auto tensor = inferref::ReadIRTensor(argv[1]);
            Require(inferref::IsPacked(tensor.dtype), "packed golden input");
            inferref::WriteIRTensor(argv[2], tensor);
        }
        std::cout << "INT4/UINT4: all 512 byte pairs and padding/version gates passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
