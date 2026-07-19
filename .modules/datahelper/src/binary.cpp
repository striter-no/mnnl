#include <data/binary.hpp>
#include <fstream>
#include <iterator>

namespace nn::data {

std::vector<u8> load_bin_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    std::vector<u8> buffer(std::istreambuf_iterator<char>(input), {});

    return buffer;
}

void save_bin_file(const std::string& path, const std::vector<u8>& data) {
    std::ofstream output(path, std::ios::binary);
    for (auto& byte : data) {
        output << byte;
    }
}

}  // namespace nn::data
