#pragma once

#include <string>
#include <vector>

#include "nnmath/number.hpp"

namespace nn::data {

std::vector<u8> load_bin_file(const std::string& path);

void save_bin_file(const std::string& path, const std::vector<u8>& data);

}  // namespace nn::data
