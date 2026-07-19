// clang-format off
#include <vkgpu/compute.hpp>

namespace nn::gpu {

void ComputeShader::load_from_spv(const std::string &path){
    auto [computeSpv, spvErr] = yst::core::LoadSpvFile(path);
    if (spvErr) throw std::runtime_error("Failed to load SPV file: " + spvErr.str());
    this->spv = computeSpv;
}

void ComputeShader::load_from_source(const std::string &path){
    auto [computeSpv, spvErr] = yst::core::LoadAndCompileGlslFile(path, VK_SHADER_STAGE_COMPUTE_BIT);
    if (spvErr) throw std::runtime_error("Failed to load GLSL file: " + spvErr.str());
    this->spv = computeSpv;
}

} // namespace nn::gpu
