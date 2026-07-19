// clang-format off
#include <vkgpu/compute.hpp>
#include <spirv_reflect.h>

namespace nn::gpu {

void ComputeShader::load_from_spv(const std::string &path){
    auto [computeSpv, spvErr] = yst::core::LoadSpvFile(path);
    if (spvErr) throw std::runtime_error("Failed to load SPV file: " + spvErr.str());
    this->spv = computeSpv;

    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(spv.size() * 4, spv.data(), &module) == SPV_REFLECT_RESULT_SUCCESS) {
        if (module.entry_point_count > 0) {
            this->localX = module.entry_points[0].local_size.x;
            this->localY = module.entry_points[0].local_size.y;
            this->localZ = module.entry_points[0].local_size.z;
        }
        spvReflectDestroyShaderModule(&module);
    }
}

void ComputeShader::load_from_source(const std::string &path){
    auto [computeSpv, spvErr] = yst::core::LoadAndCompileGlslFile(path, VK_SHADER_STAGE_COMPUTE_BIT);
    if (spvErr) throw std::runtime_error("Failed to load GLSL file: " + spvErr.str());
    this->spv = computeSpv;

    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(spv.size() * 4, spv.data(), &module) == SPV_REFLECT_RESULT_SUCCESS) {
        if (module.entry_point_count > 0) {
            this->localX = module.entry_points[0].local_size.x;
            this->localY = module.entry_points[0].local_size.y;
            this->localZ = module.entry_points[0].local_size.z;
        }
        spvReflectDestroyShaderModule(&module);
    }
}

} // namespace nn::gpu
