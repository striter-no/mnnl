// clang-format off
#include <vkgpu/layout.hpp>

namespace nn::gpu {

void ComputeLayout::build(yst::core::Device& device) {
    auto bglCfg = yst::core::BindGroupLayoutBuilder();
    for (auto& b : bindings) {
        if (b.type == BindingType::Storage) bglCfg.AddStorageBuffer(b.index, false, yst::core::ShaderStageBits::Compute);
        else if (b.type == BindingType::ReadOnlyStorage) bglCfg.AddStorageBuffer(b.index, true, yst::core::ShaderStageBits::Compute);
        else if (b.type == BindingType::Uniform) bglCfg.AddUniformBuffer(b.index, yst::core::ShaderStageBits::Compute);
    }

    auto [bgl, bglErr] = yst::core::CreateBindGroupLayout(device, bglCfg.Build());
    if (bglErr) throw std::runtime_error("Failed to create bind group layout: " + bglErr.str());
    this->bgl = std::move(bgl);

    auto plBuilder = yst::core::PipelineLayoutBuilder(yst::core::PipelineLayoutPreset::Empty)
                        .AddBindGroupLayout(this->bgl);

    if (pushConstantSize > 0) {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = pushConstantSize;
        plBuilder.AddPushConstantRange(pcr);
    }

    auto [pl, plErr] = yst::core::CreatePipelineLayout(device, plBuilder.Build());
    if (plErr) throw std::runtime_error("Failed to create pipeline: " + plErr.str());
    this->pipelineLayout = std::move(pl);
}

ComputeLayout::ComputeLayout(ComputeLayout&& other) noexcept {
    this->bgl = std::move(other.bgl);
    this->pipelineLayout = std::move(other.pipelineLayout);
    this->bindings = std::move(other.bindings);
    this->pushConstantSize = other.pushConstantSize;
    other.pushConstantSize = 0;
}

ComputeLayout& ComputeLayout::operator=(ComputeLayout&& other) noexcept {
    if (this != &other) {
        this->bgl = std::move(other.bgl);
        this->pipelineLayout = std::move(other.pipelineLayout);
        this->bindings = std::move(other.bindings);
        this->pushConstantSize = other.pushConstantSize;
        other.pushConstantSize = 0;
    }
    return *this;
}

}  // namespace nn::gpu
