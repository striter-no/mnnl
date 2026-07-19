// clang-format off
#include <vkgpu/layout.hpp>

namespace nn::gpu {

void ComputeLayout::build(yst::core::Device& device) {
    auto bglCfg = yst::core::BindGroupLayoutBuilder(yst::core::BindGroupLayoutPreset::Empty);
    for (auto& b : bindings) {
        bglCfg.AddStorageBuffer(b.index, b.readOnly, yst::core::ShaderStageBits::Compute);
    }
    auto [bgl, bglErr] = yst::core::CreateBindGroupLayout(device, bglCfg.Build());
    if (bglErr)
        throw std::runtime_error("Failed to create bind group layout: " + bglErr.str());

    this->bgl = std::move(bgl);

    auto plCfg = yst::core::PipelineLayoutBuilder(yst::core::PipelineLayoutPreset::Empty)
                        .AddBindGroupLayout(this->bgl)
                        .Build();
    auto [pl, plErr] = yst::core::CreatePipelineLayout(device, plCfg);
    if (plErr)
        throw std::runtime_error("Failed to create pipeline: " + plErr.str());

    this->pipelineLayout = std::move(pl);
}

ComputeLayout::ComputeLayout(ComputeLayout&& other) noexcept {
    this->bgl = std::move(other.bgl);
    this->pipelineLayout = std::move(other.pipelineLayout);
    this->bindings = std::move(other.bindings);
}

ComputeLayout& ComputeLayout::operator=(ComputeLayout&& other) noexcept {
    this->bgl = std::move(other.bgl);
    this->pipelineLayout = std::move(other.pipelineLayout);
    this->bindings = std::move(other.bindings);

    return *this;
}

}  // namespace nn::gpu
