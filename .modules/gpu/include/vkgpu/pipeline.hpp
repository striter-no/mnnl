// clang-format off
#pragma once

#include <functional>
#include <memory>
#include "device.hpp"
#include "compute.hpp"
#include "layout.hpp"
#include <descriptor/descriptor_pool.hpp>
#include <descriptor/bind_group.hpp>
#include <pipeline/compute_pipeline.hpp>
#include <command/command.hpp>

namespace nn::gpu {

class ComputePipeline {
    std::shared_ptr<GPUDevice> gpu;
    ComputeShader shader;
    ComputeLayout layout;

    yst::core::DescriptorPool pool;
    yst::core::BindGroup bg;
    yst::core::ComputePipeline computePipeline;

public:
    void wait_idle();
    void build();
    void submit(uint32_t localSize = 64);

    ComputePipeline &set_layout(ComputeLayout &layout);
    ComputePipeline &set_shader(ComputeShader &shader);
    ComputePipeline &set_gpu(std::shared_ptr<GPUDevice> &device);

    ComputePipeline() = default;
    ~ComputePipeline() = default;
};

}  // namespace nn::gpu
