#pragma once
#include "compute.hpp"
#include "device.hpp"
#include "layout.hpp"
#include <command/command.hpp>
#include <descriptor/bind_group.hpp>
#include <descriptor/descriptor_pool.hpp>
#include <functional>
#include <memory>
#include <pipeline/compute_pipeline.hpp>

namespace nn::gpu {

class ComputePipeline {
    std::shared_ptr<GPUDevice> gpu;
    ComputeShader shader;
    ComputeLayout layout;

    yst::core::DescriptorPool pool;
    yst::core::BindGroup bg;
    yst::core::ComputePipeline computePipeline;

    yst::core::CommandList cmdList;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool isRecorded = false;

public:
    void wait_idle();
    void build();
    void record(uint32_t elementsX, uint32_t elementsY = 1, uint32_t elementsZ = 1, const void* pushData = nullptr);
    void submit_async();
    void wait();

    ComputePipeline& set_layout(ComputeLayout& layout);
    ComputePipeline& set_shader(ComputeShader& shader);
    ComputePipeline& set_gpu(std::shared_ptr<GPUDevice>& device);

    ComputePipeline();
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;

    void destroy();
};

} // namespace nn::gpu
