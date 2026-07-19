#include <vkgpu/pipeline.hpp>

namespace nn::gpu {

void ComputePipeline::wait_idle()
{
    vkDeviceWaitIdle(gpu->device.LogicalDevice);
}

void ComputePipeline::build()
{
    layout.build(gpu->device);

    auto computePipelineCfg = yst::core::ComputePipelineBuilder()
                                  .WithPipelineLayout(layout.pipelineLayout.layout)
                                  .WithShaderSpv(shader.spv)
                                  .Build();
    auto [cp, cpErr] = yst::core::CreateComputePipeline(gpu->device, computePipelineCfg);
    if (cpErr)
        throw std::runtime_error("Failed to create compute pipeline: " + cpErr.str());
    this->computePipeline = std::move(cp);

    yst::core::DescriptorPoolConfig poolCfg = { .MaxSets = 1 };
    poolCfg.AutoSizeFromLayouts({ &layout.bgl });
    auto [dPool, dPoolErr] = yst::core::CreateDescriptorPool(gpu->device, poolCfg);
    if (dPoolErr)
        throw std::runtime_error("Failed to create descriptor pool");
    this->pool = std::move(dPool);

    auto [cPool, cPoolErr] = yst::core::CreateCommandPool(gpu->device);
    if (cPoolErr)
        throw std::runtime_error("Failed to create command pool");
    this->cmdPool = cPool;

    auto [cmd, allocErr] = yst::core::AllocateCommandList(gpu->device, cmdPool);
    this->cmdList = cmd;

    VkFenceCreateInfo fenceInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(gpu->device.LogicalDevice, &fenceInfo, nullptr, &fence);

    yst::core::BindGroupConfig bindGroupCfg;
    bindGroupCfg.Layout = &layout.bgl;
    for (auto& bb : shader.boundBuffers) {
        bindGroupCfg.Entries.push_back({ .Binding = bb.binding_id,
            .Buffer = bb.gpuBuffer,
            .Range = VK_WHOLE_SIZE });
    }
    auto [bg, bgErr] = yst::core::CreateBindGroup(gpu->device, this->pool, bindGroupCfg);
    if (bgErr)
        throw std::runtime_error("Failed to create bind group: " + bgErr.str());
    this->bg = std::move(bg);
}

void ComputePipeline::submit_async()
{
    if (!isRecorded)
        throw std::runtime_error("Call record() before submit_async()");

    vkResetFences(gpu->device.LogicalDevice, 1, &fence);
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdList.buffer;

    if (vkQueueSubmit(gpu->device.GraphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit queue");
    }
}

void ComputePipeline::wait()
{
    vkWaitForFences(gpu->device.LogicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
}

ComputePipeline& ComputePipeline::set_layout(ComputeLayout& layout)
{
    this->layout = std::move(layout);
    return *this;
}
ComputePipeline& ComputePipeline::set_shader(ComputeShader& shader)
{
    this->shader = std::move(shader);
    return *this;
}
ComputePipeline& ComputePipeline::set_gpu(std::shared_ptr<GPUDevice>& device)
{
    this->gpu = device;
    return *this;
}

ComputePipeline::~ComputePipeline()
{
    destroy();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
{
    *this = std::move(other);
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
{
    if (this != &other) {
        destroy();

        gpu = std::move(other.gpu);
        shader = std::move(other.shader);
        layout = std::move(other.layout);
        pool = std::move(other.pool);
        bg = std::move(other.bg);
        computePipeline = std::move(other.computePipeline);

        cmdList = other.cmdList;
        cmdPool = other.cmdPool;
        fence = other.fence;
        isRecorded = other.isRecorded;

        other.cmdPool = VK_NULL_HANDLE;
        other.fence = VK_NULL_HANDLE;
        other.isRecorded = false;
    }
    return *this;
}

void ComputePipeline::destroy()
{
    if (gpu && gpu->device.LogicalDevice != VK_NULL_HANDLE) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(gpu->device.LogicalDevice, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (cmdPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(gpu->device.LogicalDevice, cmdPool, nullptr);
            cmdPool = VK_NULL_HANDLE;
        }
    }
}

} // namespace nn::gpu
