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

    template <class T>
    void record(uint32_t elementsX, uint32_t elementsY = 1, uint32_t elementsZ = 1, const T* pushData = nullptr)
    {
        vkResetCommandBuffer(cmdList.buffer, 0);
        cmdList.Begin(0);

        for (auto& bb : shader.boundBuffers) {
            if (bb.stagingIn) {
                cmdList.CopyBuffer(bb.stagingIn, bb.gpuBuffer, bb.byteSize);
                cmdList.PipelineBarrierBuffer(bb.gpuBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            }
        }

        cmdList.BindPipeline(computePipeline.pipeline, yst::core::PipelineBindPoint::Compute);
        cmdList.BindBindGroup(layout.pipelineLayout.layout, 0, bg, yst::core::PipelineBindPoint::Compute);

        size_t pushConstantSize = sizeof(T);
        if (pushData && pushConstantSize > 0) {
            vkCmdPushConstants(cmdList.buffer, layout.pipelineLayout.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushData);
        }

        if (shader.localX == 0 || shader.localY == 0 || shader.localZ == 0) {
            throw std::runtime_error("Shader localX/Y/Z are equal to 0");
        }

        uint32_t lx = std::max(1u, shader.localX);
        uint32_t ly = std::max(1u, shader.localY);
        uint32_t lz = std::max(1u, shader.localZ);
        cmdList.Dispatch(std::max(1u, (elementsX + lx - 1) / lx),
            std::max(1u, (elementsY + ly - 1) / ly),
            std::max(1u, (elementsZ + lz - 1) / lz));

        for (auto& bb : shader.boundBuffers) {
            if (bb.stagingOut) {
                cmdList.PipelineBarrierBuffer(bb.gpuBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                cmdList.CopyBuffer(bb.gpuBuffer, bb.stagingOut, bb.byteSize);
            }
        }

        cmdList.End();
        isRecorded = true;
    }
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
