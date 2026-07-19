// clang-format off
#include <vkgpu/pipeline.hpp>

namespace nn::gpu {

void ComputePipeline::wait_idle() {
    vkDeviceWaitIdle(gpu->device.LogicalDevice);
}

void ComputePipeline::build() {
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
    auto [p, poolErr] = yst::core::CreateDescriptorPool(gpu->device, poolCfg);
    if (poolErr) throw std::runtime_error("Pool err: " + poolErr.str());
    this->pool = std::move(p);

    yst::core::BindGroupConfig bindGroupCfg;
    bindGroupCfg.Layout = &layout.bgl;
    for (auto& bb : shader.boundBuffers) {
        bindGroupCfg.Entries.push_back({
            .Binding = bb.binding_id,
            .Buffer = bb.gpuBuffer,
            .Range = VK_WHOLE_SIZE
        });
    }
    auto [bg, bgErr] = yst::core::CreateBindGroup(gpu->device, pool, bindGroupCfg);
    if (bgErr)
        throw std::runtime_error("Failed to create bind group: " + bgErr.str());

    this->bg = std::move(bg);
}

void ComputePipeline::submit(uint32_t localSize) {
    uint32_t totalElements = 0;
    if (!shader.boundBuffers.empty()) {
        totalElements = shader.boundBuffers[0].byteSize / sizeof(float);
    }
    uint32_t groupCount = (totalElements + localSize - 1) / localSize;

    auto submitErr = yst::core::SubmitOneTimeCommands(
        gpu->device, [&](yst::core::CommandList& cmd) -> yst::CustomError {
            cmd.Begin();

            for (auto& bb : shader.boundBuffers) {
                if (bb.stagingIn != VK_NULL_HANDLE) {
                    cmd.CopyBuffer(bb.stagingIn, bb.gpuBuffer, bb.byteSize);
                    cmd.PipelineBarrierBuffer(
                        bb.gpuBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
                }
            }

            cmd.BindPipeline(computePipeline.pipeline, yst::core::PipelineBindPoint::Compute);
            cmd.BindBindGroup(layout.pipelineLayout.layout, 0, bg, yst::core::PipelineBindPoint::Compute);
            cmd.Dispatch(groupCount, 1, 1);

            for (auto& bb : shader.boundBuffers) {
                if (bb.stagingOut != VK_NULL_HANDLE) {
                    cmd.PipelineBarrierBuffer(
                        bb.gpuBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT);
                    cmd.CopyBuffer(bb.gpuBuffer, bb.stagingOut, bb.byteSize);
                }
            }

            cmd.End();
            return yst::CustomError();
        });

    if (submitErr) throw std::runtime_error("Submit err: " + submitErr.str());
}

ComputePipeline &ComputePipeline::set_layout(ComputeLayout &layout) {
    this->layout = std::move(layout);
    return *this;
}

ComputePipeline &ComputePipeline::set_shader(ComputeShader &shader) {
    this->shader = std::move(shader);
    return *this;
}

ComputePipeline &ComputePipeline::set_gpu(std::shared_ptr<GPUDevice> &device) {
    gpu = device;
    return *this;
}

}  // namespace nn::gpu
