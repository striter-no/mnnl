#include <iostream>
#include <vkgpu/buffer.hpp>
#include <vkgpu/compute.hpp>
#include <vkgpu/device.hpp>
#include <vkgpu/layout.hpp>
#include <vkgpu/pipeline.hpp>

using namespace nn::gpu;

int main()
{
    uint32_t count = GPUDevice::get_available_device_count();
    std::cout << "Available GPUs: " << count << std::endl;

    std::vector<std::shared_ptr<GPUDevice>> gpus;
    for (uint32_t i = 0; i < count; ++i) {
        gpus.push_back(std::make_shared<GPUDevice>(
            yst::gpuc::ApiVersion { 1, 3, 0 },
            yst::gpuc::DEFAULT_CONFIG,
            i));
    }

    auto dev = gpus[0];
    std::cout << "Using: " << dev->device.GetDeviceName() << std::endl;

    const size_t dataSize = 512;
    Buffer<float>
        staging_to_gpu(dataSize, BufferType::STAGING_TO_GPU, *dev),
        staging_from_gpu(dataSize, BufferType::STAGING_FROM_GPU, *dev),
        gpu_in(dataSize, BufferType::GPU_INPUT, *dev),
        gpu_out(dataSize, BufferType::GPU_OUTPUT, *dev);

    staging_to_gpu.fill(std::vector<float>(dataSize, 5.f));

    struct PushConsts {
        float multiplier;
        float addend;
    } pc;

    auto layout = ComputeLayout();
    layout.add_buffer(gpu_in, BindingType::ReadOnlyStorage)
        .add_buffer(gpu_out, BindingType::Storage)
        .set_push_constant_size(sizeof(PushConsts));

    auto shader = ComputeShader();
    shader.load_from_source("./assets/compute.comp");
    shader.set_bindings(0, gpu_in, &staging_to_gpu);
    shader.set_bindings(1, gpu_out, &staging_from_gpu);

    auto pipeline = ComputePipeline();
    pipeline.set_gpu(dev)
        .set_layout(layout)
        .set_shader(shader)
        .build();

    pc.multiplier = 2.0f;
    pc.addend = 10.0f;

    pipeline.record(dataSize, 1, 1, &pc);
    pipeline.submit_async();
    pipeline.wait();

    auto data = staging_from_gpu.fetch();

    // 5.0 * 2.0 + 10.0 = 20.0
    std::cout << "Compute result [0]: " << data[0] << std::endl;

    pipeline.wait_idle();
    return 0;
}
