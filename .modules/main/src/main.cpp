#include <iostream>
#include <vkgpu/buffer.hpp>
#include <vkgpu/compute.hpp>
#include <vkgpu/device.hpp>
#include <vkgpu/layout.hpp>
#include <vkgpu/pipeline.hpp>

using namespace nn::gpu;

int main()
{
    auto dev = std::make_shared<GPUDevice>();
    std::cout << dev->device.GetDeviceName() << std::endl;

    const size_t dataSize = 512;
    Buffer<float>
        staging_to_gpu(dataSize, BufferType::STAGING_TO_GPU, *dev),
        staging_from_gpu(dataSize, BufferType::STAGING_FROM_GPU, *dev),
        gpu_in(dataSize, BufferType::GPU_INPUT, *dev),
        gpu_out(dataSize, BufferType::GPU_OUTPUT, *dev);

    staging_to_gpu.fill(std::vector(dataSize, 5.f));

    auto layout = ComputeLayout();
    layout.add_buffer(gpu_in)
        .add_buffer(gpu_out);

    auto shader = ComputeShader();
    shader.load_from_source("./assets/compute.comp");
    shader.set_bindings(0, gpu_in, &staging_to_gpu);
    shader.set_bindings(1, gpu_out, &staging_from_gpu);

    auto pipeline = ComputePipeline();
    pipeline.set_gpu(dev)
        .set_layout(layout)
        .set_shader(shader)
        .build();

    pipeline.submit(); // blocks thread until data from GPU is transfered to staging_from_gpu

    auto data = staging_from_gpu.fetch();
    std::cout << "Compute result [0]: " << data[0] << std::endl;

    pipeline.wait_idle();
    return 0;
}
