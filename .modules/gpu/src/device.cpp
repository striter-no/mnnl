#include <vkgpu/device.hpp>

namespace nn::gpu {

GPUDevice::GPUDevice(yst::gpuc::ApiVersion apiVers, yst::gpuc::Preset preset)
{
    auto deviceCfg = yst::core::CreateConfig(preset);
    deviceCfg.AppName = "mnnl-vkgpgpu";
    deviceCfg.EngineName = "mnnl-gpu-backend";
    deviceCfg.AppVersion = apiVers;

    auto [device, err] = yst::core::CreateDevice(deviceCfg);
    if (err)
        throw std::runtime_error("Failed to create device: " + err.str());

    this->device = std::move(device);
}

GPUDevice::GPUDevice(GPUDevice&& other) noexcept
{
    this->device = std::move(other.device);
}

GPUDevice& GPUDevice::operator=(GPUDevice&& other) noexcept
{
    this->device = std::move(other.device);

    return *this;
}

} // namespace nn::gpu
