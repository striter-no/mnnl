// clang-format off
#pragma once
#include <device/config.hpp>
#include <device/device.hpp>

namespace nn::gpu {

class GPUDevice {
public:
    yst::core::Device device;

    GPUDevice(
        yst::gpuc::ApiVersion apiVers = {1, 3, 0},
        yst::gpuc::Preset preset = yst::gpuc::DEFAULT_CONFIG
    ) {
        auto deviceCfg = yst::core::CreateConfig(preset);
        deviceCfg.AppName = "mnnl-vkgpgpu";
        deviceCfg.EngineName = "mnnl-gpu-backend";
        deviceCfg.AppVersion = apiVers;

        auto [device, err] = yst::core::CreateDevice(deviceCfg);
        if (err)
            throw std::runtime_error("Failed to create device: " + err.str());

        this->device = std::move(device);
    }

    GPUDevice(GPUDevice&& other) noexcept {
        this->device = std::move(other.device);
    }

    GPUDevice& operator=(GPUDevice&& other) noexcept {
        this->device = std::move(other.device);

        return *this;
    }

    GPUDevice(const GPUDevice&) = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;
    ~GPUDevice() = default;
};

}  // namespace nn::gpu
