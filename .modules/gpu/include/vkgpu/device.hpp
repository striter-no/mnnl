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
    );

    GPUDevice(GPUDevice&& other) noexcept;

    GPUDevice& operator=(GPUDevice&& other) noexcept;

    GPUDevice(const GPUDevice&) = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;
    ~GPUDevice() = default;
};

}  // namespace nn::gpu
