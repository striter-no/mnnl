// clang-format off
#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

#include "device.hpp"
#include <buffer/buffer.hpp>

namespace nn::gpu {

enum class BufferType {
    GPU_INPUT,
    GPU_OUTPUT,
    STAGING_FROM_GPU,
    STAGING_TO_GPU,
};

template<class T>
class Buffer {
    friend class ComputeShader;
    friend class ComputePipeline;
    friend class ComputeLayout;

    yst::core::Buffer buf;
    BufferType  type;
    std::size_t size;

public:
    void fill(const std::vector<T> &data) {
        if (type != BufferType::STAGING_TO_GPU)
            throw std::runtime_error("Cannot fill data to buffer, that is not supposed to be filled");

        buf.UploadData(data.data(), data.size() * sizeof(T));
    }

    std::vector<T> fetch() {
        if (type != BufferType::STAGING_FROM_GPU)
            throw std::runtime_error("Cannot fetch data from buffer, that is not supposed to be fetched");

        auto data = static_cast<T*>(buf.MappedData());
        return std::vector<T>(data, data + size);
    }

    Buffer(std::size_t size, BufferType type, GPUDevice &dev):
        size(size), type(type)
    {
        size_t byteSize = size * sizeof(T);
        switch (type) {
        case BufferType::GPU_INPUT: {
            auto gpuCfg = yst::core::BufferBuilder(yst::core::BufferPreset::Empty)
                    .WithSize(byteSize)
                    .WithUsage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                    .Build();
            auto [buf, err] = yst::core::CreateBuffer(dev.device, gpuCfg);

            if (err)
                throw std::runtime_error("Failed to create input ->gpu buffer: " + err.str());

            this->buf = std::move(buf);
        } break;
        case BufferType::GPU_OUTPUT: {
            auto gpuCfg = yst::core::BufferBuilder(yst::core::BufferPreset::Empty)
                    .WithSize(byteSize)
                    .WithUsage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
                    .Build();
            auto [buf, err] = yst::core::CreateBuffer(dev.device, gpuCfg);

            if (err)
                throw std::runtime_error("Failed to create output gpu-> buffer: " + err.str());

            this->buf = std::move(buf);
        } break;
        case BufferType::STAGING_FROM_GPU: {
            auto stagingCfg = yst::core::BufferBuilder(yst::core::BufferPreset::Staging)
                    .WithSize(byteSize)
                    .AddUsageFlags(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                    .Build();
            auto [buf, err] = yst::core::CreateBuffer(dev.device, stagingCfg);

            if (err)
                throw std::runtime_error("Failed to create staging buffer <- GPU: " + err.str());

            this->buf = std::move(buf);
        } break;
        case BufferType::STAGING_TO_GPU: {
            auto [buf, err] = yst::core::CreateStagingBuffer(dev.device, byteSize);

            if (err)
                throw std::runtime_error("Failed to create staging buffer -> GPU: " + err.str());

            this->buf = std::move(buf);
        } break;
        }
    }
    Buffer() = default;
    ~Buffer() = default;
};

}  // namespace nn::gpu
