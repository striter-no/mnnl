// clang-format off
#pragma once

#include <vulkan/vulkan_core.h>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <shader/shader.hpp>

#include "buffer.hpp"

namespace nn::gpu {

class ComputeShader {
    friend class ComputePipeline;

    std::vector<uint32_t> spv;

    struct BoundBuffer {
        uint32_t binding_id;
        VkBuffer gpuBuffer;
        VkBuffer stagingIn;
        VkBuffer stagingOut;
        size_t byteSize;
    };
    std::vector<BoundBuffer> boundBuffers;

public:
    void load_from_spv(const std::string &path);

    void load_from_source(const std::string &path);

    template<class T>
    void set_bindings(uint32_t binding_id, const Buffer<T> &gpuBuf, const Buffer<T>* stagingBuf = nullptr) {
        BoundBuffer bb;
        bb.binding_id = binding_id;
        bb.gpuBuffer = gpuBuf.buf.buffer;
        bb.byteSize = gpuBuf.size * sizeof(T);
        bb.stagingIn = VK_NULL_HANDLE;
        bb.stagingOut = VK_NULL_HANDLE;

        if (stagingBuf) {
            if (stagingBuf->type == BufferType::STAGING_TO_GPU) bb.stagingIn = stagingBuf->buf.buffer;
            else if (stagingBuf->type == BufferType::STAGING_FROM_GPU) bb.stagingOut = stagingBuf->buf.buffer;
        }
        boundBuffers.push_back(bb);
    }

    ComputeShader() = default;
    ~ComputeShader() = default;
};

}  // namespace nn::gpu
