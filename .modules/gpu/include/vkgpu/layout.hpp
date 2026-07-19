// clang-format off
#pragma once

#include "buffer.hpp"
#include <descriptor/bind_group_layout.hpp>
#include <descriptor/pipeline_layout.hpp>
#include <vector>

namespace nn::gpu {

class ComputeLayout {
    friend class ComputePipeline;

    struct BindingDesc {
        uint32_t index;
        bool readOnly;
    };
    std::vector<BindingDesc> bindings;

    yst::core::BindGroupLayout bgl;
    yst::core::PipelineLayout pipelineLayout;

public:
    template<class T>
    ComputeLayout &add_buffer(const Buffer<T> &buf) {
        bool readOnly = (buf.type == BufferType::GPU_INPUT);
        bindings.push_back({ (uint32_t)bindings.size(), readOnly });
        return *this;
    }

    void build(yst::core::Device& device);

    ComputeLayout(const ComputeLayout&) = delete;
    ComputeLayout& operator=(const ComputeLayout&) = delete;

    ComputeLayout(ComputeLayout&& other) noexcept;

    ComputeLayout& operator=(ComputeLayout&& other) noexcept;

    ComputeLayout() = default;
    ~ComputeLayout() = default;
};

}  // namespace nn::gpu
