// clang-format off
#pragma once

#include "buffer.hpp"
#include <descriptor/bind_group_layout.hpp>
#include <descriptor/pipeline_layout.hpp>
#include <vector>

namespace nn::gpu {

enum class BindingType {
    Storage,
    Uniform,
    ReadOnlyStorage
};

class ComputeLayout {
    friend class ComputePipeline;
    struct BindingDesc {
        uint32_t index;
        BindingType type;
    };
    std::vector<BindingDesc> bindings;

    yst::core::BindGroupLayout bgl;
    yst::core::PipelineLayout pipelineLayout;
    uint32_t pushConstantSize = 0;

public:
    template<class T>
    ComputeLayout& add_buffer(const Buffer<T>& buf, BindingType type = BindingType::Storage) {
        bindings.push_back({ (uint32_t)bindings.size(), type });
        return *this;
    }

    ComputeLayout& set_push_constant_size(uint32_t size) {
        pushConstantSize = size;
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
