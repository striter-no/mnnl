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

    void build(yst::core::Device& device) {
        auto bglCfg = yst::core::BindGroupLayoutBuilder(yst::core::BindGroupLayoutPreset::Empty);
        for (auto& b : bindings) {
            bglCfg.AddStorageBuffer(b.index, b.readOnly, yst::core::ShaderStageBits::Compute);
        }
        auto [bgl, bglErr] = yst::core::CreateBindGroupLayout(device, bglCfg.Build());
        if (bglErr)
            throw std::runtime_error("Failed to create bind group layout: " + bglErr.str());

        this->bgl = std::move(bgl);

        auto plCfg = yst::core::PipelineLayoutBuilder(yst::core::PipelineLayoutPreset::Empty)
                         .AddBindGroupLayout(this->bgl)
                         .Build();
        auto [pl, plErr] = yst::core::CreatePipelineLayout(device, plCfg);
        if (plErr)
            throw std::runtime_error("Failed to create pipeline: " + plErr.str());

        this->pipelineLayout = std::move(pl);
    }

    ComputeLayout(const ComputeLayout&) = delete;
    ComputeLayout& operator=(const ComputeLayout&) = delete;

    ComputeLayout(ComputeLayout&& other) noexcept {
        this->bgl = std::move(other.bgl);
        this->pipelineLayout = std::move(other.pipelineLayout);
        this->bindings = std::move(other.bindings);
    }

    ComputeLayout& operator=(ComputeLayout&& other) noexcept {
        this->bgl = std::move(other.bgl);
        this->pipelineLayout = std::move(other.pipelineLayout);
        this->bindings = std::move(other.bindings);

        return *this;
    }

    ComputeLayout() = default;
    ~ComputeLayout() = default;
};

}  // namespace nn::gpu
