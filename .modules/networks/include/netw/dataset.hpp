// clang-format off
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"
namespace nn {

template<class T = f32>
struct DatasetEntry {
    math::Tensor<T> input;
    math::Tensor<T> true_output;
};

template<class T = f32>
class Dataset {
    std::vector<DatasetEntry<T>> data;

public:

    void add_entry(
        const math::Tensor<T> &input,
        const math::Tensor<T> &output
    ){
        data.push_back({input, output});
    }

    const DatasetEntry<T> batch(size_t n = 1, bool shuffle = false) {
        if (data.empty()) {
            throw std::runtime_error("Dataset is empty");
        }

        if (n == 0 || n > data.size()) {
            n = data.size();
        }

        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < data.size(); ++i) indices[i] = i;

        if (shuffle) {
            static std::mt19937 mt(std::random_device{}());
            std::shuffle(indices.begin(), indices.end(), mt);
        }

        indices.resize(n);

        std::vector<size_t> in_shape = data[0].input.get_shape();
        std::vector<size_t> out_shape = data[0].true_output.get_shape();

        std::vector<size_t> batched_in_shape = in_shape;
        batched_in_shape.insert(batched_in_shape.begin(), n); // [n, ...]

        std::vector<size_t> batched_out_shape = out_shape;
        batched_out_shape.insert(batched_out_shape.begin(), n); // [n, ...]

        math::Tensor<T> batched_input(batched_in_shape);
        math::Tensor<T> batched_output(batched_out_shape);

        size_t in_size = data[0].input.dsize();
        size_t out_size = data[0].true_output.dsize();

        for (size_t i = 0; i < n; ++i) {
            size_t idx = indices[i];

            for (size_t j = 0; j < in_size; ++j) {
                batched_input.data[i * in_size + j] = data[idx].input.data[j];
            }

            for (size_t j = 0; j < out_size; ++j) {
                batched_output.data[i * out_size + j] = data[idx].true_output.data[j];
            }
        }

        return {batched_input, batched_output};
    }

    const DatasetEntry<T> &random() {
        static std::mt19937 mt(std::random_device{}());
        std::uniform_int_distribution<size_t> range(0, data.size() - 1);

        size_t inx = range(mt);
        return data[inx];
    }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }

    Dataset() {}
    ~Dataset() = default;
};

}  // namespace nn
