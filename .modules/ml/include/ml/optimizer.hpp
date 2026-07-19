// clang-format off
#pragma once
#include "ml/layer.hpp"

namespace nn::optim {

template<class T = f32>
class Optimizer {
public:
    virtual void step(layers::Linear<T>& layer) = 0;
    virtual ~Optimizer() = default;
};

template<class T = f32>
class SGD : public Optimizer<T> {
    T lr;
public:
    SGD(T learning_rate) : lr(learning_rate) {}

    void step(layers::Linear<T>& layer) override {
        for (size_t i = 0; i < layer.weights.get_shape()[0]; ++i)
            for (size_t j = 0; j < layer.weights.get_shape()[1]; ++j)
                layer.weights(i, j) -= lr * layer.grad_weights(i, j);

        for (size_t j = 0; j < layer.bias.get_shape()[0]; ++j)
            layer.bias(j) -= lr * layer.grad_bias(j);
    }
};

} // namespace nn::optim
