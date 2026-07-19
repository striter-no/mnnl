// clang-format off
#pragma once
#include "nnmath/tensor.hpp"

namespace nn::loss {

template<class T = f32>
class Loss {
public:
    virtual T forward(const math::Tensor<T>& y_pred, const math::Tensor<T>& y_true) = 0;
    virtual math::Tensor<T> backward(const math::Tensor<T>& y_pred, const math::Tensor<T>& y_true) = 0;

    virtual ~Loss() = default;
};

template<class T = f32>
class MSE : public Loss<T> {
public:
    T forward(const math::Tensor<T>& y_pred, const math::Tensor<T>& y_true) override {
        if (y_pred.dsize() != y_true.dsize()) {
            throw std::invalid_argument("MSE: size mismatch");
        }

        T sum = 0;
        T n = y_pred.dsize();

        for (size_t i = 0; i < n; ++i) {
            T diff = y_pred.data[i] - y_true.data[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    math::Tensor<T> backward(const math::Tensor<T>& y_pred, const math::Tensor<T>& y_true) override {
        if (y_pred.dsize() != y_true.dsize()) {
            throw std::invalid_argument("MSE grad: size mismatch");
        }

        math::Tensor<T> grad(y_pred.get_shape());
        T n = y_pred.dsize();

        for (size_t i = 0; i < n; ++i) {
            grad.data[i] = 2.0f * (y_pred.data[i] - y_true.data[i]) / n;
        }
        return grad;
    }
};

} // namespace nn::lossf
