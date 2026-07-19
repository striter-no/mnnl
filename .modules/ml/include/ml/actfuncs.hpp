// clang-format off
#pragma once

#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"

namespace nn::actf {

/**
 * @brief Abstract base class for activation functions.
 *
 * This class defines the interface for activation functions, which must implement `forward` and `derivative` methods.
 */
template<class T = f32>
class ActivateFunction {
public:
    /**
     * @brief Applies the activation function to a tensor.
     *
     * @param t Input tensor.
     * @return Output tensor after applying the activation function.
     */
    virtual math::Tensor<T> forward(const math::Tensor<T> &t) = 0;

    /**
     * @brief Computes the derivative of the activation function with respect to the input.
     *
     * @param grad_y_act Gradient of the loss with respect to the output of the activation function.
     * @param y Raw output tensor before applying the activation function.
     * @return Tensor containing the derivatives.
     */
    virtual math::Tensor<T> derivative(const math::Tensor<T> &grad_y_act, const math::Tensor<T> &y) = 0;

    /**
     * @brief Virtual destructor for derived classes.
     */
    virtual ~ActivateFunction() = default;
};

/**
 * @brief ReLU activation function implementation.
 *
 * This class provides a concrete implementation of the ReLU activation function.
 */
template<class T = f32>
class ReLU: public ActivateFunction<T> {
public:
    /**
     * @brief Applies the ReLU activation function to a tensor.
     *
     * @param t Input tensor.
     * @return Output tensor after applying the ReLU activation function.
     */
    math::Tensor<T> forward(const math::Tensor<T> &t) override {
        return t.apply([](const T &v) {
            return v > 0 ? v : 0;
        });
    }

    /**
     * @brief Computes the derivative of the ReLU activation function with respect to the input.
     *
     * @param grad_y_act Gradient of the loss with respect to the output of the ReLU function.
     * @param y_raw Raw output tensor before applying the ReLU function.
     * @return Tensor containing the derivatives.
     */
    math::Tensor<T> derivative(const math::Tensor<T>& grad_y_act, const math::Tensor<T>& y_raw) override {
        return grad_y_act.apply_idx([&](T v, size_t idx) {
            return y_raw.data[idx] > 0 ? v : 0;
        });
    }

    /**
     * @brief Virtual destructor for derived classes.
     */
    ~ReLU() override = default;
};

/**
 * @brief Sigmoid activation function template class.
 *
 * This class represents a sigmoid activation function that can be used with
 * tensors of type T (defaulting to f32). It inherits from ActivateFunction<T>.
 */
template<class T = f32>
class Sigmoid : public ActivateFunction<T> {
public:
    /**
     * @brief Forward pass through the sigmoid function.
     *
     * Applies the sigmoid function element-wise to the input tensor `t`.
     *
     * @param t Input tensor.
     * @return Tensor containing the result of applying the sigmoid function.
     */
    math::Tensor<T> forward(const math::Tensor<T>& t) {
        return t.apply([](const T& v) {
            return v > 0 ? v : 0;
        });
    }

    /**
     * @brief Computes the derivative of the sigmoid function.
     *
     * Given the gradient of the output `grad_y_act` and the raw output `y_raw`,
     * this method computes the derivative of the sigmoid function element-wise.
     *
     * @param grad_y_act Gradient of the output tensor.
     * @param y_raw Raw output tensor from the forward pass.
     * @return Tensor containing the result of applying the derivative of the
     *         sigmoid function.
     */
    math::Tensor<T> derivative(const math::Tensor<T>& grad_y_act, const math::Tensor<T>& y_raw) {
        return grad_y_act.apply_idx([&](T v, size_t idx) {
            return y_raw.data[idx] > 0 ? v : 0;
        });
    }

    ~Sigmoid() = default;
};

}
