// clang-format off
#pragma once
#include <string>
#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"
#include "nnmath/mexcp.hpp"

namespace nn::layers {

enum class WeightsInitType {
    HE, XAVIER, UNIFORM
};

template <class T = f32>
class Linear {
public:

    math::Tensor<T> weights; // 2d (in_feats, out_feats)
    math::Tensor<T> bias;    // 1d (out_feats)

    math::Tensor<T> grad_weights;
    math::Tensor<T> grad_bias;

    math::Tensor<T> forward(const math::Tensor<T> &X) {
        saved_X = X;

        math::Tensor<T> X_input = X;
        if (X_input.ndim() == 1) {
            X_input.reshape({1, X_input.dsize()});
        }

        // Y = X * W
        auto Y = X_input.matmul(weights);

        size_t last_dim = Y.get_shape().back();
        if (last_dim != out_feats) {
            throw excp::NotImplemented("Shape mismatch in Linear forward");
        }

        for (size_t i = 0; i < Y.dsize(); i += last_dim) {
            for (size_t j = 0; j < last_dim; j++) {
                Y.data[i + j] += bias.data[j];
            }
        }

        if (X.ndim() == 1) Y.reshape({out_feats});
        return Y;
    }

    math::Tensor<T> backward(const math::Tensor<T> &grad_output){
        math::Tensor<T> grad_2d = grad_output;

        if (grad_2d.ndim() == 1)
            grad_2d.reshape({1, grad_2d.dsize()});

        size_t batch_size_X = saved_X.dsize() / in_feats;
        math::Tensor<T> X_2d = saved_X;
        if (X_2d.ndim() == 1)
            X_2d.reshape({1, in_feats});

        grad_weights = X_2d.transpose().matmul(grad_2d);

        grad_bias = math::Tensor<T>({out_feats});
        grad_bias.fill(0);
        for (size_t i = 0; i < grad_2d.get_shape()[0]; i++) {
            for (size_t j = 0; j < out_feats; j++) {
                grad_bias(j) += grad_2d(i, j);
            }
        }

        auto grad_X = grad_2d.matmul(weights.transpose());

        if (saved_X.ndim() == 1)
            grad_X.reshape({in_feats});
        else
            grad_X.reshape(saved_X.get_shape());

        return grad_X;
    }

    Linear(
        size_t in_feats, size_t out_feats,
        WeightsInitType winit = WeightsInitType::HE
    )
    :   in_feats(in_feats),
        out_feats(out_feats),
        weights({in_feats, out_feats}),
        bias({out_feats}),
        grad_weights({in_feats, out_feats}),
        grad_bias({out_feats})
    {
        switch (winit){
        case WeightsInitType::HE: {
            T stddev = std::sqrt(2.0 / in_feats);
            weights.normal(0, stddev);
        } break;
        case WeightsInitType::XAVIER: {
            T limit = std::sqrt(6.0 / (in_feats + out_feats));
            weights.random(-limit, limit);
        } break;
        case WeightsInitType::UNIFORM: {
            weights.random(-0.5, 0.5);
        } break;
        }
        bias.fill(0.);
    }


    Linear() {}
    ~Linear() = default;

private:

    size_t in_feats;
    size_t out_feats;

    math::Tensor<T> saved_X;
};

}
