// clang-format off
#pragma once
#include <vector>
#include <memory>
#include "ml/actfuncs.hpp"
#include "ml/layer.hpp"
#include "ml/optimizer.hpp"
#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"

namespace nn {

template<class T = f32>
struct NetLayer {
    layers::Linear<T> layer;
    std::shared_ptr<actf::ActivateFunction<T>> act;
};

template<class T = f32>
struct LayerCache {
    math::Tensor<T> x;
    math::Tensor<T> y_raw;
};

template<class T = f32>
class LinearNetwork {
    std::vector<NetLayer<T>> layers;
    std::vector<LayerCache<T>> cache;

public:
    math::Tensor<T> forward(const math::Tensor<T> &X) {
        cache.clear();
        math::Tensor<T> x = X;

        for (auto &l : layers) {
            LayerCache<T> c;
            c.x = x;
            c.y_raw = l.layer.forward(x);

            if (l.act) x = l.act->forward(c.y_raw);
            else  x = c.y_raw;

            cache.push_back(c);
        }
        return x;
    }

    void backward(const math::Tensor<T> &grad_y) {
        math::Tensor<T> current_grad = grad_y;

        for (int i = layers.size() - 1; i >= 0; --i) {
            auto grad_raw = (layers[i].act)
                        ? layers[i].act->derivative(current_grad, cache[i].y_raw)
                        : current_grad;
            current_grad = layers[i].layer.backward(grad_raw);
        }
    }

    void add_layer(layers::Linear<T> l, const std::shared_ptr<actf::ActivateFunction<T>> &act) {
        layers.push_back(NetLayer<T>{l, act});
    }

    void add_layer(layers::Linear<T> l) {
        layers.push_back(NetLayer<T>{l, nullptr});
    }

    void update_weights(optim::Optimizer<T>& optimizer) {
        for (auto &l : layers) {
            optimizer.step(l.layer);
        }
    }
};

}  // namespace nn
