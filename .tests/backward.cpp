#include <iostream>

#include "ml/actfuncs.hpp"
#include "ml/layer.hpp"
#include "ml/lossfs.hpp"
#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"

int backward_test() {
    using namespace nn;

    math::Tensor<f32> x({1, 3});
    x(0, 0) = 1;
    x(0, 1) = 2;
    x(0, 2) = 3;

    math::Tensor<f32> y_true({1, 2});
    y_true(0, 0) = 3;
    y_true(0, 1) = 1;

    layers::Linear<f32> layer(3, 2);
    auto y = layer.forward(x);
    auto y_act = actf::ReLU(y);

    auto loss_y = loss::mse(y_act, y_true);
    auto grad_y_act = loss::mse_grad(y_act, y_true);

    auto grad_y = actf::ReLU_derv(grad_y_act, y);
    auto grad_x = layer.backward(grad_y);

    std::cout << grad_x.to_string() << std::endl;

    return 0;
}

int main() {
    try {
        return backward_test();
    } catch (std::exception& ex) {
        std::cerr << "[exception]: " << ex.what() << std::endl;
        return -1;
    }
}
