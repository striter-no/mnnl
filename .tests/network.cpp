#include <iomanip>
#include <iostream>
#include <memory>
#include <netw/network.hpp>

#include "ml/actfuncs.hpp"
#include "ml/lossfs.hpp"
#include "ml/optimizer.hpp"
#include "netw/dataset.hpp"

int network_test() {
    using namespace nn;

    Dataset ds;
    ds.add_entry(math::Tensor<f32>::create({0, 1}),
                 math::Tensor<f32>::create({1}));

    ds.add_entry(math::Tensor<f32>::create({0, 0}),
                 math::Tensor<f32>::create({0}));

    ds.add_entry(math::Tensor<f32>::create({1, 0}),
                 math::Tensor<f32>::create({1}));

    ds.add_entry(math::Tensor<f32>::create({1, 1}),
                 math::Tensor<f32>::create({0}));

    LinearNetwork net;
    net.add_layer(layers::Linear(2, 4), std::make_shared<actf::ReLU<f32>>());
    net.add_layer(layers::Linear(4, 1));

    loss::MSE loss_fn;
    optim::SGD optimizer(0.01f);

    for (int epoch = 0; epoch < 100000; epoch++) {
        auto entry = ds.batch(0, true);
        auto y_pred = net.forward(entry.input);

        auto loss_v = loss_fn.forward(y_pred, entry.true_output);
        if (epoch % 1000 == 0)
            std::cout << " - epoch " << epoch
                      << " [loss: " << std::setprecision(8) << loss_v << "]\n";

        if (loss_v < 0.01) {
            std::cout << "[stopped, loss is low enough]" << std::endl;
            break;
        }

        auto grad = loss_fn.backward(y_pred, entry.true_output);
        net.backward(grad);

        net.update_weights(optimizer);
    }

    for (auto& entry : ds) {
        std::cout << "[test]\n input: " << entry.input.to_string() << std::endl;

        auto y_pred = net.forward(entry.input);
        std::cout << "output: " << y_pred.to_string() << std::endl;
    }

    return 0;
}

int main() {
    try {
        return network_test();
    } catch (std::exception& ex) {
        std::cerr << "[exception]: " << ex.what() << std::endl;
        return -1;
    }
}
