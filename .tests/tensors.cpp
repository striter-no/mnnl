#include <iostream>

#include "nnmath/number.hpp"
#include "nnmath/tensor.hpp"

int tensor_test() {
    using namespace nn::math;

    Tensor<f32> t({2, 3});
    t(0, 0) = 1.f;
    t(0, 1) = 2.f;
    t(0, 2) = 3.f;
    t(1, 0) = 3.f;
    t(1, 1) = 5.f;
    t(1, 2) = 7.f;

    std::cout << t.to_string() << std::endl;

    Tensor<f32> t3d({2, 2, 2});
    t3d(0, 0, 0) = 10.0f;
    t3d(1, 1, 1) = 20.0f;

    std::cout << t3d(0, 0, 0) << std::endl;
    std::cout << t3d(1, 1, 1) << std::endl;
    std::cout << t3d.to_string() << std::endl;

    Tensor<f32> t2d_random({3, 4});
    std::cout << t2d_random.random(-1, 1).to_string() << std::endl;

    std::cout << t2d_random.apply([](f32 v) { return v * 2; }).to_string()
              << std::endl;

    std::cout << t2d_random.fill(0).to_string() << std::endl;

    std::cout << "--- N-D Broadcasting MatMul ---" << std::endl;

    Tensor<f32> tA({2, 1, 3, 4});
    tA.fill(1.0f);

    Tensor<f32> tB({1, 2, 4, 5});
    tB.fill(2.0f);

    auto tC = tA.matmul(tB);
    std::cout << "Result shape: ";
    for (size_t s : tC.get_shape()) std::cout << s << " ";
    std::cout << std::endl;  // 2 2 3 5

    std::cout << "Value at (0, 0, 0, 0): " << tC(0, 0, 0, 0) << std::endl;  // 8

    return 0;
}

int main() {
    try {
        return tensor_test();
    } catch (std::exception& ex) {
        std::cerr << "[exception]: " << ex.what() << std::endl;
        return -1;
    }
}
