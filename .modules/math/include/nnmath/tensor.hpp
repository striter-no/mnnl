// clang-format off
#pragma once

#include <random>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <array>

#include "nnmath/number.hpp"
#include "nnmath/mexcp.hpp"

namespace nn::math {

template<class T = f32>
class Tensor {
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;

        // row-major
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

public:

    std::vector<T> data;

    size_t ndim() const { return shape.size(); }
    size_t dsize() const { return data.size(); }

    const std::vector<size_t>& get_shape() const { return shape; }

    // utils --

    void reshape(std::vector<size_t> new_shape) {
        size_t old_size = 1;
        for (size_t s : shape) old_size *= s;

        size_t new_size = 1;
        for (size_t s : new_shape) new_size *= s;

        if (old_size != new_size) {
            throw std::invalid_argument("Cannot reshape: total size mismatch");
        }
        shape = std::move(new_shape);
        compute_strides();
    }

    constexpr size_t rank() const { return shape.size(); }

    // 1d quick fill
    Tensor &fill(const std::vector<T> &d) {
        if (rank() != 1 && !((shape[0] == 1 || shape[1] == 1) && shape.size() == 2))
            throw std::invalid_argument("Cannot vector-fill 1D: Tensor is not 1D");

        if (d.size() != data.size())
            throw std::invalid_argument("Cannot vector-fill 1D: size mismatch");

        data = d;
        return *this;
    }

    // 2d quick fill
    Tensor &fill(const std::vector<std::vector<T>> &d) {
        if (rank() != 2)
            throw std::invalid_argument("Cannot vector-fill 2D: Tensor is not 2D");

        if (d.size() != shape[1])
            throw std::invalid_argument("Rows mismatch");
        if (d[0].size() != shape[0])
            throw std::invalid_argument("Cols mismatch");

        for (size_t r = 0; r < shape[1]; ++r) {
            for (size_t c = 0; c < shape[0]; ++c) {
                (*this)(c, r) = d[r][c];
            }
        }
        return *this;
    }

    // 1d quick creation
    static Tensor create(const std::vector<T> &d) {
        Tensor o({d.size()});
        o.fill(d);
        return o;
    }

    // 2d quick creation
    static Tensor create(const std::vector<std::vector<T>> &d) {
        Tensor o({d[0].size(), d.size()});
        o.fill(d);

        return o;
    }

    Tensor &fill(const T &v) { for (auto &e: data) { e = v; } return *this; }

    Tensor &random(const T &min, const T &max) {
        static std::mt19937 mt(std::random_device{}());

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> range(min, max);
            for (auto &e: data) { e = range(mt); }
        } else {
            std::uniform_real_distribution<T> range(min, max);
            for (auto &e: data) { e = range(mt); }
        }

        return *this;
    }

    Tensor &normal(const T &mean, const T &stddev) {
        static std::mt19937 mt(std::random_device{}());

        if constexpr (std::is_floating_point_v<T>) {
            std::normal_distribution<T> range(mean, stddev);
            for (auto &e: data) { e = range(mt); }
        } else {
            throw excp::NotImplemented("Normal distribution requires floating point type");
        }

        return *this;
    }

    template <class Func>
    Tensor apply(Func&& f) const {
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = f(data[i]);
        }
        return out;
    }

    template <class Func>
    Tensor apply_idx(Func&& f) const {
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = f(data[i], i);
        }
        return out;
    }

    // math --

    Tensor operator*(const T &scalar) const {
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] * scalar;
        }
        return out;
    }

    Tensor operator+(const T &scalar) const {
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] + scalar;
        }
        return out;
    }

    // Hadamard mul
    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Hadamard product: shapes must match");
        }
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] * other.data[i];
        }
        return out;
    }

    Tensor operator+(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Tensor sum: shapes must match");
        }
        Tensor out(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            out.data[i] = data[i] + other.data[i];
        }
        return out;
    }

    Tensor transpose() const {
        if (ndim() != 2) throw std::invalid_argument("Tensor transpose is only for 2D tensors");
        Tensor out({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }

    Tensor matmul(const Tensor& other) const {
        if (ndim() < 2 || other.ndim() < 2) {
            throw std::invalid_argument("matmul requires at least 2D tensors");
        }

        if (shape[ndim() - 1] != other.shape[other.ndim() - 2]) {
            throw std::invalid_argument("matmul: inner dimensions must match");
        }

        std::vector<size_t> batch_shape_A(shape.begin(), shape.end() - 2);
        std::vector<size_t> batch_shape_B(other.shape.begin(), other.shape.end() - 2);

        size_t batch_dims = std::max(batch_shape_A.size(), batch_shape_B.size());

        while (batch_shape_A.size() < batch_dims) batch_shape_A.insert(batch_shape_A.begin(), 1);
        while (batch_shape_B.size() < batch_dims) batch_shape_B.insert(batch_shape_B.begin(), 1);

        std::vector<size_t> out_batch_shape(batch_dims);
        for (size_t i = 0; i < batch_dims; ++i) {
            if (batch_shape_A[i] == batch_shape_B[i]) {
                out_batch_shape[i] = batch_shape_A[i];
            } else if (batch_shape_A[i] == 1) {
                out_batch_shape[i] = batch_shape_B[i];
            } else if (batch_shape_B[i] == 1) {
                out_batch_shape[i] = batch_shape_A[i];
            } else {
                throw std::invalid_argument("Broadcasting mismatch in batch dimensions");
            }
        }

        size_t R = shape[ndim() - 2];
        size_t C = other.shape[other.ndim() - 1];
        std::vector<size_t> out_shape = out_batch_shape;
        out_shape.push_back(R);
        out_shape.push_back(C);

        Tensor out(out_shape);

        size_t total_batches = 1;
        for (size_t s : out_batch_shape) total_batches *= s;

        size_t K = shape[ndim() - 1];

        for (size_t b = 0; b < total_batches; ++b) {
            std::vector<size_t> coords(batch_dims);
            size_t temp = b;
            for (int i = batch_dims - 1; i >= 0; --i) {
                coords[i] = temp % out_batch_shape[i];
                temp /= out_batch_shape[i];
            }

            size_t offset_A = 0;
            size_t offset_B = 0;
            for (size_t i = 0; i < batch_dims; ++i) {
                size_t coord_A = (batch_shape_A[i] == 1) ? 0 : coords[i];
                size_t coord_B = (batch_shape_B[i] == 1) ? 0 : coords[i];

                offset_A += coord_A * strides[i];
                offset_B += coord_B * other.strides[i];
            }

            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k) {
                        sum += data[offset_A + r * K + k] * other.data[offset_B + k * C + c];
                    }
                    out.data[b * R * C + r * C + c] = sum;
                }
            }
        }

        return out;
    }

    // --

    std::string to_string() const {
        std::stringstream ss;
        if (shape.size() == 1) {
            ss << "[";
            for (size_t i = 0; i < shape[0]; ++i) {
                ss << std::fixed << std::setprecision(4) << data[i];
                if (i < shape[0] - 1) ss << ", ";
            }
            ss << "]";
        } else if (shape.size() == 2) {
            ss << "[";
            for (size_t r = 0; r < shape[0]; ++r) {
                if (r != 0) ss << " ";
                ss << "[";
                for (size_t c = 0; c < shape[1]; ++c) {
                    ss << std::fixed << std::setprecision(4) << (*this)(r, c);
                    if (c < shape[1] - 1) ss << ", ";
                }
                ss << "]";
                if (r != shape[0] - 1) ss << "\n";
            }
            ss << "]";
        } else {
            ss << "[Tensor with shape: ";
            for (size_t s : shape) ss << s << " ";
            ss << "]";
        }
        return ss.str();
    }

    Tensor(std::vector<size_t> shape_) : shape(std::move(shape_)) {
        compute_strides();
        size_t total_size = 1;
        for (size_t s : shape) total_size *= s;
        data.resize(total_size, 0.0f);
    }

    Tensor() {}

    template <class... Indices>
    T& operator()(Indices... indices) {
        std::array<size_t, sizeof...(Indices)> idx_array = {static_cast<size_t>(indices)...};
        if (idx_array.size() != shape.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }

        size_t flat_index = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (idx_array[i] >= shape[i]) throw std::out_of_range("Index out of bounds");
            flat_index += idx_array[i] * strides[i];
        }
        return data[flat_index];
    }

    template <class... Indices>
    const T& operator()(Indices... indices) const {
        std::array<size_t, sizeof...(Indices)> idx_array = {static_cast<size_t>(indices)...};
        if (idx_array.size() != shape.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }

        size_t flat_index = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (idx_array[i] >= shape[i]) throw std::out_of_range("Index out of bounds");
            flat_index += idx_array[i] * strides[i];
        }
        return data[flat_index];
    }

};

} // namespace nn::math
