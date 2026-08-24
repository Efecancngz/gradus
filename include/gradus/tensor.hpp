#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace gradus {

struct TensorImpl {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<size_t> shape;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_fn;

    TensorImpl(std::vector<double> data_, std::vector<size_t> shape_);
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    explicit Tensor(double value);
    Tensor(std::vector<double> data, std::vector<size_t> shape);
    explicit Tensor(std::shared_ptr<TensorImpl> impl_);

    size_t size() const;
    const std::vector<size_t>& shape() const;
    const std::vector<double>& data() const;
    std::vector<double>& grad();
    double item() const;

    void backward();
    void zero_grad();

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor tanh() const;
    Tensor relu() const;
    Tensor sum() const;
    Tensor softmax_cross_entropy_loss(int target_class) const;
};

}  // namespace gradus
