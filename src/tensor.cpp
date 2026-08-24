#include "gradus/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace gradus {

TensorImpl::TensorImpl(std::vector<double> data_, std::vector<size_t> shape_)
    : data(std::move(data_)), shape(std::move(shape_)) {
    grad.assign(data.size(), 0.0);
}

Tensor::Tensor(double value)
    : impl(std::make_shared<TensorImpl>(std::vector<double>{value}, std::vector<size_t>{1})) {}

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape)
    : impl(std::make_shared<TensorImpl>(std::move(data), std::move(shape))) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl_) : impl(std::move(impl_)) {}

size_t Tensor::size() const { return impl->data.size(); }

const std::vector<size_t>& Tensor::shape() const { return impl->shape; }

const std::vector<double>& Tensor::data() const { return impl->data; }

std::vector<double>& Tensor::grad() { return impl->grad; }

double Tensor::item() const {
    if (size() != 1) {
        throw std::invalid_argument("item() requires a single-element tensor");
    }
    return impl->data[0];
}

void Tensor::zero_grad() { std::fill(impl->grad.begin(), impl->grad.end(), 0.0); }

void Tensor::backward() {
    if (size() != 1) {
        throw std::invalid_argument(
            "backward() can only be called on a scalar (single-element) tensor");
    }

    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::unordered_set<TensorImpl*> visited;

    std::function<void(const std::shared_ptr<TensorImpl>&)> build_topo =
        [&](const std::shared_ptr<TensorImpl>& node) {
            if (visited.count(node.get())) {
                return;
            }
            visited.insert(node.get());
            for (const auto& parent : node->parents) {
                build_topo(parent);
            }
            topo.push_back(node);
        };
    build_topo(impl);

    impl->grad[0] = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator+");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] + other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += out_raw->grad[i];
            rhs->grad[i] += out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator-");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] - other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += out_raw->grad[i];
            rhs->grad[i] -= out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator*");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] * other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += rhs->data[i] * out_raw->grad[i];
            rhs->grad[i] += lhs->data[i] * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (impl->shape.size() != 2 || other.impl->shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    size_t m = impl->shape[0];
    size_t k = impl->shape[1];
    if (other.impl->shape[0] != k) {
        throw std::invalid_argument("matmul shape mismatch: inner dimensions differ");
    }
    size_t n = other.impl->shape[1];

    std::vector<double> result_data(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double total = 0.0;
            for (size_t p = 0; p < k; ++p) {
                total += impl->data[i * k + p] * other.impl->data[p * n + j];
            }
            result_data[i * n + j] = total;
        }
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, std::vector<size_t>{m, n});
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw, m, k, n]() {
        // dL/dlhs = dL/dout @ rhs^T
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                double grad_sum = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    grad_sum += out_raw->grad[i * n + j] * rhs->data[p * n + j];
                }
                lhs->grad[i * k + p] += grad_sum;
            }
        }
        // dL/drhs = lhs^T @ dL/dout
        for (size_t p = 0; p < k; ++p) {
            for (size_t j = 0; j < n; ++j) {
                double grad_sum = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    grad_sum += lhs->data[i * k + p] * out_raw->grad[i * n + j];
                }
                rhs->grad[p * n + j] += grad_sum;
            }
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::tanh() const {
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = std::tanh(impl->data[i]);
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            double t = out_raw->data[i];
            input->grad[i] += (1.0 - t * t) * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::relu() const {
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] > 0.0 ? impl->data[i] : 0.0;
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            input->grad[i] += (input->data[i] > 0.0 ? 1.0 : 0.0) * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::sum() const {
    double total = 0.0;
    for (double v : impl->data) {
        total += v;
    }

    auto out_impl =
        std::make_shared<TensorImpl>(std::vector<double>{total}, std::vector<size_t>{1});
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            input->grad[i] += out_raw->grad[0];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::softmax_cross_entropy_loss(int target_class) const {
    if (impl->shape.size() != 2 || impl->shape[0] != 1) {
        throw std::invalid_argument("softmax_cross_entropy_loss requires a {1, C} logits tensor");
    }
    size_t num_classes = impl->shape[1];
    if (target_class < 0 || static_cast<size_t>(target_class) >= num_classes) {
        throw std::invalid_argument("softmax_cross_entropy_loss: target_class out of range");
    }

    double max_logit = impl->data[0];
    for (size_t i = 1; i < num_classes; ++i) {
        max_logit = std::max(max_logit, impl->data[i]);
    }

    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(impl->data[i] - max_logit);
    }

    double log_sum_exp = std::log(sum_exp) + max_logit;
    double loss_value = log_sum_exp - impl->data[static_cast<size_t>(target_class)];

    auto out_impl =
        std::make_shared<TensorImpl>(std::vector<double>{loss_value}, std::vector<size_t>{1});
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw, target_class, max_logit, sum_exp, num_classes]() {
        for (size_t i = 0; i < num_classes; ++i) {
            double softmax_i = std::exp(input->data[i] - max_logit) / sum_exp;
            double indicator = (static_cast<int>(i) == target_class) ? 1.0 : 0.0;
            input->grad[i] += (softmax_i - indicator) * out_raw->grad[0];
        }
    };

    return Tensor(out_impl);
}

}  // namespace gradus
