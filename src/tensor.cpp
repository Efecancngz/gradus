#include "gradus/tensor.hpp"

#include <algorithm>
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

size_t Tensor::size() const {
    return impl->data.size();
}

const std::vector<size_t>& Tensor::shape() const {
    return impl->shape;
}

const std::vector<double>& Tensor::data() const {
    return impl->data;
}

std::vector<double>& Tensor::grad() {
    return impl->grad;
}

double Tensor::item() const {
    if (size() != 1) {
        throw std::invalid_argument("item() requires a single-element tensor");
    }
    return impl->data[0];
}

void Tensor::zero_grad() {
    std::fill(impl->grad.begin(), impl->grad.end(), 0.0);
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::invalid_argument("backward() can only be called on a scalar (single-element) tensor");
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

}  // namespace gradus
