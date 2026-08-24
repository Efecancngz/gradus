#include "gradus/optim.hpp"

#include <stdexcept>

namespace gradus {

SGD::SGD(std::vector<Tensor> parameters, double learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::step() {
    for (auto& p : parameters_) {
        for (size_t i = 0; i < p.size(); ++i) {
            const_cast<std::vector<double>&>(p.data())[i] -= learning_rate_ * p.grad()[i];
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : parameters_) {
        p.zero_grad();
    }
}

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("mse_loss: prediction and target shapes differ");
    }
    Tensor diff = prediction - target;
    Tensor squared = diff * diff;
    Tensor total = squared.sum();
    double n = static_cast<double>(prediction.size());
    return total * Tensor(1.0 / n);
}

}  // namespace gradus
