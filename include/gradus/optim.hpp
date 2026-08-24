#pragma once

#include <vector>
#include "gradus/tensor.hpp"

namespace gradus {

class SGD {
public:
    SGD(std::vector<Tensor> parameters, double learning_rate);

    void step();
    void zero_grad();

private:
    std::vector<Tensor> parameters_;
    double learning_rate_;
};

Tensor mse_loss(const Tensor& prediction, const Tensor& target);

}  // namespace gradus
