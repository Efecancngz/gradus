#pragma once

#include <vector>
#include "gradus/tensor.hpp"

namespace gradus {

class Linear {
public:
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> parameters() const;

    Tensor weight;
    Tensor bias;
};

class MLP {
public:
    MLP(size_t in_features, std::vector<size_t> layer_sizes);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> parameters() const;

private:
    std::vector<Linear> layers_;
};

}  // namespace gradus
