#include "gradus/nn.hpp"

#include <cmath>
#include <random>

namespace gradus {

Linear::Linear(size_t in_features, size_t out_features)
    : weight(std::vector<double>(in_features * out_features, 0.0), {in_features, out_features}),
      bias(std::vector<double>(out_features, 0.0), {1, out_features}) {
    // Fixed seed: every training run (and every test) is reproducible.
    std::mt19937 rng(42);
    double bound = 1.0 / std::sqrt(static_cast<double>(in_features));
    std::uniform_real_distribution<double> dist(-bound, bound);

    for (size_t i = 0; i < weight.size(); ++i) {
        const_cast<std::vector<double>&>(weight.data())[i] = dist(rng);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        const_cast<std::vector<double>&>(bias.data())[i] = dist(rng);
    }
}

Tensor Linear::forward(const Tensor& x) const {
    return x.matmul(weight) + bias;
}

std::vector<Tensor> Linear::parameters() const {
    return {weight, bias};
}

MLP::MLP(size_t in_features, std::vector<size_t> layer_sizes) {
    size_t prev = in_features;
    for (size_t sz : layer_sizes) {
        layers_.emplace_back(prev, sz);
        prev = sz;
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor out = x;
    for (const auto& layer : layers_) {
        out = layer.forward(out).tanh();
    }
    return out;
}

std::vector<Tensor> MLP::parameters() const {
    std::vector<Tensor> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

}  // namespace gradus
