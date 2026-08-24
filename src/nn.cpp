#include "gradus/nn.hpp"

#include <cmath>
#include <random>

namespace gradus {

namespace {
// Every Linear layer gets a distinct-but-deterministic seed: reusing the
// same seed for every layer left correlated initial weights across layers,
// which for a small XOR network failed to break symmetry during training
// (all hidden units stayed functionally identical, so the network could
// only ever output a constant regardless of input). Incrementing the seed
// per layer keeps runs fully reproducible while decorrelating layers.
unsigned int next_linear_seed = 42;
}  // namespace

Linear::Linear(size_t in_features, size_t out_features)
    : weight(std::vector<double>(in_features * out_features, 0.0), {in_features, out_features}),
      bias(std::vector<double>(out_features, 0.0), {1, out_features}) {
    std::mt19937 rng(next_linear_seed++);
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

MLP::MLP(size_t in_features, std::vector<size_t> layer_sizes, bool activate_output)
    : activate_output_(activate_output) {
    size_t prev = in_features;
    for (size_t sz : layer_sizes) {
        layers_.emplace_back(prev, sz);
        prev = sz;
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor out = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        out = layers_[i].forward(out);
        bool is_last_layer = (i == layers_.size() - 1);
        if (!is_last_layer || activate_output_) {
            out = out.tanh();
        }
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
