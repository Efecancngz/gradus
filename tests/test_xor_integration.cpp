#include <catch2/catch_test_macros.hpp>

#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

using gradus::MLP;
using gradus::mse_loss;
using gradus::SGD;
using gradus::Tensor;

namespace {

std::vector<std::vector<double>> xor_inputs() {
    return {{-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}};
}

std::vector<double> xor_targets() { return {-1.0, 1.0, 1.0, -1.0}; }

double average_loss(const MLP& mlp) {
    auto inputs = xor_inputs();
    auto targets = xor_targets();
    double total = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor x(inputs[i], {1, 2});
        Tensor y(std::vector<double>{targets[i]}, {1, 1});
        Tensor pred = mlp.forward(x);
        total += mse_loss(pred, y).item();
    }
    return total / static_cast<double>(inputs.size());
}

}  // namespace

TEST_CASE("MLP trained on XOR substantially reduces average loss") {
    MLP mlp(2, {4, 1});
    SGD optimizer(mlp.parameters(), 0.1);

    double initial_loss = average_loss(mlp);

    auto inputs = xor_inputs();
    auto targets = xor_targets();
    for (int epoch = 0; epoch < 300; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor x(inputs[i], {1, 2});
            Tensor y(std::vector<double>{targets[i]}, {1, 1});
            Tensor pred = mlp.forward(x);
            Tensor loss = mse_loss(pred, y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }

    double final_loss = average_loss(mlp);
    REQUIRE(final_loss < initial_loss * 0.5);
}
