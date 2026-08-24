#include <catch2/catch_test_macros.hpp>
#include "gradus/nn.hpp"

using gradus::Linear;
using gradus::MLP;
using gradus::Tensor;

TEST_CASE("Linear forward produces the requested output shape") {
    Linear layer(3, 2);
    Tensor x({1.0, 2.0, 3.0}, {1, 3});
    Tensor y = layer.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 2});
}

TEST_CASE("Linear exposes weight and bias as parameters") {
    Linear layer(3, 2);
    auto params = layer.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0].shape() == std::vector<size_t>{3, 2});  // weight
    REQUIRE(params[1].shape() == std::vector<size_t>{1, 2});  // bias
}

TEST_CASE("MLP forward produces the final layer's shape and is differentiable") {
    MLP mlp(2, {4, 1});
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 1});

    y.backward();
    bool any_param_has_nonzero_grad = false;
    for (auto& p : mlp.parameters()) {
        for (double g : p.grad()) {
            if (g != 0.0) {
                any_param_has_nonzero_grad = true;
            }
        }
    }
    REQUIRE(any_param_has_nonzero_grad);
}

TEST_CASE("MLP with activate_output=true (default) bounds its output via tanh") {
    MLP mlp(2, {4, 1});  // default: activate_output = true
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    REQUIRE(y.item() > -1.0);
    REQUIRE(y.item() < 1.0);
}

TEST_CASE("MLP with activate_output=false emits raw, unbounded logits") {
    MLP mlp(2, {4, 3}, /*activate_output=*/false);
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 3});

    y.sum().backward();
    bool any_param_has_nonzero_grad = false;
    for (auto& p : mlp.parameters()) {
        for (double g : p.grad()) {
            if (g != 0.0) any_param_has_nonzero_grad = true;
        }
    }
    REQUIRE(any_param_has_nonzero_grad);
}
