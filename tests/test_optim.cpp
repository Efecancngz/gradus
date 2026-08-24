#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

using gradus::Linear;
using gradus::mse_loss;
using gradus::SGD;
using gradus::Tensor;

TEST_CASE("mse_loss is zero when prediction equals target") {
    Tensor pred({1.0, 2.0}, {1, 2});
    Tensor target({1.0, 2.0}, {1, 2});
    Tensor loss = mse_loss(pred, target);
    REQUIRE(loss.item() == Catch::Approx(0.0));
}

TEST_CASE("mse_loss computes mean squared error") {
    Tensor pred({3.0}, {1, 1});
    Tensor target({1.0}, {1, 1});
    Tensor loss = mse_loss(pred, target);
    // (3-1)^2 / 1 = 4.0
    REQUIRE(loss.item() == Catch::Approx(4.0));
}

TEST_CASE("SGD single step reduces loss on a toy linear problem") {
    Linear layer(1, 1);
    // Force a known, deterministic starting point instead of the random init.
    const_cast<std::vector<double>&>(layer.weight.data())[0] = 0.0;
    const_cast<std::vector<double>&>(layer.bias.data())[0] = 0.0;

    Tensor x({2.0}, {1, 1});
    Tensor target({4.0}, {1, 1});

    SGD optimizer(layer.parameters(), 0.1);

    Tensor pred1 = layer.forward(x);
    double loss1 = mse_loss(pred1, target).item();

    optimizer.zero_grad();
    Tensor loss1_tensor = mse_loss(layer.forward(x), target);
    loss1_tensor.backward();
    optimizer.step();

    Tensor pred2 = layer.forward(x);
    double loss2 = mse_loss(pred2, target).item();

    REQUIRE(loss2 < loss1);
}
