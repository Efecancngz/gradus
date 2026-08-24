#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "gradient_check.hpp"
#include "gradus/tensor.hpp"

using gradus::Tensor;
using gradus::testutil::numerical_gradient;

TEST_CASE("backward accumulates gradient across a diamond graph (a used twice)") {
    Tensor a(3.0);
    Tensor b = a * a;  // b = a^2
    Tensor c = b + a;  // c = a^2 + a
    c.backward();

    // dc/da = 2a + 1
    auto f = [](const std::vector<double>& x) { return x[0] * x[0] + x[0]; };
    auto numgrad = numerical_gradient(f, {3.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
}

TEST_CASE("backward works through a composite graph mixing matmul, tanh, and sum") {
    Tensor x({1.0, -2.0}, {1, 2});
    Tensor w({0.5, -0.5, 1.0, 2.0}, {2, 2});
    Tensor y = x.matmul(w).tanh().sum();
    y.backward();

    auto f = [](const std::vector<double>& v) {
        double x0 = v[0], x1 = v[1];
        double w00 = v[2], w01 = v[3], w10 = v[4], w11 = v[5];
        double c0 = std::tanh(x0 * w00 + x1 * w10);
        double c1 = std::tanh(x0 * w01 + x1 * w11);
        return c0 + c1;
    };
    auto numgrad = numerical_gradient(f, {1.0, -2.0, 0.5, -0.5, 1.0, 2.0});

    REQUIRE(x.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(x.grad()[1] == Catch::Approx(numgrad[1]).epsilon(1e-4));
    REQUIRE(w.grad()[0] == Catch::Approx(numgrad[2]).epsilon(1e-4));
    REQUIRE(w.grad()[1] == Catch::Approx(numgrad[3]).epsilon(1e-4));
    REQUIRE(w.grad()[2] == Catch::Approx(numgrad[4]).epsilon(1e-4));
    REQUIRE(w.grad()[3] == Catch::Approx(numgrad[5]).epsilon(1e-4));
}
