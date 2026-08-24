#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradient_check.hpp"
#include "gradus/tensor.hpp"

using gradus::Tensor;
using gradus::testutil::numerical_gradient;

TEST_CASE("operator+ forward computes the sum") {
    Tensor a(2.0);
    Tensor b(3.0);
    Tensor c = a + b;
    REQUIRE(c.item() == Catch::Approx(5.0));
}

TEST_CASE("operator+ backward matches numerical gradient") {
    Tensor a(2.0);
    Tensor b(3.0);
    Tensor c = a + b;
    c.backward();

    auto f = [](const std::vector<double>& x) { return x[0] + x[1]; };
    auto numgrad = numerical_gradient(f, {2.0, 3.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(b.grad()[0] == Catch::Approx(numgrad[1]).epsilon(1e-4));
}

TEST_CASE("operator+ throws on shape mismatch") {
    Tensor a({1.0, 2.0}, {2});
    Tensor b({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(a + b, std::invalid_argument);
}

TEST_CASE("operator- forward computes the difference") {
    Tensor a(5.0);
    Tensor b(3.0);
    Tensor c = a - b;
    REQUIRE(c.item() == Catch::Approx(2.0));
}

TEST_CASE("operator- backward matches numerical gradient") {
    Tensor a(5.0);
    Tensor b(3.0);
    Tensor c = a - b;
    c.backward();

    auto f = [](const std::vector<double>& x) { return x[0] - x[1]; };
    auto numgrad = numerical_gradient(f, {5.0, 3.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(b.grad()[0] == Catch::Approx(numgrad[1]).epsilon(1e-4));
}
