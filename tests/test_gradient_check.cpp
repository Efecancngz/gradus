#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradient_check.hpp"

TEST_CASE("numerical_gradient matches a known analytic derivative") {
    // f(x) = x^2 + 3x  =>  df/dx = 2x + 3
    auto f = [](const std::vector<double>& x) {
        return x[0] * x[0] + 3.0 * x[0];
    };
    auto grad = gradus::testutil::numerical_gradient(f, {2.0});
    REQUIRE(grad[0] == Catch::Approx(2.0 * 2.0 + 3.0).epsilon(1e-4));
}

TEST_CASE("numerical_gradient handles multiple inputs independently") {
    // f(x, y) = x*y  =>  df/dx = y, df/dy = x
    auto f = [](const std::vector<double>& v) {
        return v[0] * v[1];
    };
    auto grad = gradus::testutil::numerical_gradient(f, {3.0, 7.0});
    REQUIRE(grad[0] == Catch::Approx(7.0).epsilon(1e-4));
    REQUIRE(grad[1] == Catch::Approx(3.0).epsilon(1e-4));
}
