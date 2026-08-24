#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gradus/tensor.hpp"

TEST_CASE("scalar constructor produces a size-1 tensor") {
    gradus::Tensor a(5.0);
    REQUIRE(a.size() == 1);
    REQUIRE(a.shape() == std::vector<size_t>{1});
    REQUIRE(a.item() == Catch::Approx(5.0));
}

TEST_CASE("vector constructor produces the requested shape") {
    gradus::Tensor a({1.0, 2.0, 3.0, 4.0}, {2, 2});
    REQUIRE(a.size() == 4);
    REQUIRE(a.shape() == std::vector<size_t>{2, 2});
}

TEST_CASE("item() throws on a non-scalar tensor") {
    gradus::Tensor a({1.0, 2.0}, {2});
    REQUIRE_THROWS_AS(a.item(), std::invalid_argument);
}

TEST_CASE("backward() seeds gradient to 1.0 on a scalar leaf") {
    gradus::Tensor a(5.0);
    a.backward();
    REQUIRE(a.grad()[0] == Catch::Approx(1.0));
}

TEST_CASE("backward() throws on a non-scalar tensor") {
    gradus::Tensor a({1.0, 2.0}, {2});
    REQUIRE_THROWS_AS(a.backward(), std::invalid_argument);
}

TEST_CASE("zero_grad() resets gradient to zero") {
    gradus::Tensor a(5.0);
    a.backward();
    REQUIRE(a.grad()[0] == Catch::Approx(1.0));
    a.zero_grad();
    REQUIRE(a.grad()[0] == Catch::Approx(0.0));
}
