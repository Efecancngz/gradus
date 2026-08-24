#include <cmath>
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

TEST_CASE("operator* forward computes the elementwise product") {
    Tensor a(4.0);
    Tensor b(5.0);
    Tensor c = a * b;
    REQUIRE(c.item() == Catch::Approx(20.0));
}

TEST_CASE("operator* backward matches numerical gradient") {
    Tensor a(4.0);
    Tensor b(5.0);
    Tensor c = a * b;
    c.backward();

    auto f2 = [](const std::vector<double>& x) { return x[0] * x[1]; };
    auto numgrad2 = numerical_gradient(f2, {4.0, 5.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad2[0]).epsilon(1e-4));
    REQUIRE(b.grad()[0] == Catch::Approx(numgrad2[1]).epsilon(1e-4));
}

TEST_CASE("operator* throws on shape mismatch") {
    Tensor a({1.0, 2.0}, {2});
    Tensor b({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(a * b, std::invalid_argument);
}

TEST_CASE("matmul forward computes the matrix product") {
    Tensor a({1.0, 2.0, 3.0, 4.0}, {2, 2});  // [[1,2],[3,4]]
    Tensor b({5.0, 6.0, 7.0, 8.0}, {2, 2});  // [[5,6],[7,8]]
    Tensor c = a.matmul(b);                  // [[19,22],[43,50]]

    REQUIRE(c.shape() == std::vector<size_t>{2, 2});
    REQUIRE(c.data()[0] == Catch::Approx(19.0));
    REQUIRE(c.data()[1] == Catch::Approx(22.0));
    REQUIRE(c.data()[2] == Catch::Approx(43.0));
    REQUIRE(c.data()[3] == Catch::Approx(50.0));
}

TEST_CASE("matmul backward matches numerical gradient") {
    // sum() doesn't exist until Task 8, so this test reduces the {1,2}
    // output to a scalar by matmul-ing against a {2,1} column of ones —
    // multiplying by a ones column is itself just a sum, expressed as a
    // matmul.
    Tensor a({1.0, 2.0}, {1, 2});
    Tensor b({3.0, 4.0, 5.0, 6.0}, {2, 2});
    Tensor c = a.matmul(b);
    Tensor ones_col({1.0, 1.0}, {2, 1});
    Tensor loss = c.matmul(ones_col);  // shape {1,1}: sums c's two entries
    loss.backward();

    auto f = [](const std::vector<double>& x) {
        double a0 = x[0], a1 = x[1];
        double b00 = x[2], b01 = x[3], b10 = x[4], b11 = x[5];
        double c0v = a0 * b00 + a1 * b10;
        double c1v = a0 * b01 + a1 * b11;
        return c0v + c1v;
    };
    auto numgrad = numerical_gradient(f, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(a.grad()[1] == Catch::Approx(numgrad[1]).epsilon(1e-4));
    REQUIRE(b.grad()[0] == Catch::Approx(numgrad[2]).epsilon(1e-4));
    REQUIRE(b.grad()[1] == Catch::Approx(numgrad[3]).epsilon(1e-4));
    REQUIRE(b.grad()[2] == Catch::Approx(numgrad[4]).epsilon(1e-4));
    REQUIRE(b.grad()[3] == Catch::Approx(numgrad[5]).epsilon(1e-4));
}

TEST_CASE("matmul throws on inner dimension mismatch") {
    Tensor a({1.0, 2.0}, {1, 2});
    Tensor b({1.0, 2.0, 3.0}, {3, 1});
    REQUIRE_THROWS_AS(a.matmul(b), std::invalid_argument);
}

TEST_CASE("tanh forward matches std::tanh") {
    Tensor a(0.5);
    Tensor b = a.tanh();
    REQUIRE(b.item() == Catch::Approx(std::tanh(0.5)));
}

TEST_CASE("tanh backward matches numerical gradient") {
    Tensor a(0.5);
    Tensor b = a.tanh();
    b.backward();

    auto f = [](const std::vector<double>& x) { return std::tanh(x[0]); };
    auto numgrad = numerical_gradient(f, {0.5});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
}

TEST_CASE("relu forward zeroes negative inputs") {
    Tensor a({-2.0, 3.0}, {2});
    Tensor b = a.relu();
    REQUIRE(b.data()[0] == Catch::Approx(0.0));
    REQUIRE(b.data()[1] == Catch::Approx(3.0));
}

TEST_CASE("relu backward matches numerical gradient at a positive input") {
    Tensor a(2.0);
    Tensor b = a.relu();
    b.backward();

    auto f = [](const std::vector<double>& x) { return x[0] > 0.0 ? x[0] : 0.0; };
    auto numgrad = numerical_gradient(f, {2.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
}

TEST_CASE("sum forward adds all elements") {
    Tensor a({1.0, 2.0, 3.0, 4.0}, {4});
    Tensor b = a.sum();
    REQUIRE(b.item() == Catch::Approx(10.0));
}

TEST_CASE("sum backward broadcasts gradient 1.0 to every element") {
    Tensor a({1.0, 2.0, 3.0}, {3});
    Tensor b = a.sum();
    b.backward();

    REQUIRE(a.grad()[0] == Catch::Approx(1.0));
    REQUIRE(a.grad()[1] == Catch::Approx(1.0));
    REQUIRE(a.grad()[2] == Catch::Approx(1.0));
}
