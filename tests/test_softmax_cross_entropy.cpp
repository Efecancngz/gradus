#include <algorithm>
#include <cmath>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradient_check.hpp"
#include "gradus/tensor.hpp"

using gradus::Tensor;
using gradus::testutil::numerical_gradient;

TEST_CASE("softmax_cross_entropy_loss forward matches the closed-form definition") {
    Tensor logits({2.0, 1.0, 0.1}, {1, 3});
    Tensor loss = logits.softmax_cross_entropy_loss(0);

    double max_logit = 2.0;
    double sum_exp = std::exp(2.0 - max_logit) + std::exp(1.0 - max_logit) + std::exp(0.1 - max_logit);
    double expected = (std::log(sum_exp) + max_logit) - 2.0;

    REQUIRE(loss.item() == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("softmax_cross_entropy_loss backward matches numerical gradient") {
    Tensor logits({2.0, 1.0, 0.1}, {1, 3});
    Tensor loss = logits.softmax_cross_entropy_loss(1);
    loss.backward();

    auto f = [](const std::vector<double>& x) {
        double max_logit = *std::max_element(x.begin(), x.end());
        double sum_exp = 0.0;
        for (double v : x) sum_exp += std::exp(v - max_logit);
        return (std::log(sum_exp) + max_logit) - x[1];
    };
    auto numgrad = numerical_gradient(f, {2.0, 1.0, 0.1});

    REQUIRE(logits.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(logits.grad()[1] == Catch::Approx(numgrad[1]).epsilon(1e-4));
    REQUIRE(logits.grad()[2] == Catch::Approx(numgrad[2]).epsilon(1e-4));
}

TEST_CASE("softmax_cross_entropy_loss backward equals softmax minus one-hot") {
    Tensor logits({1.0, 2.0, 3.0}, {1, 3});
    Tensor loss = logits.softmax_cross_entropy_loss(2);
    loss.backward();

    double max_logit = 3.0;
    double sum_exp = std::exp(1.0 - max_logit) + std::exp(2.0 - max_logit) + std::exp(3.0 - max_logit);
    double softmax0 = std::exp(1.0 - max_logit) / sum_exp;
    double softmax1 = std::exp(2.0 - max_logit) / sum_exp;
    double softmax2 = std::exp(3.0 - max_logit) / sum_exp;

    REQUIRE(logits.grad()[0] == Catch::Approx(softmax0).epsilon(1e-6));
    REQUIRE(logits.grad()[1] == Catch::Approx(softmax1).epsilon(1e-6));
    REQUIRE(logits.grad()[2] == Catch::Approx(softmax2 - 1.0).epsilon(1e-6));
}

TEST_CASE("softmax_cross_entropy_loss throws on out-of-range target_class") {
    Tensor logits({1.0, 2.0}, {1, 2});
    REQUIRE_THROWS_AS(logits.softmax_cross_entropy_loss(5), std::invalid_argument);
    REQUIRE_THROWS_AS(logits.softmax_cross_entropy_loss(-1), std::invalid_argument);
}

TEST_CASE("softmax_cross_entropy_loss throws on non-row-vector shape") {
    Tensor logits({1.0, 2.0, 3.0, 4.0}, {2, 2});
    REQUIRE_THROWS_AS(logits.softmax_cross_entropy_loss(0), std::invalid_argument);
}
