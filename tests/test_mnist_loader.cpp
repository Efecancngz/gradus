#include <cstdio>
#include <fstream>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradus/mnist_loader.hpp"

namespace {
std::string write_temp_csv(const std::string& name, const std::string& content) {
    std::ofstream out(name);
    out << content;
    out.close();
    return name;
}
}  // namespace

TEST_CASE("load_mnist_csv parses rows into images and labels") {
    std::string path = write_temp_csv(
        "test_mnist_sample1.csv",
        "7,0,255,0,255,0,255,0,255,0\n"
        "3,255,255,255,255,255,255,255,255,255\n");

    auto dataset = gradus::load_mnist_csv(path);

    REQUIRE(dataset.labels.size() == 2);
    REQUIRE(dataset.images.size() == 2);
    REQUIRE(dataset.labels[0] == 7);
    REQUIRE(dataset.labels[1] == 3);
    REQUIRE(dataset.images[0].size() == 9);

    std::remove(path.c_str());
}

TEST_CASE("load_mnist_csv normalizes pixels to [-1, 1]") {
    std::string path = write_temp_csv("test_mnist_sample2.csv", "5,0,255,127\n");
    auto dataset = gradus::load_mnist_csv(path);

    REQUIRE(dataset.images[0][0] == Catch::Approx(-1.0));
    REQUIRE(dataset.images[0][1] == Catch::Approx(1.0));
    REQUIRE(dataset.images[0][2] == Catch::Approx(127.0 / 127.5 - 1.0).epsilon(1e-6));

    std::remove(path.c_str());
}

TEST_CASE("load_mnist_csv throws when the file does not exist") {
    REQUIRE_THROWS_AS(gradus::load_mnist_csv("nonexistent_file_xyz.csv"), std::runtime_error);
}

TEST_CASE("load_mnist_csv throws on a malformed row") {
    // This loader doesn't hardcode 785 fields — it infers the expected
    // field count from the first row, then requires every later row to
    // match. Here the second row has fewer fields than the first.
    std::string path = write_temp_csv(
        "test_mnist_sample3.csv",
        "7,0,255,0\n"
        "3,255,255\n");
    REQUIRE_THROWS_AS(gradus::load_mnist_csv(path), std::runtime_error);
    std::remove(path.c_str());
}
