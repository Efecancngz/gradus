#pragma once

#include <string>
#include <vector>

namespace gradus {

struct MnistDataset {
    std::vector<std::vector<double>> images;  // normalized to [-1, 1]
    std::vector<int> labels;
};

MnistDataset load_mnist_csv(const std::string& path);

}  // namespace gradus
