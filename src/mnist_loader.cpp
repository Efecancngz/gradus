#include "gradus/mnist_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gradus {

MnistDataset load_mnist_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("load_mnist_csv: could not open file: " + path);
    }

    MnistDataset dataset;
    std::string line;
    size_t expected_fields = 0;
    size_t line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::vector<double> fields;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            fields.push_back(std::stod(token));
        }

        if (expected_fields == 0) {
            expected_fields = fields.size();
        } else if (fields.size() != expected_fields) {
            throw std::runtime_error("load_mnist_csv: row " + std::to_string(line_number) +
                                     " has " + std::to_string(fields.size()) +
                                     " fields, expected " + std::to_string(expected_fields));
        }

        dataset.labels.push_back(static_cast<int>(fields[0]));

        std::vector<double> pixels;
        pixels.reserve(fields.size() - 1);
        for (size_t i = 1; i < fields.size(); ++i) {
            pixels.push_back(fields[i] / 127.5 - 1.0);
        }
        dataset.images.push_back(std::move(pixels));
    }

    return dataset;
}

}  // namespace gradus
