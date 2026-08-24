#include <chrono>
#include <iostream>

#include "gradus/mnist_loader.hpp"
#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

namespace {

int argmax(const std::vector<double>& v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = i;
    }
    return static_cast<int>(best);
}

}  // namespace

int main() {
    using namespace gradus;
    using Clock = std::chrono::steady_clock;

    std::cout << "Loading MNIST (run scripts/download_mnist.sh first if this fails)...\n";
    auto train = load_mnist_csv("data/mnist_train.csv");
    auto test = load_mnist_csv("data/mnist_test.csv");
    std::cout << "Loaded " << train.images.size() << " train / " << test.images.size()
              << " test examples.\n";

    MLP mlp(784, {128, 10}, /*activate_output=*/false);
    SGD optimizer(mlp.parameters(), 0.01);

    const int epochs = 5;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = Clock::now();
        double total_loss = 0.0;

        for (size_t i = 0; i < train.images.size(); ++i) {
            Tensor x(train.images[i], {1, 784});
            Tensor logits = mlp.forward(x);
            Tensor loss = logits.softmax_cross_entropy_loss(train.labels[i]);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item();
        }

        auto epoch_seconds = std::chrono::duration<double>(Clock::now() - epoch_start).count();
        std::cout << "epoch " << epoch << "  avg loss " << (total_loss / train.images.size())
                  << "  (" << epoch_seconds << "s)\n";
    }

    std::cout << "\nEvaluating on test set...\n";
    int correct = 0;
    for (size_t i = 0; i < test.images.size(); ++i) {
        Tensor x(test.images[i], {1, 784});
        Tensor logits = mlp.forward(x);
        if (argmax(logits.data()) == test.labels[i]) {
            ++correct;
        }
    }
    double accuracy =
        100.0 * static_cast<double>(correct) / static_cast<double>(test.images.size());
    std::cout << "Test accuracy: " << accuracy << "% (" << correct << "/" << test.images.size()
              << ")\n";

    return 0;
}
