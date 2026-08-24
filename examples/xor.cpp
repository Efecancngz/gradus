#include <iostream>

#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

int main() {
    using namespace gradus;

    std::vector<std::vector<double>> inputs = {{-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}};
    std::vector<double> targets = {-1.0, 1.0, 1.0, -1.0};

    MLP mlp(2, {4, 1});
    SGD optimizer(mlp.parameters(), 0.1);

    for (int epoch = 0; epoch < 300; ++epoch) {
        double epoch_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor x(inputs[i], {1, 2});
            Tensor y(std::vector<double>{targets[i]}, {1, 1});

            Tensor pred = mlp.forward(x);
            Tensor loss = mse_loss(pred, y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item();
        }
        if (epoch % 30 == 0) {
            std::cout << "epoch " << epoch << "  avg loss " << (epoch_loss / inputs.size()) << "\n";
        }
    }

    std::cout << "\nFinal predictions:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor x(inputs[i], {1, 2});
        Tensor pred = mlp.forward(x);
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << "  ->  " << pred.item()
                  << "  (target " << targets[i] << ")\n";
    }

    return 0;
}
