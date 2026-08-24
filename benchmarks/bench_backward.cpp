#include <benchmark/benchmark.h>

#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

static void BM_MLPForwardBackward(benchmark::State& state) {
    using namespace gradus;
    MLP mlp(2, {4, 1});
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor target({1.0}, {1, 1});

    for (auto _ : state) {
        Tensor pred = mlp.forward(x);
        Tensor loss = mse_loss(pred, target);
        loss.backward();
        for (auto& p : mlp.parameters()) {
            p.zero_grad();
        }
    }
}
BENCHMARK(BM_MLPForwardBackward);

BENCHMARK_MAIN();
