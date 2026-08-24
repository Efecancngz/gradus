# gradus v2 MNIST Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note for this project specifically:** the user is building gradus to learn engineering fundamentals, not just to get a finished repo (see [[feedback_teach_while_building_gradus]]). Explain each step's *reasoning* (why this design, what tradeoff it resolves) while implementing — not just execute silently. Each task below has a "Concept" paragraph for this purpose.

**Goal:** Extend gradus with a fused `softmax_cross_entropy_loss` primitive and an `MLP::activate_output` option, then train the same from-scratch engine on the full MNIST dataset (60k train / 10k test) and report real test accuracy via `examples/mnist.cpp`.

**Architecture:** One new `Tensor` primitive (custom forward+backward, like `matmul`/`tanh`), one small non-breaking extension to `MLP`, a small CSV loader module, and a new example binary — no changes to `Tensor`'s core graph/backward machinery from v1.

**Tech Stack:** Same as v1 (C++17, CMake, Catch2). New: a bash download script for the CSV data source, no new C++ dependencies.

**Spec:** `docs/superpowers/specs/2026-08-24-gradus-mnist-classifier-design.md`

## Global Constraints

- No batching, no SIMD, no BLAS — `matmul` and the training loop stay single-example-at-a-time, unchanged from v1.
- CI must stay hermetic: no test may read from `data/` or the network. Only `examples/mnist.cpp` (never run by `ctest`) touches real MNIST files.
- `data/` is gitignored — the 127MB of CSVs are never committed.
- `MLP`'s default behavior is unchanged (`activate_output` defaults to `true`) — the existing XOR construction and its passing tests must not need any changes.
- Pixel normalization: `(raw_value / 127.5) - 1.0`, matching the `[-1, 1]` convention already used by `examples/xor.cpp`.
- Data source URLs (verified live before this plan was written): `https://data.pjreddie.com/files/mnist_train.csv`, `https://data.pjreddie.com/files/mnist_test.csv`. Row format: `label,pixel0,pixel1,...,pixel783` (785 comma-separated fields, no header row).

---

### Task 1: `softmax_cross_entropy_loss` — Fused Primitive

**Concept:** Softmax and cross-entropy, differentiated as two separate steps, would require a full Jacobian matrix for the softmax step — expensive and numerically messy. Differentiated *together* as one function, the gradient collapses to a single elegant subtraction: `dL/dx_i = softmax(x)_i - [i == true_class]`. This is exactly why every real framework fuses these two operations (`torch.nn.CrossEntropyLoss` takes raw logits for this reason, not as a convenience shortcut but because the fused gradient is what you actually want). The forward pass also needs the "log-sum-exp trick" — subtracting the maximum logit before exponentiating — because `exp()` of an ordinary-sized number (say, 50) silently overflows a `double` into `inf`, and `inf/inf` produces `NaN`. Subtracting the max first keeps every exponent `<= 0`, so `exp()` never overflows, and the math works out to the exact same result (this is a standard numerical-stability technique, not an approximation).

**Files:**
- Modify: `include/gradus/tensor.hpp`
- Modify: `src/tensor.cpp`
- Create: `tests/test_softmax_cross_entropy.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorImpl`/`Tensor` core from v1 (Task 2 of the v1 plan).
- Produces: `Tensor::softmax_cross_entropy_loss(int target_class) const -> Tensor` (scalar output) — consumed by `examples/mnist.cpp` (Task 5).

- [ ] **Step 1: Write the failing test — `tests/test_softmax_cross_entropy.cpp`**

```cpp
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

    // CE = logsumexp(logits) - logits[target]
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
```

- [ ] **Step 2: Add to `tests/CMakeLists.txt`**

Add `test_softmax_cross_entropy.cpp` to the `gradus_tests` source list.

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `Tensor` has no member `softmax_cross_entropy_loss`.

- [ ] **Step 4: Declare in `include/gradus/tensor.hpp`**

Add inside `class Tensor`, after `sum()`:

```cpp
    Tensor softmax_cross_entropy_loss(int target_class) const;
```

- [ ] **Step 5: Define in `src/tensor.cpp`**

Add after `Tensor::sum()`. `<cmath>` and `<algorithm>` are already included at the top of this file (used by `tanh`/`zero_grad`), so no new includes are needed.

```cpp
Tensor Tensor::softmax_cross_entropy_loss(int target_class) const {
    if (impl->shape.size() != 2 || impl->shape[0] != 1) {
        throw std::invalid_argument("softmax_cross_entropy_loss requires a {1, C} logits tensor");
    }
    size_t num_classes = impl->shape[1];
    if (target_class < 0 || static_cast<size_t>(target_class) >= num_classes) {
        throw std::invalid_argument("softmax_cross_entropy_loss: target_class out of range");
    }

    double max_logit = impl->data[0];
    for (size_t i = 1; i < num_classes; ++i) {
        max_logit = std::max(max_logit, impl->data[i]);
    }

    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(impl->data[i] - max_logit);
    }

    double log_sum_exp = std::log(sum_exp) + max_logit;
    double loss_value = log_sum_exp - impl->data[static_cast<size_t>(target_class)];

    auto out_impl = std::make_shared<TensorImpl>(std::vector<double>{loss_value}, std::vector<size_t>{1});
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw, target_class, max_logit, sum_exp, num_classes]() {
        for (size_t i = 0; i < num_classes; ++i) {
            double softmax_i = std::exp(input->data[i] - max_logit) / sum_exp;
            double indicator = (static_cast<int>(i) == target_class) ? 1.0 : 0.0;
            input->grad[i] += (softmax_i - indicator) * out_raw->grad[0];
        }
    };

    return Tensor(out_impl);
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS, all new test cases green including the "softmax minus one-hot" identity check.

- [ ] **Step 7: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_softmax_cross_entropy.cpp tests/CMakeLists.txt
git commit -m "feat: add fused softmax_cross_entropy_loss primitive"
```

---

### Task 2: `MLP::activate_output`

**Concept:** v1's `MLP::forward` applies `tanh` after every layer, including the last — correct for XOR, where the output must land in `(-1, 1)` to match its targets. For classification with `softmax_cross_entropy_loss`, the final layer must emit *raw, unbounded logits* — squashing them through `tanh` first would compress the very signal `softmax` needs to work with, and would fight against the loss's own gradient. This task adds a constructor flag so both use cases coexist without duplicating the `MLP` class — a real need driving the change, not speculative flexibility added ahead of time.

**Files:**
- Modify: `include/gradus/nn.hpp`
- Modify: `src/nn.cpp`
- Modify: `tests/test_nn.cpp`

**Interfaces:**
- Consumes: `Linear`, `Tensor::tanh()` from v1.
- Produces: `MLP(size_t in_features, std::vector<size_t> layer_sizes, bool activate_output = true)` — consumed by `examples/mnist.cpp` (Task 5) with `activate_output=false`; `examples/xor.cpp` and existing tests are unaffected (default unchanged).

- [ ] **Step 1: Write the failing test — append to `tests/test_nn.cpp`**

```cpp
#include <cmath>

TEST_CASE("MLP with activate_output=true (default) bounds its output via tanh") {
    MLP mlp(2, {4, 1});  // default: activate_output = true
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    // tanh's range is strictly (-1, 1)
    REQUIRE(y.item() > -1.0);
    REQUIRE(y.item() < 1.0);
}

TEST_CASE("MLP with activate_output=false emits raw, unbounded logits") {
    MLP mlp(2, {4, 3}, /*activate_output=*/false);
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 3});
    // Not asserting a specific magnitude (depends on random init) — the
    // point is the *architecture* difference is exercised, not a specific
    // numeric output.
    y.backward();
    bool any_param_has_nonzero_grad = false;
    for (auto& p : mlp.parameters()) {
        for (double g : p.grad()) {
            if (g != 0.0) any_param_has_nonzero_grad = true;
        }
    }
    REQUIRE(any_param_has_nonzero_grad);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `MLP` constructor doesn't accept a third argument yet.

- [ ] **Step 3: Update `include/gradus/nn.hpp`**

```cpp
class MLP {
public:
    MLP(size_t in_features, std::vector<size_t> layer_sizes, bool activate_output = true);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> parameters() const;

private:
    std::vector<Linear> layers_;
    bool activate_output_;
};
```

- [ ] **Step 4: Update `src/nn.cpp`**

```cpp
MLP::MLP(size_t in_features, std::vector<size_t> layer_sizes, bool activate_output)
    : activate_output_(activate_output) {
    size_t prev = in_features;
    for (size_t sz : layer_sizes) {
        layers_.emplace_back(prev, sz);
        prev = sz;
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor out = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        out = layers_[i].forward(out);
        bool is_last_layer = (i == layers_.size() - 1);
        if (!is_last_layer || activate_output_) {
            out = out.tanh();
        }
    }
    return out;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS — including the pre-existing XOR integration test (Task 12 of the v1 plan), which must still pass unchanged since it relies on the default `activate_output=true` behavior.

- [ ] **Step 6: Commit**

```bash
git add include/gradus/nn.hpp src/nn.cpp tests/test_nn.cpp
git commit -m "feat: add MLP activate_output option for raw-logit classification heads"
```

---

### Task 3: MNIST CSV Loader

**Concept:** MNIST's *original* distribution format is a custom binary format (IDX) with its own header/magic-number conventions — parsing it would be a file-format exercise unrelated to this project's actual point (autograd internals). A pre-converted CSV mirror (`label,pixel0,...,pixel783`, verified live before writing this plan) sidesteps that entirely: one `std::getline` + one `std::stringstream` per row. This task also introduces the `[-1, 1]` pixel normalization: raw pixel bytes are `0-255`; dividing by `127.5` and subtracting `1.0` maps `0 → -1.0` and `255 → 1.0`, keeping inputs zero-centered — the same range convention `examples/xor.cpp` already uses, and one that trains better with `tanh` hidden layers than raw `[0, 255]` values would (large, all-positive inputs bias every neuron's pre-activation in the same direction).

**Files:**
- Create: `include/gradus/mnist_loader.hpp`
- Create: `src/mnist_loader.cpp`
- Modify: `CMakeLists.txt` — add `src/mnist_loader.cpp` to the `gradus` library sources
- Create: `tests/test_mnist_loader.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `gradus::MnistDataset { images: vector<vector<double>>, labels: vector<int> }`, `gradus::load_mnist_csv(const std::string& path) -> MnistDataset` — consumed by `examples/mnist.cpp` (Task 5).

- [ ] **Step 1: Write the failing test — `tests/test_mnist_loader.cpp`**

This test writes a small synthetic CSV to a temp file (2 rows, 3x3=9 "pixels" instead of 784, to keep the test fast and readable) rather than depending on the real 127MB download — the parsing logic is identical regardless of row width.

```cpp
#include <fstream>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradus/mnist_loader.hpp"

namespace {
std::string write_temp_csv(const std::string& content) {
    std::string path = "test_mnist_sample.csv";
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
}
}  // namespace

TEST_CASE("load_mnist_csv parses rows into images and labels") {
    std::string path = write_temp_csv(
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
    std::string path = write_temp_csv("5,0,255,127\n");
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
        "7,0,255,0\n"
        "3,255,255\n");
    REQUIRE_THROWS_AS(gradus::load_mnist_csv(path), std::runtime_error);
    std::remove(path.c_str());
}
```

- [ ] **Step 2: Add to `tests/CMakeLists.txt`**

Add `test_mnist_loader.cpp` to the `gradus_tests` source list.

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `gradus/mnist_loader.hpp` doesn't exist.

- [ ] **Step 4: Write `include/gradus/mnist_loader.hpp`**

```cpp
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
```

- [ ] **Step 5: Write `src/mnist_loader.cpp`**

```cpp
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
            throw std::runtime_error(
                "load_mnist_csv: row " + std::to_string(line_number) +
                " has " + std::to_string(fields.size()) + " fields, expected " +
                std::to_string(expected_fields));
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
```

- [ ] **Step 6: Add `src/mnist_loader.cpp` to `CMakeLists.txt`**

```cmake
add_library(gradus
  src/tensor.cpp
  src/nn.cpp
  src/optim.cpp
  src/mnist_loader.cpp
)
```

- [ ] **Step 7: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/gradus/mnist_loader.hpp src/mnist_loader.cpp CMakeLists.txt tests/test_mnist_loader.cpp tests/CMakeLists.txt
git commit -m "feat: add MNIST CSV loader with [-1,1] pixel normalization"
```

---

### Task 4: Download Script & `.gitignore`

**Concept:** The dataset is a fetchable artifact, not source code — it doesn't belong in git history (127MB, and every clone would re-download bytes that never change). The convention across virtually every ML project is a small script that fetches data into a gitignored directory, run once by whoever wants to use the example. This task is pure infrastructure — no C++ code, no tests (a shell script that calls `curl` has nothing to unit-test; its correctness is verified by actually running it in Task 6).

**Files:**
- Create: `scripts/download_mnist.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Write `scripts/download_mnist.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Downloads the MNIST CSV mirror used by examples/mnist.cpp.
# Source verified live 2026-08-24: https://data.pjreddie.com/files/
mkdir -p data
echo "Downloading mnist_train.csv (~109MB)..."
curl -L -o data/mnist_train.csv https://data.pjreddie.com/files/mnist_train.csv
echo "Downloading mnist_test.csv (~18MB)..."
curl -L -o data/mnist_test.csv https://data.pjreddie.com/files/mnist_test.csv
echo "Done. Files are in data/."
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/download_mnist.sh`

- [ ] **Step 3: Add `data/` to `.gitignore`**

```
data/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_mnist.sh .gitignore
git commit -m "chore: add MNIST download script, gitignore data/"
```

---

### Task 5: `examples/mnist.cpp`

**Concept:** This is where every earlier piece (the engine, the fused loss, the raw-logit `MLP`, the loader) comes together into the actual proof: does this from-scratch autograd engine, trained on real (not toy) data, actually learn to classify handwritten digits? `argmax` over the 10 output logits gives the predicted class — softmax isn't even needed at inference time for picking the winner, since softmax is monotonic (the largest logit is always the largest softmax probability too); it only mattered for computing a differentiable *loss* during training.

**Files:**
- Create: `examples/mnist.cpp`
- Modify: `CMakeLists.txt` — add the `mnist_example` executable target

**Interfaces:**
- Consumes: `MLP` (Task 2), `SGD` (v1), `load_mnist_csv`/`MnistDataset` (Task 3), `Tensor::softmax_cross_entropy_loss` (Task 1).

- [ ] **Step 1: Write `examples/mnist.cpp`**

```cpp
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
    std::cout << "Loaded " << train.images.size() << " train / "
              << test.images.size() << " test examples.\n";

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
        std::cout << "epoch " << epoch
                  << "  avg loss " << (total_loss / train.images.size())
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
    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(test.images.size());
    std::cout << "Test accuracy: " << accuracy << "% (" << correct << "/" << test.images.size() << ")\n";

    return 0;
}
```

- [ ] **Step 2: Add the executable to `CMakeLists.txt`**

Add near `xor_example`:

```cmake
add_executable(mnist_example examples/mnist.cpp)
target_link_libraries(mnist_example PRIVATE gradus)
```

- [ ] **Step 3: Build (do not run yet — no data downloaded)**

Run: `cmake --build build --parallel`
Expected: builds cleanly. Running it now would fail at `load_mnist_csv` with "could not open file" — expected, since Task 6 downloads the data.

- [ ] **Step 4: Commit**

```bash
git add examples/mnist.cpp CMakeLists.txt
git commit -m "feat: add MNIST training/evaluation example"
```

---

### Task 6: Run It for Real

**Concept:** Definition of done requires the code to be "elle bir kez gerçekten çalıştırıldı" (manually run at least once) — a passing test suite proves individual pieces work in isolation, not that the full pipeline produces a real, meaningful result on real data. This task is that end-to-end proof: download the actual dataset, train on all 60,000 images, and confirm the reported test accuracy is genuinely far above the 10% a random guesser would get.

- [ ] **Step 1: Run the download script**

Run: `bash scripts/download_mnist.sh`
Expected: `data/mnist_train.csv` (~109MB) and `data/mnist_test.csv` (~18MB) appear. This will take a while depending on connection speed — run in the background and check on it rather than blocking.

- [ ] **Step 2: Build in Release mode for training speed**

Run: `cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++` then `cmake --build build-release --parallel`

Note: a separate build directory (`build-release`) is used rather than reconfiguring `build/` in place, so the existing Debug build used by `ctest` is untouched.

- [ ] **Step 3: Run `mnist_example` from the repo root**

Run: `./build-release/mnist_example.exe` (the binary opens `data/mnist_train.csv` with a relative path, so it must run from the repo root, or the paths in Task 5's code adjusted).

Expected: prints loading progress, then per-epoch loss + timing for 5 epochs, then a final test accuracy line. This will take real time (see the plan's Global Constraints — tens of seconds per epoch is expected, not a hang). Run in the background and monitor rather than blocking the session.

- [ ] **Step 4: Sanity-check the result**

Confirm the printed test accuracy is well above 10% (random-guess baseline for 10 classes). If it's anywhere near 10%, something is architecturally wrong (e.g., a sign error in the fused gradient, or the `activate_output=false` path not actually skipping the final `tanh`) and needs debugging before this task is considered done — do not report success without this check passing.

- [ ] **Step 5: Update `HANDOFF.md`**

Record the actual measured accuracy and epoch timing in `HANDOFF.md` so the number isn't lost — real measured results, not the plan's estimate.

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Add an MNIST section to `README.md`**

Add under Quick Start:

```md
## MNIST example
\`\`\`bash
bash scripts/download_mnist.sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/mnist_example
\`\`\`
```

- [ ] **Step 2: Add decisions-log entries to `docs/architecture.md`**

Append to the existing "Decisions log" list (do not remove v1's entries):

```md
- **Fused `softmax_cross_entropy_loss`, not composed from a separate
  `softmax()` + `log()`.** The combined gradient (`softmax(x) - one_hot`)
  is what every production framework computes directly — deriving it as
  a composition would need a full softmax Jacobian, which is both slower
  and numerically messier for no benefit.
- **`MLP::activate_output` flag, default `true`.** v1's XOR network needed
  every layer (including the last) tanh-bounded; classification needs raw
  logits at the output. A flag serves both without duplicating `MLP`.
- **MNIST loaded from a CSV mirror, not the original IDX binary format.**
  Parsing IDX would be a file-format exercise orthogonal to this project's
  actual point (autograd internals) — a deliberate build-vs-buy call.
```

- [ ] **Step 3: Update `HANDOFF.md`** to reflect v2 completion (fold in the real accuracy/timing numbers from Task 6, Step 5).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/architecture.md HANDOFF.md
git commit -m "docs: document MNIST example and v2 architecture decisions"
```

---

## Post-Plan Checklist (Definition of Done)

- [ ] All new unit tests pass locally (`ctest --test-dir build --output-on-failure`), with no test touching `data/` or the network
- [ ] Existing v1 tests (XOR integration test especially) still pass unchanged
- [ ] `mnist_example` actually run once, end to end, on real downloaded data — accuracy confirmed well above the 10% random baseline
- [ ] `README.md` / `docs/architecture.md` / `HANDOFF.md` updated in the same set of commits as the code they describe
- [ ] No `Co-Authored-By: Claude` in any commit
- [ ] `data/` never appears in `git status` as staged/tracked
