# gradus v1 Autograd Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note for this project specifically:** the user is building gradus to learn autograd/backprop fundamentals, not just to get a finished repo (see [[feedback_teach_while_building_gradus]] in the vault memory). Each task below includes a "Concept" paragraph explaining the underlying idea — whoever executes this plan should walk through that explanation with the user before/while writing the code, not just produce the diff silently.

**Goal:** Build a working, tested, from-scratch reverse-mode automatic differentiation engine in C++17 (`gradus`), with a small neural-net layer on top, that trains an MLP on XOR end-to-end.

**Architecture:** A single unified `Tensor` class (scalar = 1-element tensor) backed by a reference-counted `TensorImpl` graph node holding data, gradient, shape, parent pointers, and a backward closure. `backward()` performs a topological sort of the graph and walks it in reverse, calling each node's local chain-rule closure. `Linear`/`MLP`/`SGD` are built entirely on top of `Tensor`'s public operators — no separate code path.

**Tech Stack:** C++17, CMake (FetchContent for deps, no system package manager requirement), Catch2 v3 (tests), Google Benchmark (optional, non-gating), AddressSanitizer/UndefinedBehaviorSanitizer (dedicated CI job), clang-format.

**Spec:** `docs/superpowers/specs/2026-08-24-gradus-autograd-engine-design.md`

## Global Constraints

- Language standard: C++17, enforced via `CMAKE_CXX_STANDARD 17` / `CMAKE_CXX_STANDARD_REQUIRED ON`.
- Dependency versions are pinned exactly: Catch2 `v3.6.0`, Google Benchmark `v1.8.4` — do not float to `main`/`master`.
- **No `requires_grad` flag.** Every `Tensor` always builds a graph node and always tracks gradients — this is a deliberate v1 simplification (YAGNI: no `no_grad` mode is needed for XOR-scale examples). Do not add this flag in any task; if a later task seems to need it, that's a signal to stop and reconsider, not to add it silently.
- **No broadcasting.** `operator+`, `operator-`, `operator*` (elementwise) all require the two operands to have exactly equal `shape`. Mismatches throw `std::invalid_argument`. This keeps every op's backward pass a simple one-to-one loop — no reduction-over-broadcast-dims logic anywhere in v1.
- `matmul` only supports 2D tensors (shape size 2). `Linear` always treats its input as a `{1, in_features}` row-vector tensor — batching is out of scope for v1.
- Namespace: everything lives in `namespace gradus`.
- Commit messages: Conventional Commits (`feat:`, `test:`, `docs:`, `chore:`), English, imperative mood — no AI co-author trailer, ever.
- Every task's test step must actually be run and observed failing, then passing — do not skip the "verify it fails" step even though it feels repetitive; that step is what proves the test is actually exercising the new code.

---

### Task 1: Project Scaffolding & Build Skeleton

**Concept:** Before any autograd code exists, we need a build system that can fetch a test framework and compile an (empty) test binary. CMake's `FetchContent` module downloads a pinned dependency at configure time and makes its build targets available — this avoids requiring the user to `apt install` or `vcpkg install` anything; `git clone` + `cmake` is the entire setup story. Getting this right first means every later task's "write test, watch it fail, make it pass" cycle just works.

**Files:**
- Create: `CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `tests/test_placeholder.cpp`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `HANDOFF.md`

**Interfaces:**
- Produces: a `gradus` CMake library target (empty for now — populated in Task 2+), a `gradus_tests` CTest-registered executable, a `catch2` FetchContent target every later task's test files link against.

- [ ] **Step 1: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(gradus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GRADUS_ENABLE_SANITIZERS "Build with ASan/UBSan" OFF)
option(GRADUS_BUILD_BENCHMARKS "Build the benchmark executable" OFF)

include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(catch2)

add_library(gradus
  src/tensor.cpp
)
target_include_directories(gradus PUBLIC include)

if(GRADUS_ENABLE_SANITIZERS)
  target_compile_options(gradus PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(gradus PUBLIC -fsanitize=address,undefined)
endif()

enable_testing()
add_subdirectory(tests)
```

Note: `src/tensor.cpp` is referenced here but doesn't exist until Task 2. Create an empty placeholder now so this configures:

```cpp
// src/tensor.cpp intentionally left minimal until Task 2.
```

- [ ] **Step 2: Write `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_placeholder.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Write `tests/test_placeholder.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>

TEST_CASE("build skeleton compiles and runs") {
    REQUIRE(1 + 1 == 2);
}
```

- [ ] **Step 4: Configure and build**

Run: `cmake -S . -B build && cmake --build build --parallel`
Expected: configures successfully (FetchContent downloads Catch2), builds `gradus_tests` with no errors.

- [ ] **Step 5: Run the test**

Run: `ctest --test-dir build --output-on-failure`
Expected: 1 test passes.

- [ ] **Step 6: Write `.gitignore`**

```
build/
build-*/
.vs/
.vscode/
*.obj
*.o
CMakeFiles/
CMakeCache.txt
```

- [ ] **Step 7: Write `LICENSE`**

Use the standard MIT license text with copyright line `Copyright (c) 2026 Efecan Cengiz`.

- [ ] **Step 8: Write `README.md`**

```md
# gradus

A reverse-mode automatic differentiation (autograd) engine written from scratch in C++17.

## Why
This project exists to learn — not to replace PyTorch. Every existing project
in this author's portfolio uses AI/ML through an API; none of them show an
understanding of how a neural network actually learns. `gradus` implements
backpropagation by hand: a `Tensor` class that builds a computation graph as
operations run, and a `backward()` that walks that graph in reverse applying
the chain rule at every node. A mature library (libtorch, dlib) exists for
every one of these operations — the point of `gradus` is writing them anyway.

## Stack
C++17 · CMake · Catch2 · Google Benchmark · ASan/UBSan · GitHub Actions

## Quick start
\`\`\`bash
git clone <repo-url>
cd gradus
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/xor_example
\`\`\`

## Documentation
- [Architecture](docs/architecture.md)

## License
MIT — see [LICENSE](LICENSE)
```

- [ ] **Step 9: Write `CLAUDE.md`**

```md
# gradus

C++17 autograd engine, portfolio + learning project. See `docs/architecture.md`
for the design and decisions log; see `docs/superpowers/specs/` for the
original design spec.

## Run commands
- Build: `cmake -S . -B build && cmake --build build --parallel`
- Test: `ctest --test-dir build --output-on-failure`
- Sanitizer build: `cmake -S . -B build-san -DGRADUS_ENABLE_SANITIZERS=ON && cmake --build build-san && ctest --test-dir build-san`
- Benchmarks (optional): `cmake -S . -B build-bench -DGRADUS_BUILD_BENCHMARKS=ON && cmake --build build-bench && ./build-bench/gradus_bench`

## Why these choices
- Single unified `Tensor` type (no separate scalar `Value` class) — avoids
  duplicating the graph/backward machinery for two node types.
- No `requires_grad` flag — every tensor always tracks gradients (v1 scope
  never needs to disable it).
- No broadcasting — every elementwise op requires exact shape match; keeps
  every backward pass a simple loop.

See `HANDOFF.md` for current status.
```

- [ ] **Step 10: Write `HANDOFF.md`**

```md
# Handoff — gradus

Son güncelleme: 2026-08-24, güncelleyen: Claude Sonnet 5

## Şu an ne yapılıyor
Proje iskeleti kuruldu (CMake + Catch2), Task 1 tamamlandı.

## Sıradaki somut adım
Task 2 — Tensor/TensorImpl çekirdek veri yapısı ve backward() sürücüsü.

## Bilinmesi gerekenler
- Denenip işe yaramayan yaklaşım yok (proje yeni başladı).
- Dikkat: `requires_grad` flag'i bilinçli olarak yok — eklenmemeli.

## İlgili dosyalar
- docs/superpowers/plans/2026-08-24-gradus-v1-autograd-engine.md — tüm plan
- docs/superpowers/specs/2026-08-24-gradus-autograd-engine-design.md — tasarım

## Son 3 commit
- (henüz yok)
```

- [ ] **Step 11: Commit**

```bash
git add CMakeLists.txt tests/ .gitignore LICENSE README.md CLAUDE.md HANDOFF.md src/tensor.cpp
git commit -m "chore: scaffold CMake + Catch2 build, add project docs"
```

---

### Task 2: Tensor Core — Construction, Shape, and the `backward()` Driver

**Concept:** This is the heart of the whole engine. Every `Tensor` the user creates or computes is really a thin handle (`Tensor`) around a shared, reference-counted graph node (`TensorImpl`). We use `shared_ptr` because the same node can be a parent of *multiple* other nodes (e.g. `a` used twice in `a*a + a`) — ordinary single ownership can't express that. Each node stores its own `data`, its own `grad` (same size, starts at zero), pointers to its `parents`, and a `backward_fn` closure that knows *how to push gradient from this node backward onto its parents* — that closure is where the chain rule for that specific operation lives. `backward()` itself doesn't know any calculus; it just does a **topological sort** (visit every node's parents before the node itself, so we never process a node before all its dependents have contributed to its output) and then walks that order **in reverse**, calling each node's `backward_fn`. This task builds that machinery with no actual math operations yet — just the plumbing, proven with a trivial one-node graph.

**Files:**
- Create: `include/gradus/tensor.hpp`
- Modify: `src/tensor.cpp`
- Create: `tests/test_tensor_basics.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_tensor_basics.cpp` to the `gradus_tests` sources list, remove `test_placeholder.cpp`
- Delete: `tests/test_placeholder.cpp`

**Interfaces:**
- Produces: `gradus::Tensor` with constructors `Tensor(double)` and `Tensor(std::vector<double>, std::vector<size_t>)`, methods `size()`, `shape()`, `item()`, `grad()`, `data()`, `backward()`, `zero_grad()`. Every later task's operators build on `TensorImpl`'s `parents`/`backward_fn` fields.

- [ ] **Step 1: Write `include/gradus/tensor.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace gradus {

struct TensorImpl {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<size_t> shape;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_fn;

    TensorImpl(std::vector<double> data_, std::vector<size_t> shape_);
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    explicit Tensor(double value);
    Tensor(std::vector<double> data, std::vector<size_t> shape);
    explicit Tensor(std::shared_ptr<TensorImpl> impl_);

    size_t size() const;
    const std::vector<size_t>& shape() const;
    const std::vector<double>& data() const;
    std::vector<double>& grad();
    double item() const;

    void backward();
    void zero_grad();
};

}  // namespace gradus
```

- [ ] **Step 2: Write `src/tensor.cpp`**

```cpp
#include "gradus/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace gradus {

TensorImpl::TensorImpl(std::vector<double> data_, std::vector<size_t> shape_)
    : data(std::move(data_)), shape(std::move(shape_)) {
    grad.assign(data.size(), 0.0);
}

Tensor::Tensor(double value)
    : impl(std::make_shared<TensorImpl>(std::vector<double>{value}, std::vector<size_t>{1})) {}

Tensor::Tensor(std::vector<double> data, std::vector<size_t> shape)
    : impl(std::make_shared<TensorImpl>(std::move(data), std::move(shape))) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl_) : impl(std::move(impl_)) {}

size_t Tensor::size() const {
    return impl->data.size();
}

const std::vector<size_t>& Tensor::shape() const {
    return impl->shape;
}

const std::vector<double>& Tensor::data() const {
    return impl->data;
}

std::vector<double>& Tensor::grad() {
    return impl->grad;
}

double Tensor::item() const {
    if (size() != 1) {
        throw std::invalid_argument("item() requires a single-element tensor");
    }
    return impl->data[0];
}

void Tensor::zero_grad() {
    std::fill(impl->grad.begin(), impl->grad.end(), 0.0);
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::invalid_argument("backward() can only be called on a scalar (single-element) tensor");
    }

    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::unordered_set<TensorImpl*> visited;

    std::function<void(const std::shared_ptr<TensorImpl>&)> build_topo =
        [&](const std::shared_ptr<TensorImpl>& node) {
            if (visited.count(node.get())) {
                return;
            }
            visited.insert(node.get());
            for (const auto& parent : node->parents) {
                build_topo(parent);
            }
            topo.push_back(node);
        };
    build_topo(impl);

    impl->grad[0] = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

}  // namespace gradus
```

- [ ] **Step 3: Write `tests/test_tensor_basics.cpp`**

```cpp
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
```

- [ ] **Step 4: Update `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 5: Delete `tests/test_placeholder.cpp`**

- [ ] **Step 6: Build and run, verify failure first**

Temporarily this step is about confirming the *previous* skeleton test is gone and the new one compiles — run:
Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: all 6 new test cases pass (this task writes implementation alongside tests rather than strict red-green per-case, since the "core" is plumbing with no separate bug to catch — but still run and observe green before moving on).

- [ ] **Step 7: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_basics.cpp tests/CMakeLists.txt
git rm tests/test_placeholder.cpp
git commit -m "feat: add Tensor core with topological-sort backward() driver"
```

---

### Task 3: Numerical Gradient-Check Test Utility

**Concept:** Every operator we add from Task 4 onward needs its backward pass (the analytic gradient) verified. The standard way to check "is this derivative formula actually right" is **numerical gradient checking**: nudge one input by a tiny amount (`epsilon`), see how much the output changes, and divide — that's the definition of a derivative (`(f(x+ε) − f(x−ε)) / 2ε`, the *central difference* form, which is more accurate than a one-sided difference). If the analytic gradient from `backward()` doesn't match this numerical estimate within a small tolerance, the backward formula has a bug. This is exactly the technique PyTorch's own `torch.autograd.gradcheck` uses. We build this utility now, generically (operating on plain `std::vector<double>`, no `Tensor` involved yet) so every later task can reuse it.

**Files:**
- Create: `tests/gradient_check.hpp`
- Create: `tests/test_gradient_check.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_gradient_check.cpp`

**Interfaces:**
- Produces: `gradus::testutil::numerical_gradient(f, x, epsilon = 1e-5) -> std::vector<double>`, used by every operator test from Task 4 onward.

- [ ] **Step 1: Write the failing test — `tests/test_gradient_check.cpp`**

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradient_check.hpp"

TEST_CASE("numerical_gradient matches a known analytic derivative") {
    // f(x) = x^2 + 3x  =>  df/dx = 2x + 3
    auto f = [](const std::vector<double>& x) {
        return x[0] * x[0] + 3.0 * x[0];
    };
    auto grad = gradus::testutil::numerical_gradient(f, {2.0});
    REQUIRE(grad[0] == Catch::Approx(2.0 * 2.0 + 3.0).epsilon(1e-4));
}

TEST_CASE("numerical_gradient handles multiple inputs independently") {
    // f(x, y) = x*y  =>  df/dx = y, df/dy = x
    auto f = [](const std::vector<double>& v) {
        return v[0] * v[1];
    };
    auto grad = gradus::testutil::numerical_gradient(f, {3.0, 7.0});
    REQUIRE(grad[0] == Catch::Approx(7.0).epsilon(1e-4));
    REQUIRE(grad[1] == Catch::Approx(3.0).epsilon(1e-4));
}
```

- [ ] **Step 2: Add the new test file to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `gradient_check.hpp` doesn't exist yet, compile error `fatal error: gradient_check.hpp: No such file or directory`.

- [ ] **Step 4: Write `tests/gradient_check.hpp`**

```cpp
#pragma once

#include <functional>
#include <vector>

namespace gradus::testutil {

// Central-difference numerical gradient of `f` at `x`. Used to verify
// analytic backward() implementations against a ground truth that doesn't
// depend on the analytic code being correct.
inline std::vector<double> numerical_gradient(
    const std::function<double(const std::vector<double>&)>& f,
    std::vector<double> x,
    double epsilon = 1e-5) {
    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double original = x[i];

        x[i] = original + epsilon;
        double f_plus = f(x);

        x[i] = original - epsilon;
        double f_minus = f(x);

        x[i] = original;
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return grad;
}

}  // namespace gradus::testutil
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS — both new test cases green.

- [ ] **Step 6: Commit**

```bash
git add tests/gradient_check.hpp tests/test_gradient_check.cpp tests/CMakeLists.txt
git commit -m "test: add numerical gradient-check utility for verifying backward passes"
```

---

### Task 4: `operator+` and `operator-` (Elementwise) With Backward

**Concept:** This is the first *real* operator, and it establishes the pattern every later operator follows: (1) compute the forward result, (2) allocate a new `TensorImpl` for it, (3) record the inputs as `parents`, (4) attach a `backward_fn` closure implementing that operation's piece of the chain rule. For addition, `d(a+b)/da = 1` and `d(a+b)/db = 1` — so the local gradient contribution is just "pass the output's gradient straight through to both inputs unchanged". Note the closure captures the input `shared_ptr`s **by value** (keeps them alive as long as the graph exists) but captures the *output* node by **raw pointer** (`out_raw`) — capturing the output as a `shared_ptr` inside its own closure would create a reference cycle (the node would own a `shared_ptr` to itself via its own `backward_fn`), which `shared_ptr` can never free. The raw pointer is safe here because the closure only ever runs while the node is still alive and owned elsewhere (by the graph).

**Files:**
- Modify: `include/gradus/tensor.hpp` — declare `operator+`, `operator-`
- Modify: `src/tensor.cpp` — define them
- Create: `tests/test_tensor_ops.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_tensor_ops.cpp`

**Interfaces:**
- Consumes: `Tensor::size()`, `.shape()`, `.item()`, `.backward()` from Task 2; `numerical_gradient` from Task 3.
- Produces: `Tensor::operator+(const Tensor&) const`, `Tensor::operator-(const Tensor&) const` — both used by `mse_loss` in Task 11 and by every composite example from here on.

- [ ] **Step 1: Write the failing test — `tests/test_tensor_ops.cpp`**

```cpp
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
```

- [ ] **Step 2: Add `test_tensor_ops.cpp` to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
  test_tensor_ops.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — compile error, `Tensor` has no member `operator+`/`operator-`.

- [ ] **Step 4: Declare the operators in `include/gradus/tensor.hpp`**

Add inside `class Tensor` (after `zero_grad()`):

```cpp
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
```

- [ ] **Step 5: Define them in `src/tensor.cpp`**

Add after `Tensor::backward()`:

```cpp
Tensor Tensor::operator+(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator+");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] + other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += out_raw->grad[i];
            rhs->grad[i] += out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator-");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] - other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += out_raw->grad[i];
            rhs->grad[i] -= out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS — all test cases green, including the shape-mismatch exception tests.

- [ ] **Step 7: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_ops.cpp tests/CMakeLists.txt
git commit -m "feat: add elementwise operator+ and operator- with backward"
```

---

### Task 5: `operator*` (Elementwise / Hadamard) With Backward

**Concept:** Elementwise multiplication is where the **product rule** first shows up: `d(a*b)/da = b` and `d(a*b)/db = a` — each input's local gradient depends on the *other* input's value, not a constant like addition. This is the first operator where the backward closure needs to read `data` (not just pass `grad` through), which is exactly why every node keeps its own `data` around after the forward pass — it's needed again during backward.

**Files:**
- Modify: `include/gradus/tensor.hpp` — declare `operator*`
- Modify: `src/tensor.cpp` — define it
- Modify: `tests/test_tensor_ops.cpp` — append tests

**Interfaces:**
- Consumes: same as Task 4.
- Produces: `Tensor::operator*(const Tensor&) const` — used by `mse_loss` (squaring the error) and by the diamond-graph test in Task 9.

- [ ] **Step 1: Write the failing test — append to `tests/test_tensor_ops.cpp`**

```cpp
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

    auto f = [](const std::vector<double>& x) { return x[0] * x[1]; };
    auto numgrad = numerical_gradient(f, {4.0, 5.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(b.grad()[0] == Catch::Approx(numgrad[1]).epsilon(1e-4));
}

TEST_CASE("operator* throws on shape mismatch") {
    Tensor a({1.0, 2.0}, {2});
    Tensor b({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(a * b, std::invalid_argument);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `Tensor` has no member `operator*`.

- [ ] **Step 3: Declare in `include/gradus/tensor.hpp`**

Add after `operator-`:

```cpp
    Tensor operator*(const Tensor& other) const;
```

- [ ] **Step 4: Define in `src/tensor.cpp`**

```cpp
Tensor Tensor::operator*(const Tensor& other) const {
    if (impl->shape != other.impl->shape) {
        throw std::invalid_argument("shape mismatch in operator*");
    }
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] * other.impl->data[i];
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw]() {
        for (size_t i = 0; i < lhs->data.size(); ++i) {
            lhs->grad[i] += rhs->data[i] * out_raw->grad[i];
            rhs->grad[i] += lhs->data[i] * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_ops.cpp
git commit -m "feat: add elementwise operator* with product-rule backward"
```

---

### Task 6: `matmul` With Backward

**Concept:** Matrix multiplication's backward rule looks intimidating but follows directly from the chain rule applied per-element: if `C = A @ B` (shapes `{m,k} @ {k,n} -> {m,n}`), then `dL/dA = dL/dC @ B^T` and `dL/dB = A^T @ dL/dC`. You can derive this yourself by writing out `C[i][j] = sum_p A[i][p]*B[p][j]` and differentiating — but for now, trust the two formulas and verify them with the numerical gradient checker, which doesn't care whether you understand the derivation, only whether the numbers match. This op is what makes `Linear` layers possible in Task 10 — a fully-connected layer is fundamentally "multiply the input by a weight matrix."

**Files:**
- Modify: `include/gradus/tensor.hpp` — declare `matmul`
- Modify: `src/tensor.cpp` — define it
- Modify: `tests/test_tensor_ops.cpp` — append tests

**Interfaces:**
- Consumes: same as Task 4/5.
- Produces: `Tensor::matmul(const Tensor&) const` — used by `Linear::forward` in Task 10.

- [ ] **Step 1: Write the failing test — append to `tests/test_tensor_ops.cpp`**

```cpp
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
    // matmul, which is a legitimate (if slightly indirect) way to get a
    // scalar loss to call backward() on.
    Tensor a({1.0, 2.0}, {1, 2});
    Tensor b({3.0, 4.0, 5.0, 6.0}, {2, 2});
    Tensor c = a.matmul(b);
    Tensor ones_col({1.0, 1.0}, {2, 1});
    Tensor loss = c.matmul(ones_col);  // shape {1,1}: sums c's two entries
    loss.backward();

    auto f = [](const std::vector<double>& x) {
        // x = [a0, a1, b00, b01, b10, b11]
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `Tensor` has no member `matmul`.

- [ ] **Step 3: Declare in `include/gradus/tensor.hpp`**

```cpp
    Tensor matmul(const Tensor& other) const;
```

- [ ] **Step 4: Define in `src/tensor.cpp`**

```cpp
Tensor Tensor::matmul(const Tensor& other) const {
    if (impl->shape.size() != 2 || other.impl->shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    size_t m = impl->shape[0];
    size_t k = impl->shape[1];
    if (other.impl->shape[0] != k) {
        throw std::invalid_argument("matmul shape mismatch: inner dimensions differ");
    }
    size_t n = other.impl->shape[1];

    std::vector<double> result_data(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double total = 0.0;
            for (size_t p = 0; p < k; ++p) {
                total += impl->data[i * k + p] * other.impl->data[p * n + j];
            }
            result_data[i * n + j] = total;
        }
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, std::vector<size_t>{m, n});
    out_impl->parents = {impl, other.impl};

    auto lhs = impl;
    auto rhs = other.impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [lhs, rhs, out_raw, m, k, n]() {
        // dL/dlhs = dL/dout @ rhs^T
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                double grad_sum = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    grad_sum += out_raw->grad[i * n + j] * rhs->data[p * n + j];
                }
                lhs->grad[i * k + p] += grad_sum;
            }
        }
        // dL/drhs = lhs^T @ dL/dout
        for (size_t p = 0; p < k; ++p) {
            for (size_t j = 0; j < n; ++j) {
                double grad_sum = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    grad_sum += lhs->data[i * k + p] * out_raw->grad[i * n + j];
                }
                rhs->grad[p * n + j] += grad_sum;
            }
        }
    };

    return Tensor(out_impl);
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_ops.cpp
git commit -m "feat: add matmul with backward"
```

---

### Task 7: `tanh()` and `relu()` Activations With Backward

**Concept:** Without a non-linear activation function between `Linear` layers, stacking multiple layers would be mathematically pointless — two matrix multiplications in a row collapse into a single matrix multiplication, so the network could never learn XOR (which is *not* linearly separable). `tanh` and `relu` are the two most common non-linearities. `tanh`'s derivative has a convenient closed form in terms of its *own output*: `d(tanh(x))/dx = 1 - tanh(x)^2` — so the backward closure doesn't even need the original input, just the output it already computed. `relu(x) = max(0, x)` has derivative `1` where `x > 0` and `0` where `x <= 0` (the derivative at exactly `x=0` is technically undefined, but treating it as `0` there is the standard convention every real framework uses).

**Files:**
- Modify: `include/gradus/tensor.hpp` — declare `tanh`, `relu`
- Modify: `src/tensor.cpp` — define them
- Modify: `tests/test_tensor_ops.cpp` — append tests

**Interfaces:**
- Consumes: same as prior tasks.
- Produces: `Tensor::tanh() const`, `Tensor::relu() const` — used by `MLP::forward` in Task 10.

- [ ] **Step 1: Write the failing test — append to `tests/test_tensor_ops.cpp`**

```cpp
#include <cmath>

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `Tensor` has no members `tanh`/`relu`.

- [ ] **Step 3: Declare in `include/gradus/tensor.hpp`**

```cpp
    Tensor tanh() const;
    Tensor relu() const;
```

- [ ] **Step 4: Define in `src/tensor.cpp`**

```cpp
Tensor Tensor::tanh() const {
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = std::tanh(impl->data[i]);
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            double t = out_raw->data[i];
            input->grad[i] += (1.0 - t * t) * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}

Tensor Tensor::relu() const {
    std::vector<double> result_data(size());
    for (size_t i = 0; i < size(); ++i) {
        result_data[i] = impl->data[i] > 0.0 ? impl->data[i] : 0.0;
    }

    auto out_impl = std::make_shared<TensorImpl>(result_data, impl->shape);
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            input->grad[i] += (input->data[i] > 0.0 ? 1.0 : 0.0) * out_raw->grad[i];
        }
    };

    return Tensor(out_impl);
}
```

Add `#include <cmath>` to the top of `src/tensor.cpp` if not already present.

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_ops.cpp
git commit -m "feat: add tanh and relu activations with backward"
```

---

### Task 8: `sum()` Reduction With Backward

**Concept:** `sum()` reduces a tensor of any shape down to a single scalar by adding all its elements — this is what turns a per-example error vector into the single scalar number that `backward()` requires as its starting point (recall from Task 2: `backward()` only works on size-1 tensors). Its backward rule is the simplest of all: since every element contributes equally (with coefficient 1) to the sum, the gradient flowing back is just the output's single gradient value broadcast to every input position.

**Files:**
- Modify: `include/gradus/tensor.hpp` — declare `sum`
- Modify: `src/tensor.cpp` — define it
- Modify: `tests/test_tensor_ops.cpp` — append tests

**Interfaces:**
- Consumes: same as prior tasks.
- Produces: `Tensor::sum() const` — used by `mse_loss` in Task 11 (replaces the `ones_col` matmul trick from Task 6's test, which was only a workaround for not having `sum()` yet).

- [ ] **Step 1: Write the failing test — append to `tests/test_tensor_ops.cpp`**

```cpp
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `Tensor` has no member `sum`.

- [ ] **Step 3: Declare in `include/gradus/tensor.hpp`**

```cpp
    Tensor sum() const;
```

- [ ] **Step 4: Define in `src/tensor.cpp`**

```cpp
Tensor Tensor::sum() const {
    double total = 0.0;
    for (double v : impl->data) {
        total += v;
    }

    auto out_impl = std::make_shared<TensorImpl>(std::vector<double>{total}, std::vector<size_t>{1});
    out_impl->parents = {impl};

    auto input = impl;
    TensorImpl* out_raw = out_impl.get();
    out_impl->backward_fn = [input, out_raw]() {
        for (size_t i = 0; i < input->data.size(); ++i) {
            input->grad[i] += out_raw->grad[0];
        }
    };

    return Tensor(out_impl);
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/gradus/tensor.hpp src/tensor.cpp tests/test_tensor_ops.cpp
git commit -m "feat: add sum() reduction with backward"
```

---

### Task 9: Autograd Correctness — Diamond Graph Gradient Accumulation

**Concept:** Every operator has been tested in isolation so far, but real expressions reuse the same tensor multiple times — e.g. `a*a + a` uses `a` in two different places in the graph, forming a "diamond" shape (two paths from `a` that later converge). This matters because it's exactly the scenario that would break a naive implementation: if `backward_fn` used `=` instead of `+=` to write gradients, the second path to visit `a` would **overwrite** the first path's contribution instead of adding to it, silently producing a wrong gradient. Every operator you wrote in Tasks 4-8 already uses `+=` — this task is the test that proves that choice was necessary and correct, by exercising a graph where a node truly has more than one path back to a shared ancestor. It also exercises the topological sort's `visited` set: without it, `a`'s `backward_fn` would run twice.

**Files:**
- Create: `tests/test_autograd.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_autograd.cpp`

**Interfaces:**
- Consumes: `operator+`, `operator*`, `matmul`, `tanh`, `sum` from Tasks 4-8.

- [ ] **Step 1: Write the test — `tests/test_autograd.cpp`**

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "gradient_check.hpp"
#include "gradus/tensor.hpp"

using gradus::Tensor;
using gradus::testutil::numerical_gradient;

TEST_CASE("backward accumulates gradient across a diamond graph (a used twice)") {
    Tensor a(3.0);
    Tensor b = a * a;   // b = a^2
    Tensor c = b + a;   // c = a^2 + a
    c.backward();

    // dc/da = 2a + 1
    auto f = [](const std::vector<double>& x) { return x[0] * x[0] + x[0]; };
    auto numgrad = numerical_gradient(f, {3.0});

    REQUIRE(a.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
}

TEST_CASE("backward works through a composite graph mixing matmul, tanh, and sum") {
    Tensor x({1.0, -2.0}, {1, 2});
    Tensor w({0.5, -0.5, 1.0, 2.0}, {2, 2});
    Tensor y = x.matmul(w).tanh().sum();
    y.backward();

    auto f = [](const std::vector<double>& v) {
        double x0 = v[0], x1 = v[1];
        double w00 = v[2], w01 = v[3], w10 = v[4], w11 = v[5];
        double c0 = std::tanh(x0 * w00 + x1 * w10);
        double c1 = std::tanh(x0 * w01 + x1 * w11);
        return c0 + c1;
    };
    auto numgrad = numerical_gradient(f, {1.0, -2.0, 0.5, -0.5, 1.0, 2.0});

    REQUIRE(x.grad()[0] == Catch::Approx(numgrad[0]).epsilon(1e-4));
    REQUIRE(x.grad()[1] == Catch::Approx(numgrad[1]).epsilon(1e-4));
    REQUIRE(w.grad()[0] == Catch::Approx(numgrad[2]).epsilon(1e-4));
    REQUIRE(w.grad()[1] == Catch::Approx(numgrad[3]).epsilon(1e-4));
    REQUIRE(w.grad()[2] == Catch::Approx(numgrad[4]).epsilon(1e-4));
    REQUIRE(w.grad()[3] == Catch::Approx(numgrad[5]).epsilon(1e-4));
}
```

Add `#include <cmath>` to the top of this file for `std::tanh`.

- [ ] **Step 2: Add to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
  test_tensor_ops.cpp
  test_autograd.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Run and verify both tests pass**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS. If the diamond test fails, the most likely bug is a `+=` accidentally written as `=` in one of the Task 4-6 backward closures — go back and check.

- [ ] **Step 4: Commit**

```bash
git add tests/test_autograd.cpp tests/CMakeLists.txt
git commit -m "test: verify gradient accumulation across diamond and composite graphs"
```

---

### Task 10: `Linear` Layer and `MLP`

**Concept:** A `Linear` layer is nothing more than `output = input.matmul(weight) + bias` — everything it needs (the matrix multiply, the addition, and both operations' backward passes) already exists from Tasks 4 and 6. What's new here is *ownership*: a `Linear` layer owns its `weight` and `bias` as member `Tensor`s that persist across many forward/backward calls (unlike the temporary tensors created inside an expression, which get freed once nothing references them). `MLP` just chains several `Linear` layers together with a `tanh` non-linearity after each one — and it's precisely that non-linearity that lets a 2-layer network solve XOR, which a single `Linear` layer alone provably cannot (XOR isn't linearly separable — draw the 4 points and try to separate them with one straight line).

**Files:**
- Create: `include/gradus/nn.hpp`
- Create: `src/nn.cpp`
- Modify: `CMakeLists.txt` — add `src/nn.cpp` to the `gradus` library sources
- Create: `tests/test_nn.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_nn.cpp`

**Interfaces:**
- Consumes: `Tensor::matmul`, `operator+`, `.tanh()` from Tasks 4/6/7.
- Produces: `gradus::Linear{weight, bias, forward(x), parameters()}`, `gradus::MLP{forward(x), parameters()}` — both consumed by `SGD` in Task 11 and the XOR example in Task 12.

- [ ] **Step 1: Write the failing test — `tests/test_nn.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "gradus/nn.hpp"

using gradus::Linear;
using gradus::MLP;
using gradus::Tensor;

TEST_CASE("Linear forward produces the requested output shape") {
    Linear layer(3, 2);
    Tensor x({1.0, 2.0, 3.0}, {1, 3});
    Tensor y = layer.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 2});
}

TEST_CASE("Linear exposes weight and bias as parameters") {
    Linear layer(3, 2);
    auto params = layer.parameters();
    REQUIRE(params.size() == 2);
    REQUIRE(params[0].shape() == std::vector<size_t>{3, 2});  // weight
    REQUIRE(params[1].shape() == std::vector<size_t>{1, 2});  // bias
}

TEST_CASE("MLP forward produces the final layer's shape and is differentiable") {
    MLP mlp(2, {4, 1});
    Tensor x({0.5, -0.5}, {1, 2});
    Tensor y = mlp.forward(x);
    REQUIRE(y.shape() == std::vector<size_t>{1, 1});

    y.backward();
    bool any_param_has_nonzero_grad = false;
    for (auto& p : mlp.parameters()) {
        for (double g : p.grad()) {
            if (g != 0.0) {
                any_param_has_nonzero_grad = true;
            }
        }
    }
    REQUIRE(any_param_has_nonzero_grad);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `gradus/nn.hpp` doesn't exist.

- [ ] **Step 3: Write `include/gradus/nn.hpp`**

```cpp
#pragma once

#include <vector>
#include "gradus/tensor.hpp"

namespace gradus {

class Linear {
public:
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> parameters() const;

    Tensor weight;
    Tensor bias;
};

class MLP {
public:
    MLP(size_t in_features, std::vector<size_t> layer_sizes);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor> parameters() const;

private:
    std::vector<Linear> layers_;
};

}  // namespace gradus
```

- [ ] **Step 4: Write `src/nn.cpp`**

```cpp
#include "gradus/nn.hpp"

#include <cmath>
#include <random>

namespace gradus {

Linear::Linear(size_t in_features, size_t out_features)
    : weight(std::vector<double>(in_features * out_features, 0.0), {in_features, out_features}),
      bias(std::vector<double>(out_features, 0.0), {1, out_features}) {
    // Fixed seed: every training run (and every test) is reproducible.
    std::mt19937 rng(42);
    double bound = 1.0 / std::sqrt(static_cast<double>(in_features));
    std::uniform_real_distribution<double> dist(-bound, bound);

    for (double& v : weight.grad()) {
        (void)v;  // grad already zero-initialized; nothing to do here
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        const_cast<std::vector<double>&>(weight.data())[i] = dist(rng);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        const_cast<std::vector<double>&>(bias.data())[i] = dist(rng);
    }
}

Tensor Linear::forward(const Tensor& x) const {
    return x.matmul(weight) + bias;
}

std::vector<Tensor> Linear::parameters() const {
    return {weight, bias};
}

MLP::MLP(size_t in_features, std::vector<size_t> layer_sizes) {
    size_t prev = in_features;
    for (size_t sz : layer_sizes) {
        layers_.emplace_back(prev, sz);
        prev = sz;
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor out = x;
    for (const auto& layer : layers_) {
        out = layer.forward(out).tanh();
    }
    return out;
}

std::vector<Tensor> MLP::parameters() const {
    std::vector<Tensor> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

}  // namespace gradus
```

**Note on weight initialization:** `Tensor::data()` returns `const std::vector<double>&` (Task 2), so `Linear`'s constructor uses `const_cast` to fill in the randomized initial values after construction. This is a legitimate, contained use of `const_cast` (initializing a freshly-constructed object's own data isn't really "mutating a const value" in spirit, just working around an API that wasn't designed with in-place mutation in mind) — but it's a code smell worth noticing. If it bothers you once everything is working, a cleaner fix is adding a non-const `std::vector<double>& Tensor::data_mut()` accessor to `tensor.hpp`/`tensor.cpp` and using that here instead of `const_cast`. That's an optional refactor, not a required step.

- [ ] **Step 5: Add `src/nn.cpp` to `CMakeLists.txt`**

```cmake
add_library(gradus
  src/tensor.cpp
  src/nn.cpp
)
```

- [ ] **Step 6: Add `test_nn.cpp` to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
  test_tensor_ops.cpp
  test_autograd.cpp
  test_nn.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 7: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/gradus/nn.hpp src/nn.cpp CMakeLists.txt tests/test_nn.cpp tests/CMakeLists.txt
git commit -m "feat: add Linear layer and MLP"
```

---

### Task 11: `SGD` Optimizer and `mse_loss`

**Concept:** Everything up to now computes gradients; nothing yet *uses* them to actually improve a model. **Stochastic Gradient Descent** is the simplest possible learning rule: for every parameter, move it a small step (`learning_rate`) in the direction that *reduces* the loss — and since `grad` points in the direction of *steepest increase*, moving opposite to it (`data -= learning_rate * grad`) is what decreases the loss. `zero_grad()` matters because `backward()` always **accumulates** (`+=`) into `.grad` — if you didn't clear it before the next `backward()` call, gradients from every past training step would keep piling up, corrupting the update. **MSE (mean squared error)** — `mean((prediction - target)^2)` — is the standard loss for regression-style problems; squaring makes errors always positive (so they don't cancel out) and penalizes large errors more than small ones.

**Files:**
- Create: `include/gradus/optim.hpp`
- Create: `src/optim.cpp`
- Modify: `CMakeLists.txt` — add `src/optim.cpp`
- Create: `tests/test_optim.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_optim.cpp`

**Interfaces:**
- Consumes: `Tensor::operator-`, `operator*`, `.sum()` from Tasks 4/5/8; `Linear::parameters()` from Task 10.
- Produces: `gradus::SGD{step(), zero_grad()}`, `gradus::mse_loss(prediction, target) -> Tensor` — both consumed by the XOR training loop in Task 12.

- [ ] **Step 1: Write the failing test — `tests/test_optim.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

using gradus::Linear;
using gradus::mse_loss;
using gradus::SGD;
using gradus::Tensor;

TEST_CASE("mse_loss is zero when prediction equals target") {
    Tensor pred({1.0, 2.0}, {1, 2});
    Tensor target({1.0, 2.0}, {1, 2});
    Tensor loss = mse_loss(pred, target);
    REQUIRE(loss.item() == Catch::Approx(0.0));
}

TEST_CASE("mse_loss computes mean squared error") {
    Tensor pred({3.0}, {1, 1});
    Tensor target({1.0}, {1, 1});
    Tensor loss = mse_loss(pred, target);
    // (3-1)^2 / 1 = 4.0
    REQUIRE(loss.item() == Catch::Approx(4.0));
}

TEST_CASE("SGD single step reduces loss on a toy linear problem") {
    Linear layer(1, 1);
    // Force a known, deterministic starting point instead of the random init.
    const_cast<std::vector<double>&>(layer.weight.data())[0] = 0.0;
    const_cast<std::vector<double>&>(layer.bias.data())[0] = 0.0;

    Tensor x({2.0}, {1, 1});
    Tensor target({4.0}, {1, 1});

    SGD optimizer(layer.parameters(), 0.1);

    Tensor pred1 = layer.forward(x);
    double loss1 = mse_loss(pred1, target).item();

    optimizer.zero_grad();
    Tensor loss1_tensor = mse_loss(layer.forward(x), target);
    loss1_tensor.backward();
    optimizer.step();

    Tensor pred2 = layer.forward(x);
    double loss2 = mse_loss(pred2, target).item();

    REQUIRE(loss2 < loss1);
}
```

- [ ] **Step 2: Add to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
  test_tensor_ops.cpp
  test_autograd.cpp
  test_nn.cpp
  test_optim.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cmake --build build --parallel`
Expected: FAIL — `gradus/optim.hpp` doesn't exist.

- [ ] **Step 4: Write `include/gradus/optim.hpp`**

```cpp
#pragma once

#include <vector>
#include "gradus/tensor.hpp"

namespace gradus {

class SGD {
public:
    SGD(std::vector<Tensor> parameters, double learning_rate);

    void step();
    void zero_grad();

private:
    std::vector<Tensor> parameters_;
    double learning_rate_;
};

Tensor mse_loss(const Tensor& prediction, const Tensor& target);

}  // namespace gradus
```

- [ ] **Step 5: Write `src/optim.cpp`**

```cpp
#include "gradus/optim.hpp"

#include <stdexcept>

namespace gradus {

SGD::SGD(std::vector<Tensor> parameters, double learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {}

void SGD::step() {
    for (auto& p : parameters_) {
        for (size_t i = 0; i < p.size(); ++i) {
            const_cast<std::vector<double>&>(p.data())[i] -= learning_rate_ * p.grad()[i];
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : parameters_) {
        p.zero_grad();
    }
}

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("mse_loss: prediction and target shapes differ");
    }
    Tensor diff = prediction - target;
    Tensor squared = diff * diff;
    Tensor total = squared.sum();
    double n = static_cast<double>(prediction.size());
    return total * Tensor(1.0 / n);
}

}  // namespace gradus
```

**Note:** `SGD::step()` uses the same `const_cast` pattern flagged in Task 10 — same reasoning applies (optional `data_mut()` refactor, not required for v1).

- [ ] **Step 6: Add `src/optim.cpp` to `CMakeLists.txt`**

```cmake
add_library(gradus
  src/tensor.cpp
  src/nn.cpp
  src/optim.cpp
)
```

- [ ] **Step 7: Run to verify it passes**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/gradus/optim.hpp src/optim.cpp CMakeLists.txt tests/test_optim.cpp tests/CMakeLists.txt
git commit -m "feat: add SGD optimizer and mse_loss"
```

---

### Task 12: XOR Example and Integration Test

**Concept:** This is the payoff — every piece built so far (Tensor graph, operators, Linear, MLP, SGD, mse_loss) comes together into an actual training loop that solves a real (if tiny) learning problem. XOR is the canonical "hello world" of neural networks specifically *because* it's not linearly separable — a single-layer network mathematically cannot solve it, so successfully training one here is real proof the multi-layer/non-linear machinery works, not just a toy that would've worked with simpler math. Inputs and targets are encoded as `-1.0`/`1.0` rather than `0.0`/`1.0` because `tanh`'s output range is `(-1, 1)` — matching the target range to the activation's range makes training converge more easily (a `0`/`1` target would ask the network to reach `tanh`'s asymptotic extremes, which is much slower to learn).

**Files:**
- Create: `examples/xor.cpp`
- Modify: `CMakeLists.txt` — add the `xor_example` executable target
- Create: `tests/test_xor_integration.cpp`
- Modify: `tests/CMakeLists.txt` — add `test_xor_integration.cpp`

**Interfaces:**
- Consumes: `MLP`, `SGD`, `mse_loss` from Tasks 10/11.

- [ ] **Step 1: Write the failing test — `tests/test_xor_integration.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

using gradus::MLP;
using gradus::mse_loss;
using gradus::SGD;
using gradus::Tensor;

namespace {

std::vector<std::vector<double>> xor_inputs() {
    return {{-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}};
}

std::vector<double> xor_targets() {
    return {-1.0, 1.0, 1.0, -1.0};
}

double average_loss(const MLP& mlp) {
    auto inputs = xor_inputs();
    auto targets = xor_targets();
    double total = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor x(inputs[i], {1, 2});
        Tensor y(std::vector<double>{targets[i]}, {1, 1});
        Tensor pred = mlp.forward(x);
        total += mse_loss(pred, y).item();
    }
    return total / static_cast<double>(inputs.size());
}

}  // namespace

TEST_CASE("MLP trained on XOR substantially reduces average loss") {
    MLP mlp(2, {4, 1});
    SGD optimizer(mlp.parameters(), 0.1);

    double initial_loss = average_loss(mlp);

    auto inputs = xor_inputs();
    auto targets = xor_targets();
    for (int epoch = 0; epoch < 300; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor x(inputs[i], {1, 2});
            Tensor y(std::vector<double>{targets[i]}, {1, 1});
            Tensor pred = mlp.forward(x);
            Tensor loss = mse_loss(pred, y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }

    double final_loss = average_loss(mlp);
    REQUIRE(final_loss < initial_loss * 0.5);
}
```

- [ ] **Step 2: Add to `tests/CMakeLists.txt`**

```cmake
add_executable(gradus_tests
  test_tensor_basics.cpp
  test_gradient_check.cpp
  test_tensor_ops.cpp
  test_autograd.cpp
  test_nn.cpp
  test_optim.cpp
  test_xor_integration.cpp
)
target_link_libraries(gradus_tests PRIVATE gradus Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(gradus_tests)
```

- [ ] **Step 3: Run to verify it fails or passes on the first attempt**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure -R "XOR"`
Expected: this test doesn't depend on anything unimplemented, so it may pass immediately. If it fails (loss doesn't drop enough with the fixed seed of 42 from `Linear`'s constructor), try increasing `epoch` count to 500 first; if it still fails, the fixed seed may have landed in a bad local minimum — try seed `7` or `123` in `src/nn.cpp`'s `std::mt19937 rng(42)` line instead, and re-run. Do not loosen the `0.5` threshold as a first resort — that would hide a real convergence problem.

- [ ] **Step 4: Write `examples/xor.cpp`**

```cpp
#include <iostream>
#include "gradus/nn.hpp"
#include "gradus/optim.hpp"

int main() {
    using namespace gradus;

    std::vector<std::vector<double>> inputs = {
        {-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}};
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
        std::cout << inputs[i][0] << " XOR " << inputs[i][1]
                  << "  ->  " << pred.item() << "  (target " << targets[i] << ")\n";
    }

    return 0;
}
```

- [ ] **Step 5: Add the `xor_example` executable to `CMakeLists.txt`**

Add near the end, after `add_subdirectory(tests)`:

```cmake
add_executable(xor_example examples/xor.cpp)
target_link_libraries(xor_example PRIVATE gradus)
```

- [ ] **Step 6: Build everything and run both the test suite and the example**

Run: `cmake --build build --parallel && ctest --test-dir build --output-on-failure && ./build/xor_example`
Expected: all tests pass; the example prints decreasing loss and final predictions close to `-1`/`1`/`1`/`-1`.

- [ ] **Step 7: Commit**

```bash
git add examples/xor.cpp CMakeLists.txt tests/test_xor_integration.cpp tests/CMakeLists.txt
git commit -m "feat: add XOR training example and integration test"
```

---

### Task 13: CI Hardening, Benchmark, and Documentation Finalization

**Concept:** The engine itself is done and tested — this task is about making the repo trustworthy and readable to someone who has never seen the code (a recruiter, a YL reviewer, future-you in six months). AddressSanitizer catches memory bugs (use-after-free, buffer overruns) that a correctness test can pass right through; UndefinedBehaviorSanitizer catches things like signed integer overflow. Running them as a *separate* CI job (rather than folding sanitizer flags into the normal build) keeps the everyday test loop fast while still gating merges on memory safety. `docs/architecture.md`'s Mermaid diagram exists because "a graph-based autograd engine" is much easier to understand as a picture of the actual data flow than as prose.

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/architecture.md`
- Create: `CONTRIBUTING.md`
- Create: `.clang-format`
- Create: `benchmarks/bench_backward.cpp`
- Modify: `CMakeLists.txt` — add the optional benchmark target
- Modify: `HANDOFF.md` — update to reflect completed v1

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build --parallel
      - name: Test
        run: ctest --test-dir build --output-on-failure

  sanitize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure with sanitizers
        run: cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DGRADUS_ENABLE_SANITIZERS=ON
      - name: Build
        run: cmake --build build-san --parallel
      - name: Test under ASan/UBSan
        run: ctest --test-dir build-san --output-on-failure

  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install clang-format
        run: sudo apt-get update && sudo apt-get install -y clang-format
      - name: Check formatting
        run: |
          find include src tests examples benchmarks -name '*.hpp' -o -name '*.cpp' \
            | xargs clang-format --dry-run --Werror
```

- [ ] **Step 2: Write `.clang-format`**

```yaml
BasedOnStyle: Google
ColumnLimit: 100
IndentWidth: 4
AccessModifierOffset: -2
```

- [ ] **Step 3: Run clang-format locally and fix any violations**

Run: `find include src tests examples -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i`
Expected: files reformatted in place to match `.clang-format`. Re-run the test suite afterward to confirm formatting didn't break anything: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`

- [ ] **Step 4: Write `benchmarks/bench_backward.cpp`**

```cpp
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
```

- [ ] **Step 5: Add the optional benchmark target to `CMakeLists.txt`**

Add near the end:

```cmake
if(GRADUS_BUILD_BENCHMARKS)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(gradus_bench benchmarks/bench_backward.cpp)
  target_link_libraries(gradus_bench PRIVATE gradus benchmark::benchmark)
endif()
```

- [ ] **Step 6: Build and run the benchmark once to get a real README number**

Run: `cmake -S . -B build-bench -DGRADUS_BUILD_BENCHMARKS=ON && cmake --build build-bench --parallel && ./build-bench/gradus_bench`
Expected: prints a table with `BM_MLPForwardBackward` and a `Time`/`CPU`/`Iterations` column — note the `Time` value for the README.

- [ ] **Step 7: Write `docs/architecture.md`**

```md
# gradus Architecture

## Data flow

\`\`\`mermaid
flowchart LR
    subgraph Forward pass
        A[Tensor a] -->|operator*, matmul, tanh, ...| B[Tensor b]
        B --> C[Tensor c]
    end
    C -->|"c.backward()"| D[Topological sort]
    D --> E[Walk nodes in reverse]
    E -->|"each node's backward_fn"| F[Accumulate grad into parents]
    F -.->|grad flows back to| A
\`\`\`

Every operation (`operator+`, `operator*`, `matmul`, `tanh`, `relu`, `sum`)
allocates a new `TensorImpl` node, records its inputs as `parents`, and
stores a `backward_fn` closure implementing that operation's local
chain-rule step. `Tensor::backward()` does not know any calculus itself —
it topologically sorts the graph (parents before children) and calls each
node's `backward_fn` in reverse order, so gradient contributions flow from
the loss all the way back to the original leaf tensors.

## Decisions log

- **Single unified `Tensor`, not a separate scalar `Value` type.** A scalar
  is just a `Tensor` with shape `{1}`. Two parallel implementations would
  duplicate every operator's forward+backward logic for no benefit at this
  project's scale.
- **No `requires_grad` flag.** Every tensor always builds a graph node.
  Simpler, and v1's scope (XOR-scale examples) never needs to skip
  gradient tracking for performance.
- **No broadcasting.** Every elementwise operator requires an exact shape
  match; mismatches throw `std::invalid_argument`. Keeps every backward
  pass a simple parallel loop with no reduction-over-broadcast-dims logic.
- **Ownership via `shared_ptr<TensorImpl>`, closures capture the owning
  node by raw pointer.** Parent nodes are captured by `shared_ptr` (keeps
  them alive for as long as the graph exists); a node's own `backward_fn`
  captures *itself* by raw pointer, not `shared_ptr`, to avoid a reference
  cycle that `shared_ptr` could never free.
- **`-1.0`/`1.0` encoding for the XOR example**, not `0.0`/`1.0` — matches
  `tanh`'s output range, which trains faster than asking the network to
  reach `tanh`'s asymptotic extremes.
```

- [ ] **Step 8: Write `CONTRIBUTING.md`**

```md
# Contributing to gradus

## Building and testing

\`\`\`bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
\`\`\`

## Workflow

- Every behavior change ships with a test — no "add tests later" PRs.
- TDD order: write the failing test, watch it fail, write the minimal
  implementation, watch it pass, then commit.
- Branch naming: `feat/<short-description>`, `fix/<short-description>`.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `chore:`), English, imperative mood.
- Format before committing: `clang-format -i` on any touched `.hpp`/`.cpp`
  file (see `.clang-format`).
- CI must be green (build, test, sanitizer job, format check) before merge.

## Adding a new Tensor operator

Follow the pattern established by `operator+`/`tanh`/`matmul` in
`src/tensor.cpp`: compute the forward result, allocate a new `TensorImpl`,
record `parents`, and attach a `backward_fn` implementing that operation's
local chain-rule step. Add a numerical-gradient-check test (see
`tests/gradient_check.hpp`) alongside the forward-value test.
```

- [ ] **Step 9: Update `HANDOFF.md`**

```md
# Handoff — gradus

Son güncelleme: 2026-08-24, güncelleyen: Claude Sonnet 5

## Şu an ne yapılıyor
v1 tamamlandı: Tensor/autograd çekirdeği, Linear/MLP, SGD, XOR örneği,
CI (build+test+sanitizer+format), docs (architecture.md, CONTRIBUTING.md).

## Sıradaki somut adım
v1 kapsam dışı bırakılan genişletmelerden biri seçilip ayrı bir plan
yazılabilir (bkz. spec §9 Extensibility): GPU backend, ek katmanlar,
Python binding.

## Bilinmesi gerekenler
- `Tensor::data()` const döndürüyor; `Linear`/`SGD` içeride `const_cast`
  kullanıyor (bkz. Task 10/11 notları) — istenirse `data_mut()` eklenip
  temizlenebilir, v1 için gerekli değil.
- XOR entegrasyon testi sabit seed (42) ile deterministik; eğer bir
  değişiklik sonrası flaky hale gelirse önce epoch sayısını, sonra seed'i
  değiştirmeyi dene (bkz. Task 12 notu) — threshold'u gevşetme.

## İlgili dosyalar
- docs/superpowers/plans/2026-08-24-gradus-v1-autograd-engine.md
- docs/superpowers/specs/2026-08-24-gradus-autograd-engine-design.md
- docs/architecture.md

## Son 3 commit
- (implementasyon sırasında güncellenecek)
```

- [ ] **Step 10: Commit**

```bash
git add .github/workflows/ci.yml .clang-format benchmarks/bench_backward.cpp CMakeLists.txt docs/architecture.md CONTRIBUTING.md HANDOFF.md
git commit -m "chore: add CI sanitizer/format jobs, benchmark, and architecture docs"
```

---

## Post-Plan Checklist (Definition of Done)

- [ ] Every task's tests pass locally (`ctest --test-dir build --output-on-failure`)
- [ ] Sanitizer build passes (`-DGRADUS_ENABLE_SANITIZERS=ON`)
- [ ] `xor_example` runs and visibly converges
- [ ] CI green on GitHub Actions after first push
- [ ] README Quick Start works from a clean clone
- [ ] `HANDOFF.md` reflects final state
- [ ] No `Co-Authored-By: Claude` in any commit
