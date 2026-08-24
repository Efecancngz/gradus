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
```bash
git clone <repo-url>
cd gradus
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/xor_example
```

## MNIST example
Trains the same from-scratch engine on the full MNIST dataset (60k train /
10k test, real handwritten digits) using a fused softmax + cross-entropy
loss. Measured result on this repo: **89.5% test accuracy** after 5 epochs
(~43s/epoch in a Release build).

```bash
bash scripts/download_mnist.sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/mnist_example
```

## Documentation
- [Architecture](docs/architecture.md)

## License
MIT — see [LICENSE](LICENSE)
