# gradus

A reverse-mode automatic differentiation (autograd) engine written from scratch in C++17.

## Why
This project exists to learn — not to replace PyTorch, and not as a library
meant for anyone to depend on. Every existing project in this author's
portfolio uses AI/ML through an API; none of them show an understanding of
how a neural network actually learns. `gradus` implements backpropagation by
hand: a `Tensor` class that builds a computation graph as operations run, and
a `backward()` that walks that graph in reverse applying the chain rule at
every node. A mature library (libtorch, dlib) exists for every one of these
operations — the point of `gradus` is writing them anyway.

**This is a portfolio/learning project, not production software.** No
batching, no GPU, no performance guarantees, no API stability — the goal was
depth of understanding on a small, honest scope, not a general-purpose tool.
If you're evaluating this as a hire signal: read `docs/architecture.md`'s
decisions log for the *reasoning* behind each choice, not just the code.

## A real bug, on purpose left visible
The first working version of the XOR example didn't learn — loss plateaued
at a constant value regardless of epoch count. The cause: every `Linear`
layer seeded its random weight initialization with the same fixed constant,
so in a 2-layer network the hidden units never broke symmetry with each
other (a 4-neuron layer behaved like 1). Diagnosed by tracing the per-epoch
loss curve rather than guessing at hyperparameters; fixed with a
per-layer-incrementing seed. Full writeup, including the exact numbers, in
`docs/architecture.md`'s decisions log — kept there deliberately instead of
being quietly fixed and forgotten, because finding and fixing it was as much
the point of this project as the code that came after it.

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
