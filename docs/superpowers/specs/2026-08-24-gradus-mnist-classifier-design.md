# gradus v2 — MNIST Classifier Design

Date: 2026-08-24
Status: Approved (brainstorming), pending implementation plan

## 1. Problem Definition

v1 (autograd engine + XOR demo) proved the engine works, but XOR is a toy —
4 examples, 2 inputs, hand-picked and trivially memorizable. It doesn't
demonstrate the engine on a real, non-trivial dataset, and doesn't exercise
classification (as opposed to regression-shaped) training. v2 closes that
gap: train the same from-scratch engine on the full MNIST handwritten-digit
dataset (60,000 training images, 10,000 test images, 784 pixels each, 10
classes) and report real test-set accuracy.

## 2. Requirements

**Functional**
- Load the full MNIST dataset from a CSV mirror (60k train + 10k test rows,
  `label,pixel0,...,pixel783`).
- Add a fused `softmax_cross_entropy_loss` primitive to `Tensor` (its own
  `backward_fn`, not composed from existing ops).
- Extend `MLP` with an `activate_output` flag so the final layer can emit
  raw logits instead of tanh-squashed values, without changing XOR's
  existing behavior (default stays `true`).
- Ship `examples/mnist.cpp`: builds a `784 → 128 → 10` MLP
  (`activate_output=false`), trains with `SGD` on the full training set,
  reports per-epoch loss/timing, and reports test-set accuracy (via
  argmax over the 10 output logits) after training.

**Non-functional**
- No batching, no SIMD, no BLAS — this stays consistent with v1's
  single-example-at-a-time architecture; `matmul` is unchanged.
- Training is expected to take on the order of tens of seconds per epoch
  in a Release build; `examples/mnist.cpp` must print progress per epoch
  so a long-running process doesn't look hung.
- CI must stay hermetic: no test may depend on the downloaded MNIST files
  or network access. `ctest` must pass on a machine with no `data/`
  directory present.

**Business / portfolio requirements**
- Same audience as v1 (GitHub visitors, YL reviewers) — this is the
  "prove it on something real" follow-up to v1's from-scratch engine.
  Success criterion: `examples/mnist.cpp` runs end to end on a clean
  checkout (after running the download script) and reports a test
  accuracy meaningfully above chance (>>10%) — a working demo, not a
  state-of-the-art result.

## 3. Technology & Data Source

- **Data source:** `https://data.pjreddie.com/files/mnist_train.csv` (60k
  rows, ~109MB) and `mnist_test.csv` (10k rows, ~18MB). Verified live
  (HTTP 200, correct `label,pixel...` row format) before committing to
  this choice — not assumed from memory.
- **From-scratch vs. reuse (build-vs-buy):** the MNIST *file format itself*
  is not parsed from the original binary IDX format. That would be a
  tangential file-format-parsing exercise unrelated to this project's
  actual learning goal (autograd/NN internals), so a pre-converted CSV
  mirror is used instead — a deliberate "buy" decision, not laziness. The
  *training code* (loss, gradients, optimizer) remains 100% from-scratch,
  same as v1.
- **Storage:** downloaded CSVs live in a gitignored `data/` directory,
  fetched via `scripts/download_mnist.sh`. Never committed (127MB is
  much too large for the repo, and the data is trivially re-fetchable).

## 4. Architecture

### 4.1 `softmax_cross_entropy_loss` — a fused primitive, not a composition

Softmax followed by cross-entropy, differentiated separately, requires a
full Jacobian for the softmax step — expensive and numerically fragile.
Differentiated *together*, the gradient collapses to a single subtraction:

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
CE(x, c)     = -log(softmax(x)_c)            (c = true class index)

dCE/dx_i     = softmax(x)_i - [i == c]        (one-hot indicator)
```

This is exactly the fusion every production framework performs
(`torch.nn.CrossEntropyLoss` takes raw logits for this reason — it is not
a shortcut, it is the standard approach). Implemented as a new `Tensor`
method (`tensor.hpp`/`tensor.cpp`), alongside `matmul`/`tanh`/etc., since
it needs a custom `backward_fn` rather than being buildable from existing
ops — architecturally a primitive, like the rest of that file's contents.

Forward pass uses the log-sum-exp trick for numerical stability (subtract
`max(x)` before exponentiating) — without it, `exp()` of a moderately
large logit overflows `double` silently into `inf`.

### 4.2 `MLP::activate_output`

```cpp
MLP(size_t in_features, std::vector<size_t> layer_sizes, bool activate_output = true);
```

`forward()` applies `.tanh()` after every layer except the last one when
`activate_output` is `false`. Default (`true`) preserves v1's exact
existing behavior — XOR's construction and tests are untouched. The MNIST
example constructs its `MLP` with `activate_output=false` since
`softmax_cross_entropy_loss` operates on raw logits.

### 4.3 `mnist_loader`

```cpp
struct MnistDataset {
    std::vector<std::vector<double>> images;  // each 784, normalized to [-1, 1]
    std::vector<int> labels;                  // 0-9
};
MnistDataset load_mnist_csv(const std::string& path);
```

Normalizes each pixel via `(value / 127.5) - 1.0`, matching the `[-1, 1]`
convention already established by the XOR example (keeps inputs
zero-centered, which trains better with `tanh` hidden layers than raw
`[0, 255]` or `[0, 1]` values would).

### 4.4 `examples/mnist.cpp`

Loads both CSVs via `mnist_loader`, builds `MLP(784, {128, 10}, false)`,
trains one example at a time (no batching, consistent with `examples/xor.cpp`'s
pattern) with `SGD`, printing average training loss and elapsed time per
epoch. After training, runs a forward pass over the full test set, takes
`argmax` of each output row, and reports accuracy against `labels`.

## 5. Data Flow

1. `scripts/download_mnist.sh` fetches both CSVs into `data/` (run once,
   manually, before using the example).
2. `mnist_loader::load_mnist_csv` parses each row into a normalized
   `std::vector<double>` (784 entries) plus an integer label.
3. Per training example: `Tensor x(image, {1, 784})` →
   `mlp.forward(x)` (raw logits, shape `{1, 10}`) →
   `logits.softmax_cross_entropy_loss(label)` (scalar) → `zero_grad()` →
   `backward()` → `step()`. Identical shape to the XOR training loop.
4. Evaluation: same forward pass, no loss/backward — just
   `argmax(logits.data())` compared to the true label, accumulated into
   an accuracy percentage.

## 6. Error Handling

- `load_mnist_csv` throws `std::runtime_error` if the file can't be
  opened (with the attempted path in the message) or if a row doesn't
  have exactly 785 comma-separated fields — a malformed/truncated
  download should fail loudly, not silently produce a garbage dataset.
- `softmax_cross_entropy_loss` throws `std::invalid_argument` if
  `target_class` is outside `[0, size())` — same "usage error → exception"
  convention as every other op's shape-mismatch check.

## 7. Testing Strategy

- **Unit test for the new operator:** `softmax_cross_entropy_loss`
  verified via the same numerical-gradient-checking technique as every
  other op (`gradient_check.hpp`), on small synthetic logit vectors — no
  MNIST data required, runs in CI like everything else.
- **`MLP::activate_output` regression check:** a test confirming the
  existing XOR-shaped construction (`activate_output=true`, the default)
  still produces `tanh`-bounded output, plus a test confirming
  `activate_output=false` produces unbounded raw logits.
- **No CI dependency on real MNIST data.** `examples/mnist.cpp` is a
  manually-run demo, not part of `ctest` — consistent with keeping the
  test suite hermetic (no network access, no multi-hundred-MB fixtures).
  This is a deliberate scope boundary, not an oversight.

## 8. Out of Scope (v2)

- Batching / mini-batch SGD (would require broadcasting-aware `matmul`,
  a nontrivial extension of v1's shape rules — a candidate for v3).
- Convolutional layers (MNIST is solvable with a plain MLP at reasonable
  accuracy; convolution is a separate, larger feature).
- Any performance optimization (SIMD, threading, BLAS) — accepted as a
  known, documented limitation for v2; `examples/mnist.cpp` prints timing
  so this tradeoff is visible, not hidden.

## 9. Documentation Set (v2)

- `README.md` — add a short "MNIST example" section under Quick Start,
  pointing at `scripts/download_mnist.sh` + `examples/mnist.cpp`.
- `docs/architecture.md` — decisions log gains the
  softmax+cross-entropy-fusion and `activate_output` entries.
- `scripts/download_mnist.sh` — the download script itself, with a
  one-line comment naming the source URLs.
- `.gitignore` — add `data/`.
