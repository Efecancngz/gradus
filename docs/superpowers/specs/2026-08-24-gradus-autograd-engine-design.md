# gradus — Autograd Engine Design

Date: 2026-08-24
Status: Approved (brainstorming), pending implementation plan

## 1. Problem Definition

Not a business problem — this is a portfolio/learning project (exception under
standard §0.4). The gap it closes: existing projects (Jobera, testcrafter,
Product Locator, ScreenTracker, Nexus Remote Center) are all
web/backend/QA-automation work that *uses* AI APIs as a black box. None of
them demonstrate understanding of how a neural network actually learns
(backpropagation, gradient computation). `gradus` closes that gap and doubles
as prep material for the Bütünleşik YL (Yapay Zeka Mühendisliği) application,
whose outcome is still pending.

## 2. Requirements

**Functional**
- Build arbitrary computation graphs out of `Tensor` operations (`+`, `*`
  elementwise, matmul, `tanh`, `relu`, `sum`).
- Compute gradients for any graph via reverse-mode automatic differentiation
  (`backward()`).
- Provide `Linear` and `MLP` layers built on top of `Tensor`.
- Provide a basic `SGD` optimizer.
- Ship a runnable example (`examples/xor.cpp`) that trains an MLP on XOR and
  demonstrably reduces loss over training steps.

**Non-functional**
- Runs on CPU only, single-threaded for v1 (no GPU, no distributed training
  — out of scope, see §7 extensibility).
- Correctness matters far more than raw speed for v1; Google Benchmark is
  wired up for future optimization work but v1 has no specific performance
  target.
- Must build cleanly with ASan/UBSan enabled in CI — memory safety is a
  first-class demonstration goal, not an afterthought.

**Business / portfolio requirements**
- Primary audience: GitHub visitors evaluating the author's engineering
  skill (recruiters, YL admissions reviewers), secondarily the author
  himself as a learning artifact.
- Success criteria: (a) repo builds and passes CI from a clean clone with
  the Quick Start in README, (b) XOR example runs and visibly converges,
  (c) test suite covers every operator's forward and backward pass via
  numerical gradient checking, (d) README/architecture docs are good enough
  that a reader unfamiliar with autograd can follow the design.

## 3. Technology & Variable Choices

- **Language:** C++17 (portfolio target: modern C++, not legacy style).
- **Build system:** CMake (`FetchContent` for dependencies — no manual
  vendoring, no system-wide package manager requirement, so `git clone` +
  `cmake --build` works standalone).
- **Test framework:** Catch2 (v3). Chosen over GoogleTest: lighter setup,
  more idiomatic for a single-author portfolio project, BDD-style
  `TEST_CASE`/`SECTION` reads better in a project meant to be read by
  reviewers. GoogleTest's main advantage (team/enterprise familiarity)
  doesn't apply to a solo project.
- **Benchmark:** Google Benchmark, wired into CI but not gating — used to
  publish a "backward pass throughput" number in the README.
- **Sanitizers:** AddressSanitizer + UndefinedBehaviorSanitizer, a dedicated
  CI job (separate from the regular test job to keep the standard job fast).
- **Static analysis / formatting:** clang-format + clang-tidy, run in CI.
- **CI:** GitHub Actions — `build`, `test`, `sanitize`, `lint` jobs.
- **License:** MIT (standard §9 default).

## 4. From-Scratch Justification (standard §0.4 exception)

This is explicitly a learning/portfolio project — the whole point is writing
the autograd graph and backprop by hand rather than depending on an existing
library (e.g. libtorch, dlib). This exception and its reasoning goes in
`README.md` §Why per §0.4's requirement that the rationale be written down,
not just implied.

## 5. Architecture

**Single unified `Tensor` abstraction**, not two parallel scalar/tensor
implementations. A scalar is just a 1-element `Tensor` (shape `{1}`). This
mirrors how PyTorch itself is designed and avoids maintaining two graph
engines with duplicated backward-pass logic — the alternative (a separate
`Value` scalar class alongside a `Tensor` class) was considered and rejected
for exactly that duplication reason.

```
Tensor
├── data: std::vector<double>       (flat buffer)
├── shape: std::vector<size_t>
├── grad: std::vector<double>       (same size as data)
├── parents: vector<shared_ptr<TensorImpl>>   (graph edges)
└── backward_fn: std::function<void()>        (local chain-rule step)
```

Ownership: `Tensor` is a thin handle wrapping `shared_ptr<TensorImpl>` (graph
nodes are shared across multiple consumers, so reference-counted ownership is
required — this is the standard micrograd/PyTorch pattern, not a novel
choice).

**Autograd engine:** `backward()` on a scalar-shaped output tensor performs a
topological sort of the graph (DFS, standard Kahn's-algorithm-style
ordering), then walks nodes in reverse order calling each node's
`backward_fn`, accumulating gradients into parent tensors.

**NN layer:** `Linear(in_features, out_features)` owns weight + bias
`Tensor`s (marked `requires_grad`), computes `x * W + b` via the same graph
mechanism. `MLP` chains `Linear` + activation layers. `SGD::step()` reads
`.grad` off every parameter tensor and updates `.data` in place, then
`.zero_grad()` clears gradients for the next iteration.

## 6. Data Flow

1. User builds an expression: `auto y = mlp.forward(x);`
2. Every operator call (`operator*`, `.tanh()`, etc.) allocates a new
   `TensorImpl`, records its parent(s), and stores a backward closure
   capturing whatever local values it needs for the chain rule.
3. `auto loss = mse(y, target); loss.backward();` triggers topological sort
   + reverse traversal, filling `.grad` on every tensor in the graph,
   including the `Linear` layers' weight/bias tensors.
4. `optimizer.step()` updates weights using the now-populated gradients.
5. `optimizer.zero_grad()` resets gradients before the next forward pass.

## 7. Error Handling

- Shape mismatches (e.g. incompatible matmul dimensions) throw
  `std::invalid_argument` with a message naming both shapes — this is a
  usage error, not an invariant violation, so it's an exception rather than
  an assertion (standard §19 assertion vs. error-handling split).
- Calling `.backward()` on a non-scalar tensor throws
  `std::invalid_argument` (reverse-mode AD requires a scalar root; this
  matches PyTorch's own behavior).
- Internal invariant violations (e.g. a `TensorImpl` with mismatched
  `data`/`grad` buffer sizes) use `assert()` — these should never happen if
  the library is internally correct, so a crash in debug builds is
  appropriate rather than a recoverable exception.

## 8. Testing Strategy

- **Per-operator unit tests** (Catch2): for every operator, verify both the
  forward computation and the backward gradient. Gradient correctness is
  verified via **numerical gradient checking** — perturb each input by a
  small epsilon, compute the numerical derivative via finite differences,
  and assert it matches the analytically computed gradient within
  tolerance. This is the standard verification technique for autograd
  implementations (used by PyTorch's own `gradcheck`).
- **Integration test:** train the XOR MLP for N steps and assert loss at
  step N is meaningfully lower than loss at step 0 (proves the whole
  pipeline — graph construction, backward, optimizer — works end to end,
  not just individual operators in isolation).
- **Sanitizer job:** the full test suite re-run under ASan+UBSan in CI.
- Per standard §6: every behavior change ships with a test; no separate
  "add tests later" pass.

## 9. Extensibility (documented, not built in v1)

- GPU backend (CUDA) — v2+, would require abstracting the storage backend.
- Additional layers (Conv, BatchNorm, more losses).
- Broader operator set (division, pow, more activations) as needed by
  future examples.
- Python bindings (pybind11) if the project outgrows pure-C++ demos.

These are called out in README's "Roadmap" / "Contributing" section so the
project reads as an actively extensible base, not a closed one-off exercise.

## 10. Out of Scope (v1)

- Multi-threading / SIMD optimization (Benchmark is wired up to measure
  this later, not to satisfy a target now).
- Model serialization / persistence.
- Any GPU support.
- Any tokenizer/LLM-adjacent functionality — that's a separate candidate
  project (mini LLM inference), not part of gradus.
