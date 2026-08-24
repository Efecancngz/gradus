# Contributing to gradus

## Building and testing

```bash
cmake -S . -B build -G Ninja
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

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
