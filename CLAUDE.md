# gradus

C++17 autograd engine, portfolio + learning project. See `docs/architecture.md`
for the design and decisions log; see `docs/superpowers/specs/` for the
original design spec.

## Run commands
- Build: `cmake -S . -B build -G Ninja && cmake --build build --parallel`
- Test: `ctest --test-dir build --output-on-failure`
- Sanitizer build: `cmake -S . -B build-san -DGRADUS_ENABLE_SANITIZERS=ON && cmake --build build-san && ctest --test-dir build-san`
- Benchmarks (optional): `cmake -S . -B build-bench -DGRADUS_BUILD_BENCHMARKS=ON && cmake --build build-bench && ./build-bench/gradus_bench`

## Local toolchain note
No system-wide compiler was preinstalled when this project started; built
with MSYS2's `mingw-w64-x86_64-gcc`/`cmake`/`ninja` (installed to
`C:\msys64\mingw64\bin`, not on PATH by default — add it to PATH or pass
full paths). Any C++17 compiler works; this is just what's on this machine.

## Why these choices
- Single unified `Tensor` type (no separate scalar `Value` class) — avoids
  duplicating the graph/backward machinery for two node types.
- No `requires_grad` flag — every tensor always tracks gradients (v1 scope
  never needs to disable it).
- No broadcasting — every elementwise op requires exact shape match; keeps
  every backward pass a simple loop.

See `HANDOFF.md` for current status.
