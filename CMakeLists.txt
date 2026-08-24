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
  src/nn.cpp
  src/optim.cpp
  src/mnist_loader.cpp
)
target_include_directories(gradus PUBLIC include)

if(GRADUS_ENABLE_SANITIZERS)
  target_compile_options(gradus PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(gradus PUBLIC -fsanitize=address,undefined)
endif()

enable_testing()
add_subdirectory(tests)

add_executable(xor_example examples/xor.cpp)
target_link_libraries(xor_example PRIVATE gradus)

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
