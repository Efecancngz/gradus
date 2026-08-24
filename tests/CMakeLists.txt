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
