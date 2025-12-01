cmake_minimum_required (VERSION 3.27)
project (Qiskit)

# We are using CMake to build our C API test suite, and we leverage ctest to run
# those tests.
enable_testing ()

# Generally include the directory for the ``qiskit.h`` file.
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/dist/c/include)
# The remaining CMake configuration is done inside of the C API test suite.
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/test/c)
