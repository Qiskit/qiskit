cmake_minimum_required (VERSION 3.5 FATAL_ERROR)
project (QISKit VERSION 0.5.0 LANGUAGES CXX)

# Verbose!!
set(CMAKE_VERBOSE_MAKEFILE True)
set(QISKIT_VERSION ${PROJECT_VERSION} CACHE INTERNAL "VERSION OF QISKIT")

include(cmake/defaults.cmake)

# Add Toolchain utilities for C++ sources
include(cmake/toolchain-utils.cmake)

# Add C++ Qiskit Simulator
add_subdirectory(${PROJECT_SOURCE_DIR}/src/qiskit-simulator)

# Add tests and linters
include(cmake/code-qa.cmake)
add_lint_target()
add_code_style_target()
add_doc_target(html "./" "_build/")


# Add Python distributable package generator
include(cmake/python-build.cmake)
add_pypi_package_target(pypi_package both)
