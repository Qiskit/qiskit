cmake_minimum_required (VERSION 3.5 FATAL_ERROR)
file(STRINGS "qiskit/VERSION.txt" VERSION_NUM)
project (Qiskit VERSION ${VERSION_NUM} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Verbose!!
set(CMAKE_VERBOSE_MAKEFILE True)
set(QISKIT_VERSION ${PROJECT_VERSION} CACHE INTERNAL "VERSION OF QISKIT")

include(cmake/defaults.cmake)

if (ENABLE_TARGETS_NON_PYTHON)
    # Add C++ Qiskit Simulator
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/qasm-simulator-cpp)
endif()

include(cmake/tests.cmake)

if (ENABLE_TARGETS_QA)
    # Add QA tools: lint, style and sphinx
    include(cmake/code-qa.cmake)
    add_lint_target()
    add_code_style_target()
    add_doc_target(html "./" "_build/")
    add_coverage_target()
endif()
