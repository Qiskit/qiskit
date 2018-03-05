# CMake config file to build SWIG wrapper and library for the C++ simulator

project(qiskit_simulator_swixg CXX)

set(QISKIT_SIMULATOR_SRC_DIR "${PROJECT_SOURCE_DIR}/..")
set(QISKIT_SIMULATOR_SRC_SWIG "${PROJECT_SOURCE_DIR}/qiskit_simulator_swig.i")

# Generic checks and directives for using SWIG.
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
set(CUSTOM_INCLUDE_DIRECTORIES
    ${QISKIT_SIMULATOR_SRC_DIR}
    ${QISKIT_SIMULATOR_SRC_DIR}/backends
    ${QISKIT_SIMULATOR_SRC_DIR}/engines
    ${QISKIT_SIMULATOR_SRC_DIR}/utilities
    ${QISKIT_SIMULATOR_SRC_DIR}/third-party/headers)

set_source_files_properties(${QISKIT_SIMULATOR_SRC_SWIG} PROPERTIES CPLUSPLUS True)
set(CMAKE_SWIG_FLAGS "-I${QISKIT_SIMULATOR_SRC_DIR}")

# Target for the SWIG shared library ("_qiskit_simulator_swig")
if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
    swig_add_module(qiskit_simulator_swig python ${QISKIT_SIMULATOR_SRC_SWIG})
else()
    # swig_add_module() is deprecated in cmake 3.8.0.
    swig_add_library(qiskit_simulator_swig SHARED LANGUAGE python SOURCES
        ${QISKIT_SIMULATOR_SRC_SWIG})
endif()

target_include_directories(_qiskit_simulator_swig PRIVATE ${QISKIT_SIMULATOR_SRC_DIR})
set_target_properties(_qiskit_simulator_swig PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SWIG_OUTDIR})

swig_link_libraries(qiskit_simulator_swig ${PYTHON_LIBRARIES} qiskit_simulator)
