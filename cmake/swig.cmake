# Creates targets for the generation of the non-python components that use
# SWIG.

# Set the output directory for the files produced by SWIG.
# NOTE: each component should ensure that the final shared library is generated
# in ${CMAKE_SWIG_OUTDIR}
set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR}/swig)

# Add targets.
add_custom_target(swig_build)
add_custom_target(swig_install)
add_dependencies(swig_install swig_build)

# Add each component to the "swig_build" target.
# QISKit C++ Simulator
add_subdirectory(${PROJECT_SOURCE_DIR}/src/qiskit-simulator/src/swig)
add_dependencies(swig_build _qiskit_simulator_swig)

