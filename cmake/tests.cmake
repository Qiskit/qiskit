enable_testing()
# Run python code tests
find_program(PYTHON "python")
if (NOT PYTHON)
    message(FATAL_ERROR "Couldn't find Python in your system. Please, install it and try again.")
endif()

add_test(NAME qiskit_python
    COMMAND ${PYTHON} -m unittest discover -s test -v
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
