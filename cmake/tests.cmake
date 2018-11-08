enable_testing()
# Run python code tests
find_program(PYTHON "python")
if (NOT PYTHON)
    message(FATAL_ERROR "Couldn't find Python in your system. Please, install it and try again.")
endif()

add_test(NAME qiskit_python
    COMMAND stestr run --concurrency 2
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
