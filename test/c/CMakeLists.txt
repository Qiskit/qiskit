# This file is inspired by the description of handling a larger number of tests
# provided here:
# https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html
cmake_minimum_required (VERSION 3.27)

# All files in this directory (non-recursively! use GLOB_RECURSE if that becomes
# necessary) which match the pattern `test_*.c` are discovered as test files by
# the command below. All paths are stored relative to the location of this file.
file (
    GLOB discovered_tests
    LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    test_*.c
)

# The following creates a test driver program which gathers all discovered test
# files into a single executable. This has the benefit that a single test
# executable is built and run rather than invididual ones for every individual
# test file.
create_test_sourcelist (source_files test_driver.c ${discovered_tests})

add_library(common STATIC common.c)
# Actually define the test driver program executable to be built...
add_executable (test_driver ${source_files})
# ...include the location of the header file...
target_include_directories (test_driver PRIVATE ${CMAKE_SOURCE_DIR})
# ...and linked with the qiskit library.
target_link_libraries (test_driver ${qiskit} common)

# On MSVC we need to link the Python dll, we search it here and adjust the PATH in the tests below
if (MSVC)
    # Store the directory where the qiskit_cext.dll is located.
    get_filename_component(qiskit_dll_dir ${qiskit} DIRECTORY)
    find_package(Python REQUIRED Interpreter Development)
endif ()

# Finally, we must define each test to be executed through `ctest` which we do
# by iterating over all discovered tests and registering it for execution.
foreach (test ${discovered_tests})
    # NOTE: the following enforces that every test filename contains its main
    # test logic in a function with the same name (minus the extension)
    get_filename_component (test_name ${test} NAME_WE)

    # The way that a test gets executed through the driver program is by
    # providing the test name as the only argument.
    add_test (
        NAME ${test_name}
        COMMAND test_driver ${test_name}
    )

    # add Python dll to the PATH of the tests
    if (MSVC)
        set_tests_properties(
            ${test_name} PROPERTIES ENVIRONMENT "PATH=%PATH%\;${qiskit_dll_dir}\;${Python_RUNTIME_LIBRARY_DIRS}"
        )
    endif ()
endforeach ()
