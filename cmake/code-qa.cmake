enable_testing()
# Run python code tests
# Create Python distrubution package
find_program(PYTHON "python")
if (NOT PYTHON)
message(FATAL_ERROR "We can't find Python in your system. Please, install it and try again.")
endif()

add_test(NAME qiskit_python
    COMMAND ${PYTHON} -m unittest discover -s test -v
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

function(add_linter_target)
    find_program(PYLINT "pylint")
    if (NOT PYLINT)
        message(FATAL_ERROR "We can't find pylint in yout system. Please, install it and try again.")
    endif()

    add_custom_target(linter)
    add_custom_command(TARGET linter
        COMMAND ${PYLINT} -rn qiskit test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

function(add_code_style_target)
    find_program(PYCODESTYLE "pycodestyle")
    if (NOT PYCODESTYLE)
        message(FATAL_ERROR "We can't find pycodestyle in yout system. Please, install it and try again.")
    endif()

    add_custom_target(style)
    add_custom_command(TARGET style
        COMMAND ${PYCODESTYLE} --exclude=qiskit/tools --max-line-length=100
            qiskit test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

