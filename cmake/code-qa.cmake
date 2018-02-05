enable_testing()
# Run python code tests
# Create Python distrubution package
find_program(PYTHON "python")
if (NOT PYTHON)
    message(FATAL_ERROR "Couldn't find Python in your system. Please, install it and try again.")
endif()

add_test(NAME qiskit_python
    COMMAND ${PYTHON} -m unittest discover -s test -v
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

function(add_lint_target)
    find_program(PYLINT "pylint")
    if (NOT PYLINT)
        message(FATAL_ERROR "Couldn't find pylint in your system. Please, install it and try again.")
    endif()

    add_custom_target(lint)
    add_custom_command(TARGET lint
        COMMAND ${PYLINT} -rn qiskit test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

function(add_code_style_target)
    find_program(PYCODESTYLE "pycodestyle")
    if (NOT PYCODESTYLE)
        message(SEND_ERROR "Couldn't find pycodestyle in your system. Please, install it and try again.")
    endif()

    add_custom_target(style)
    add_custom_command(TARGET style
        COMMAND ${PYCODESTYLE} --exclude=qiskit/tools --max-line-length=100
            qiskit test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

function(add_doc_target DOC_FORMAT SOURCE_DIR BUILD_DIR)
    find_program(PYTHON python)
    if(NOT PYTHON)
        message(SEND_ERROR "Couldn't find python in your system. Please, install it and try again.")
    endif()
    find_program(BETTER_APIDOC "better-apidoc")
    if(NOT BETTER_APIDOC)
        message(SEND_ERROR "Couldn't find better-apidoc in your system. Please, install it and try again.")
    endif()
    find_program(SPHINX_AUTOGEN "sphinx-autogen")
    if(NOT SPHINX_AUTOGEN)
        message(SEND_ERROR "Couldn't find sphinx-autogen in your system. Please, install it and try again.")
    endif()

    set(SPHINX ${PYTHON} -msphinx)

    # Set param defaults
    if(NOT DOC_FORMAT)
        set(DOC_FORMAT "html")
    endif()
    if(NOT SOURCE_DIR)
        set(SOURCE_DIR "./")
    endif()
    if(NOT BUILD_DIR)
        set(BUILD_DIR "_build/")
    endif()

    add_custom_target(doc)
    foreach(lang_dir "." "ja")
        add_custom_command(TARGET doc
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
                ${BETTER_APIDOC} -f -o doc/${lang_dir}/_autodoc
                -d 5 -e -t doc/_templates/better-apidoc qiskit qiskit/tools
                "qiskit/extensions/standard/[a-z]*"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        add_custom_command(TARGET doc
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
                ${SPHINX_AUTOGEN} -t doc/_templates doc/${lang_dir}/_autodoc/*
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        add_custom_command(TARGET doc
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}
                ${SPHINX} -M ${DOC_FORMAT} "${SOURCE_DIR}${lang_dir}" "${BUILD_DIR}${lang_dir}"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc)
    endforeach()
endfunction()