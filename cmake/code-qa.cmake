# Add targets for QA tasks:
#   lint        Target for invoking pyling
#   style       Target for invoking pycodestyle
#   coverage    Target for invoking coverage
#   doc         Target for invoking sphinx and producing the html docs
#
# If the dependencies for any of the targets cannot be found, they are skipped
# with a warning, as they are considered not essential (ie. qiskit can be
# executed and compiled even without them).

find_program(PYTHON "python")
if (NOT PYTHON)
    message(FATAL_ERROR "Couldn't find Python in your system. Please, install it and try again.")
endif()

# lint
function(add_lint_target)
    find_program(PYLINT "pylint")
    if (NOT PYLINT)
        message(WARNING "The 'lint' target will not be available: 'pylint' was not found.")
    else()
        add_custom_target(lint)
        add_custom_command(TARGET lint
            COMMAND ${PYLINT} -rn qiskit test
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
endfunction()

# style
function(add_code_style_target)
    find_program(PYCODESTYLE "pycodestyle")
    if (NOT PYCODESTYLE)
        message(WARNING "The 'style' target will not be available: 'pycodestyle' was not found.")
    else()
        add_custom_target(style)
        add_custom_command(TARGET style
            COMMAND ${PYCODESTYLE} --exclude=qiskit/tools --max-line-length=100
                qiskit test
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
endfunction()

# coverage
function(add_coverage_target)
    find_program(COV3 "coverage3")
    if (NOT COV3)
        message(WARNING "The 'coverage' target will not be available: 'coverage3' was not found.")
    else()
        add_custom_target(coverage_erase)
        add_custom_command(TARGET coverage_erase
            COMMAND ${COV3} erase
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        add_custom_target(coverage)
        add_custom_command(TARGET coverage
            COMMAND ${COV3} run --source qiskit -m unittest discover -s test -q
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        add_custom_command(TARGET coverage
            COMMAND ${COV3} report
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
endfunction()

# doc
function(add_doc_target DOC_FORMAT SOURCE_DIR BUILD_DIR)
    find_program(BETTER_APIDOC "better-apidoc")
    if(NOT BETTER_APIDOC)
        message(WARNING "The 'doc' target will not be available: 'better-apidoc' was not found.")
        return()
    endif()
    find_program(SPHINX_AUTOGEN "sphinx-autogen")
    if(NOT SPHINX_AUTOGEN)
        message(WARNING "The 'doc' target will not be available: 'sphinx-autogen' was not found.")
        return()
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
