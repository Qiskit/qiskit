
# Creates a TARGET_NAME target for creating Pypi distributable package
# Args:
#    TARGET_NAME    The name of the target to be invoked in the Makefile
#    PACKAGE_TYPE   The type of distributable package we want to build. We have
#                   these options:
#						"sdist" => Source distribution package
#						"bdist_wheel" => Platform specific distribution package
#						"both" => Creates both sdist and bdist_wheel
function(add_pypi_package_target TARGET_NAME PACKAGE_TYPE)
	# Create Python distrubution package
	find_program(PYTHON "python")
	if (NOT PYTHON)
		message(FATAL_ERROR "We can't find Python in your system. Please, install it and try again...")
	endif()
	set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
	set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
	set(EXECUTABLE_FILE_EXTENSION "")

	message("QISKIT_VERSION = ${QISKIT_VERSION}")
	configure_file(${SETUP_PY_IN} ${SETUP_PY})

	if(PACKAGE_TYPE STREQUAL "both")
		set(PIP_PACKAGE_TYPES sdist --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist
			bdist_wheel --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	elseif(PACKAGE_TYPE STREQUAL "sdist")
		set(PIP_PACKAGE_TYPES sdist --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	elseif(PACKAGE_TYPE STREQUAL "bdist_wheel")
		set(PIP_PACKAGE_TYPES bdist_wheel --dist-dir ${CMAKE_CURRENT_BINARY_DIR}/dist)
	endif()


	if(UNIX AND NOT APPLE)
		#set(PIP_WHEEL_PLATFORM ${PACKAGE_TYPE} -p manylinux1_x86_64)
	elseif(MINGW)
		set(EXECUTABLE_FILE_EXTENSION ".exe")
		#set(PIP_WHEEL_PLATFORM "bdist_wheel") # TODO: Find the correct Tag
	elseif(APPLE)
		#set(PIP_WHEEL_PLATFORM "bdist_wheel") # TODO: Find the correct Tag
	endif()

	set(COPY_QISKIT_SIM_TARGET ${TARGET_NAME}_copy_qiskit_simulator)
	add_custom_target(${COPY_QISKIT_SIM_TARGET})
    add_custom_command(TARGET ${COPY_QISKIT_SIM_TARGET}
		COMMAND ${CMAKE_COMMAND} -E copy
			${QISKIT_SIMULATOR_OUTPUT_DIR}/qiskit_simulator${EXECUTABLE_FILE_EXTENSION}
			${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends)
	if(MINGW)
		foreach(dll_file ${QISKIT_SIMULATOR_THIRD_PARTY_DLLS})
			add_custom_command(TARGET ${COPY_QISKIT_SIM_TARGET}
				COMMAND ${CMAKE_COMMAND} -E copy
					${dll_file}
					${CMAKE_CURRENT_SOURCE_DIR}/qiskit/backends)
		endforeach()
	endif()

	add_custom_target(${TARGET_NAME})
	add_custom_command(TARGET ${TARGET_NAME}
		COMMAND ${PYTHON} ${SETUP_PY} ${PIP_PACKAGE_TYPES}
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	add_dependencies(${TARGET_NAME} ${COPY_QISKIT_SIM_TARGET})
	add_dependencies(${COPY_QISKIT_SIM_TARGET} qiskit_simulator)
endfunction()

